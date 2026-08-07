"""Expected points (xPts) — the projection that replaces FCPS.

FCPS was a unitless 0-1000 blend of total points, form, a 3-fixture FDR sum and
ICT. It had no minutes term, normalised goalkeepers against Salah-class ceilings,
and divided FDR by a hard-coded 15 — so a *blank* gameweek improved a player's
score. Most importantly it wasn't in points, so it could never answer the only
question that matters: "is this transfer worth a -4?"

This module answers in points. For player ``p`` in gameweek ``g``, summed over
each fixture the player's team actually plays in ``g`` (zero for a blank, two for
a double)::

    xPts = availability x SUM_fixtures [ appearance + goals + assists
                                       + clean_sheet + conceded + saves
                                       + defcon + bonus + cards ]

Every term is returned alongside the total, so the UI can show a waterfall and
the tests can assert the total *is* the sum of its parts rather than a number
with a story attached.

Pure: bootstrap and fixture dicts in, numbers out. No I/O, no clock.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .rules import POSITIONS, UNAVAILABLE_STATUSES, position_of

# Points per goal by position, and per clean sheet by position.
GOAL_POINTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
CLEAN_SHEET_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
ASSIST_POINTS = 3

# Defensive-contribution thresholds (2 pts on reaching them).
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12, "GKP": 0}
DEFCON_POINTS = 2

# How much weight FPL's own ep_next carries in the immediate gameweek. Our model
# is better over a horizon; ep_next is a useful anchor when our inputs are thin.
#
# Pre-season it is deliberately *light*. FPL's published estimate is heavily
# compressed at that point — Haaland at £15.5m and Bruno at £12.0m both read
# exactly 4.0, and 570 players share 24 distinct values — so it separates the
# good from the great barely at all. Leaning on it pulls the whole distribution
# toward the middle and costs the premiums most, which is the opposite of
# useful. It earns its place as a sanity anchor, not as a co-model.
EP_NEXT_WEIGHT = 0.35
EP_NEXT_WEIGHT_EARLY = 0.30
EARLY_SEASON_GAMES = 3

# How fast the ep_next anchor fades across the horizon.
#
# It used to not fade at all — it applied to the first gameweek and then
# vanished, so GW1 and GW2 were produced by materially different models and the
# step between them was an artefact of a weight hitting zero rather than
# anything about the fixtures. Between seasons that step was ~28%.
#
# Fading matters most before a ball is kicked, because every per-90 rate we hold
# then describes *last* season. For a player who changed clubs it describes a
# different job at a different club: Isak's 694 minutes across an interrupted
# campaign said "bit-part squad player" while his price said "first-choice
# striker at Liverpool". ep_next is the only input we have that reflects the
# summer, so pre-season it stays in the mix across the whole horizon.
EP_NEXT_DECAY = 0.65
EP_NEXT_DECAY_PRESEASON = 0.50

# Fixture multipliers are clamped so one extreme strength rating can't dominate.
FIXTURE_MULT_MIN = 0.6
FIXTURE_MULT_MAX = 1.6
HOME_ATTACK_BONUS = 1.08
HOME_DEFENCE_BONUS = 0.94

# A knock this week says little about four weeks out.
AVAILABILITY_RECOVERY = 0.35

# Between seasons, minutes describe a job the player may no longer hold.
#
# FPL prices a player for the role it expects them to play *this* season, which
# is the one thing last season's minutes cannot know. Isak's 694 minutes across
# an interrupted campaign read as "bit-part squad player"; his £9.0m price reads
# as "first-choice striker at Liverpool". The price is right and the minutes are
# stale, so pre-season the two are blended in proportion to how much football
# the minutes actually represent.
PRICE_FLOOR = 40  # tenths — FPL's cheapest player
PRICE_ROLE_SPAN = 30  # tenths above the floor at which a player reads as nailed
PRESEASON_EVIDENCE_MINUTES = 2200.0  # a full season of starts; below this we shrink
STARTER_MINUTES = 80.0

# Per-90 rates are only as trustworthy as the minutes underneath them.
# `SHRINK_PSEUDO_MINUTES` is the point at which an observed rate and the
# positional prior carry equal weight; `RELIABLE_MINUTES` is the bar a player
# must clear to help *set* those priors.
SHRINK_PSEUDO_MINUTES = 600.0
RELIABLE_MINUTES = 900.0

# Bonus points regress hard; a player's season bonus rate over-fits small samples.
BONUS_SHRINK = 0.85

# Default assumption for a doubtful player with no published percentage.
DEFAULT_DOUBT_CHANCE = 0.5


def _f(value, default: float = 0.0) -> float:
    """Coerce an FPL field to float. Many are strings; some are None."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_fixture_index(
    fixtures: Sequence[Mapping],
    from_gameweek: int,
) -> dict[int, dict[int, list[dict]]]:
    """Index upcoming fixtures as ``team_id -> gameweek -> [fixture, ...]``.

    A list per gameweek is what makes blanks and doubles fall out correctly: a
    blank is an empty list (or a missing key), a double has two entries. The old
    ``next_3_fdr`` sum could represent neither.
    """
    index: dict[int, dict[int, list[dict]]] = {}
    for fx in fixtures:
        event = fx.get("event")
        if event is None or int(event) < from_gameweek:
            continue
        if fx.get("finished"):
            continue
        event = int(event)
        home, away = int(fx["team_h"]), int(fx["team_a"])
        index.setdefault(home, {}).setdefault(event, []).append(
            {
                "opponent": away,
                "home": True,
                "fdr": int(fx.get("team_h_difficulty", 3)),
            }
        )
        index.setdefault(away, {}).setdefault(event, []).append(
            {
                "opponent": home,
                "home": False,
                "fdr": int(fx.get("team_a_difficulty", 3)),
            }
        )
    return index


def build_team_index(teams: Sequence[Mapping]) -> dict[int, dict]:
    """Index teams by id, with league-average strengths for normalisation."""
    by_id = {int(t["id"]): t for t in teams}
    if not by_id:
        return {}

    def avg(key: str) -> float:
        values = [_f(t.get(key)) for t in by_id.values()]
        values = [v for v in values if v > 0]
        return sum(values) / len(values) if values else 1.0

    league = {
        "attack_home": avg("strength_attack_home"),
        "attack_away": avg("strength_attack_away"),
        "defence_home": avg("strength_defence_home"),
        "defence_away": avg("strength_defence_away"),
    }
    for t in by_id.values():
        t["_league"] = league
    return by_id


def _fdr_multipliers(fixture: Mapping) -> tuple[float, float]:
    """``(attacking, conceding)`` from FDR alone: 1 (easy) -> 1.3x attack, 5 -> 0.7x."""
    fdr = int(fixture.get("fdr", 3))
    att = 1.3 - 0.15 * (fdr - 1)
    con = 0.7 + 0.15 * (fdr - 1)
    return _clamp(att, FIXTURE_MULT_MIN, FIXTURE_MULT_MAX), _clamp(
        con, FIXTURE_MULT_MIN, FIXTURE_MULT_MAX
    )


def _fixture_multipliers(
    fixture: Mapping,
    teams: Mapping[int, Mapping],
) -> tuple[float, float]:
    """``(attacking, conceding)`` multipliers for one fixture.

    Attacking scales inversely with the opponent's defensive strength; conceding
    scales with their attacking strength. Falls back to FDR when strength
    ratings are missing, so the model still works on a thin payload.
    """
    opponent = teams.get(int(fixture["opponent"]))
    at_home = bool(fixture["home"])

    if not opponent or "_league" not in opponent:
        return _fdr_multipliers(fixture)

    league = opponent["_league"]
    # The opponent defends away when we're at home, and vice versa.
    opp_def = _f(
        opponent.get("strength_defence_away" if at_home else "strength_defence_home")
    )
    opp_att = _f(
        opponent.get("strength_attack_away" if at_home else "strength_attack_home")
    )
    league_def = league["defence_away"] if at_home else league["defence_home"]
    league_att = league["attack_away"] if at_home else league["attack_home"]

    # A team with no usable ratings must fall back, not limp on with a half-
    # computed pair. Dividing by a zero attack rating used to yield con == 0,
    # which the clamp then floored to FIXTURE_MULT_MIN -- silently declaring the
    # opponent the softest attack in the league instead of admitting ignorance.
    if opp_def <= 0 or opp_att <= 0 or league_def <= 0 or league_att <= 0:
        return _fdr_multipliers(fixture)

    att = league_def / opp_def
    con = opp_att / league_att

    if at_home:
        att *= HOME_ATTACK_BONUS
        con *= HOME_DEFENCE_BONUS

    return _clamp(att, FIXTURE_MULT_MIN, FIXTURE_MULT_MAX), _clamp(
        con, FIXTURE_MULT_MIN, FIXTURE_MULT_MAX
    )


def fixture_multipliers(
    fixture: Mapping, teams: Mapping[int, Mapping]
) -> tuple[float, float]:
    """Public alias — the fixture ticker uses the same definition the model does.

    One definition, two consumers, so the ticker and the projections can never
    disagree about how hard a fixture is.
    """
    return _fixture_multipliers(fixture, teams)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def defcon_probability(dc90: float, minutes_fraction: float, threshold: int) -> float:
    """P(defensive contributions reach ``threshold``) in one appearance.

    The previous form was ``mean / threshold``, which is not a probability. It
    said a defender averaging exactly 10 actions reached 10 in *every* match,
    when the true answer is about half of them, and it credited a midfielder
    averaging 6 with a 50% chance of reaching 12 — an outcome that is very
    nearly impossible. Defenders and midfielders were inflated by roughly half a
    point per gameweek each as a result.

    Counts of tackles, interceptions and clearances over a match are modelled
    here as Poisson. That is not exactly right — real defensive actions are
    overdispersed, so the true tail is slightly fatter than this and the
    estimate is mildly conservative for players below the threshold — but it is
    a principled distribution with the right shape, and the error is a fraction
    of the one it replaces.
    """
    lam = dc90 * minutes_fraction
    if lam <= 0 or threshold <= 0:
        return 0.0
    # P(X >= k) = 1 - P(X <= k-1), summed directly: k is at most 12, so the
    # factorials stay small and exact.
    below = sum(
        math.exp(-lam) * lam**i / math.factorial(i) for i in range(threshold)
    )
    return _clamp(1.0 - below, 0.0, 1.0)


def availability_factor(player: Mapping, gameweeks_ahead: int = 0) -> float:
    """Probability the player is fit to feature, 0..1.

    Hard zero for injured/suspended/unavailable. Doubtful players are scaled by
    their published chance. The penalty relaxes toward 1.0 for later gameweeks,
    because a knock now says little about a month from now — but a player who is
    flat-out unavailable stays at zero throughout, since we have no recovery date.
    """
    status = str(player.get("status", "a"))
    if status in UNAVAILABLE_STATUSES:
        return 0.0
    if status != "d":
        return 1.0

    chance = player.get("chance_of_playing_next_round")
    base = DEFAULT_DOUBT_CHANCE if chance is None else _clamp(_f(chance) / 100.0, 0.0, 1.0)
    if gameweeks_ahead <= 0:
        return base
    recovered = 1.0 - (1.0 - base) * ((1.0 - AVAILABILITY_RECOVERY) ** gameweeks_ahead)
    return _clamp(recovered, 0.0, 1.0)


COMPLETED_SEASON_GAMES = 38


def effective_team_games(team_games: int) -> float:
    """Games to divide season totals by, tolerating a reset league table.

    Between seasons the API reports zero games played while the element totals
    still describe the completed 38-match campaign. Treating that zero as "no
    games" produced a projection with no football in it: ``minutes_profile``
    returned a zero play-probability, which zeroed appearance, goals, assists
    and clean sheets, while ``bonus_pg`` divided by ``max(1.0, 0)`` and passed
    a whole season's bonus haul through as a single gameweek's. Haaland came out
    at 34.55 expected points a gameweek, composed entirely of 36.55 bonus and
    -2.0 cards.

    A completed season is the right denominator for a completed season's
    totals, so pre-season projections become "last year, adjusted for
    fixtures" — which is exactly what they should be before anyone has kicked
    a ball.
    """
    return float(team_games) if team_games and team_games > 0 else float(
        COMPLETED_SEASON_GAMES
    )


def position_rate_priors(
    elements: Sequence[Mapping],
) -> dict[str, dict[str, float]]:
    """Median per-90 rate per position, over players with real minutes.

    The prior that :func:`shrink_rate` regresses toward. Computed from players
    with at least :data:`RELIABLE_MINUTES` so the priors themselves aren't
    poisoned by the small samples they exist to correct.
    """
    keys = (
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goals_conceded_per_90",
        "saves_per_90",
        "defensive_contribution_per_90",
    )
    buckets: dict[str, dict[str, list[float]]] = {}
    for element in elements:
        if _f(element.get("minutes")) < RELIABLE_MINUTES:
            continue
        position = position_of(element)
        for key in keys:
            value = _f(element.get(key))
            if value > 0:
                buckets.setdefault(position, {}).setdefault(key, []).append(value)

    priors: dict[str, dict[str, float]] = {}
    for position, by_key in buckets.items():
        for key, values in by_key.items():
            values.sort()
            priors.setdefault(position, {})[key] = values[len(values) // 2]
    return priors


def shrink_rate(rate: float, minutes: float, prior: float) -> float:
    """Regress a per-90 rate toward ``prior`` when few minutes back it.

    FPL computes per-90 fields by dividing by actual minutes, so a player with
    two minutes on record can carry an expected-goals rate of 3.6 per 90 and a
    defensive-contribution rate of 45. Taken literally — as they were — that
    made a £5.0m substitute the third-best captain in the league and a lock for
    the squad builder. The rate is not evidence; the denominator is noise.

    Weight rises with minutes played, so a full season is essentially unshrunk
    and a cameo is essentially the positional median.
    """
    if minutes <= 0:
        return prior
    weight = minutes / (minutes + SHRINK_PSEUDO_MINUTES)
    return weight * rate + (1.0 - weight) * prior


def role_prior_from_price(now_cost: float) -> float:
    """Start probability implied by FPL's price alone.

    Squad filler sits at the floor; anyone priced well above it is priced that
    way because FPL expects them to play. Clamped at both ends because price is
    a coarse signal — it should nudge a stale estimate, not overwrite it.
    """
    return _clamp((now_cost - PRICE_FLOOR) / PRICE_ROLE_SPAN, 0.15, 0.90)


# Ownership bands for the market check below. Wide enough that each holds a
# meaningful sample, narrow enough that a £4.5m bench defender is not compared
# against a £13m striker.
OWNERSHIP_BANDS: tuple[tuple[int, int], ...] = (
    (0, 45), (45, 50), (50, 55), (55, 65), (65, 75), (75, 90), (90, 10_000)
)

# How far a price prior can be pulled down when the market disagrees with it.
# Not to zero: ownership is a crowd opinion, not a team sheet, and a genuine
# differential should be marked uncertain rather than declared a non-player.
MARKET_FLOOR = 0.35


def ownership_baseline(elements: Sequence[Mapping]) -> dict[tuple[int, int], float]:
    """Median ownership per price band — what a price *normally* attracts.

    Used to sanity-check the price prior. Computed per band rather than as one
    global median because ownership rises steeply with price: a 1.4% budget
    defender is popular for his price, while a 1.4% premium forward is being
    conspicuously avoided, and an absolute threshold cannot tell those apart.
    """
    baseline: dict[tuple[int, int], float] = {}
    for band in OWNERSHIP_BANDS:
        low, high = band
        owned = sorted(
            _f(e.get("selected_by_percent"))
            for e in elements
            # 'u' is unavailable — transferred out of the league, or similar.
            # Including them would drag every median toward zero.
            if low <= int(e.get("now_cost") or 0) < high and e.get("status") != "u"
        )
        baseline[band] = owned[len(owned) // 2] if owned else 0.0
    return baseline


def _band_for(now_cost: int) -> tuple[int, int]:
    for band in OWNERSHIP_BANDS:
        if band[0] <= now_cost < band[1]:
            return band
    return OWNERSHIP_BANDS[-1]


def market_agreement(
    player: Mapping, baseline: Mapping[tuple[int, int], float] | None
) -> float:
    """How far three million managers agree with what the price implies.

    The price prior asserts "FPL priced him to start". Ownership is the check on
    that: if a player is priced like a starter and almost nobody has picked him,
    the people who follow the club have concluded he is not starting, and the
    price is stale or aspirational.

    This is the signal that catches Nicolas Jackson — £6.5m, which the price
    prior alone reads as nailed, against 0.4% ownership when the median at that
    price is 2.1%. His own club-mate João Pedro is £7.5m and 54% owned, which is
    what an actually-nailed forward looks like.

    Returns 1.0 when the market agrees, down to MARKET_FLOOR when it does not.
    """
    if not baseline:
        return 1.0
    median = baseline.get(_band_for(int(player.get("now_cost") or 0)), 0.0)
    if median <= 0:
        # No comparable players to judge against; do not invent a correction.
        return 1.0
    ratio = _f(player.get("selected_by_percent")) / median
    return _clamp(MARKET_FLOOR + (1.0 - MARKET_FLOOR) * min(ratio, 1.0), MARKET_FLOOR, 1.0)


def minutes_profile(
    player: Mapping,
    team_games: int,
    ownership: Mapping[tuple[int, int], float] | None = None,
) -> dict[str, float]:
    """Expected minutes, start probability and 60-minute probability.

    This is the term FCPS omitted entirely, and it is the single largest driver
    of FPL returns — a hot-form 25-minute substitute is not a better pick than a
    nailed starter, however good his per-90 numbers look.

    Pre-season the minutes on record belong to last season, so they are shrunk
    toward the role FPL's price implies, weighted by how much football they
    actually represent. A player with a full season behind him is unaffected;
    one with a fraction of a season is mostly priced.
    """
    preseason = int(team_games or 0) < EARLY_SEASON_GAMES
    team_games = effective_team_games(team_games)
    if team_games <= 0:
        return {
            "p_start": 0.0, "p_play": 0.0, "p_60": 0.0,
            "minutes": 0.0, "evidence": 0.0,
        }

    starts = _f(player.get("starts"))
    minutes = _f(player.get("minutes"))

    p_start = _clamp(starts / team_games, 0.0, 1.0)
    mins_pg = minutes / team_games

    evidence = 1.0
    if preseason:
        evidence = _clamp(minutes / PRESEASON_EVIDENCE_MINUTES, 0.0, 1.0)
        # Price says what FPL expects; ownership says whether anyone believes
        # it. Applied to the prior only — once real minutes exist they are the
        # better evidence and this correction fades with `evidence`.
        prior = role_prior_from_price(_f(player.get("now_cost")))
        prior *= market_agreement(player, ownership)
        p_start = evidence * p_start + (1.0 - evidence) * prior
        # Minutes have to follow the promotion, or a player we now expect to
        # start would still be scored as though he watched from the bench.
        mins_pg = max(mins_pg, p_start * STARTER_MINUTES)

    p_60 = _clamp(mins_pg / 75.0, 0.0, 1.0)

    # Someone with minutes but few starts is a regular substitute; give them a
    # partial appearance probability rather than treating them as a non-player.
    sub_rate = _clamp((mins_pg - p_start * 90.0) / 30.0, 0.0, 1.0)
    p_play = _clamp(p_start + (1.0 - p_start) * sub_rate, 0.0, 1.0)

    return {
        "p_start": p_start,
        "p_play": p_play,
        "p_60": p_60,
        "minutes": mins_pg,
        # How much real football is behind the numbers above, 0..1. Carried out
        # so the UI can distinguish an estimate from an observation.
        "evidence": evidence,
    }


def minutes_risk(profile: Mapping[str, float]) -> str:
    """Coarse risk band. Unchanged semantics — several callers test == "high"."""
    p_start = profile.get("p_start", 0.0)
    if p_start >= 0.75:
        return "low"
    if p_start >= 0.45:
        return "medium"
    return "high"


def minutes_basis(profile: Mapping[str, float]) -> str:
    """Whether we have watched this player play, or are inferring it.

    The risk band alone was being rendered to users as the word "Nailed", which
    asserts a certainty the model does not have: before a ball is kicked there
    are no minutes on record, so the estimate is a price tag corrected by a
    crowd opinion. That may well be right, but it is an inference, and a screen
    that prints it identically to thirty observed starts is lying by omission.

    "observed" once there are real minutes behind it, "estimated" until then.
    """
    return "observed" if profile.get("evidence", 0.0) >= 0.25 else "estimated"


def _per_90(player: Mapping, per90_key: str, total_key: str, team_games: int) -> float:
    """Prefer FPL's published per-90 rate; fall back to a season average.

    Early in a season the per-90 fields can be absent or zero, and a fallback
    keeps the model from silently projecting everyone at zero.
    """
    rate = _f(player.get(per90_key))
    if rate > 0:
        return rate
    team_games = effective_team_games(team_games)
    if team_games <= 0:
        return 0.0
    minutes = _f(player.get("minutes"))
    if minutes <= 0:
        return 0.0
    return _f(player.get(total_key)) / minutes * 90.0


def project_player(
    player: Mapping,
    gameweeks: Sequence[int],
    fixture_index: Mapping[int, Mapping[int, Sequence[Mapping]]],
    teams: Mapping[int, Mapping],
    team_games: int,
    rate_priors: Mapping[str, Mapping[str, float]] | None = None,
    ownership: Mapping[tuple[int, int], float] | None = None,
) -> dict:
    """Project one player over ``gameweeks``.

    Returns the per-gameweek breakdown, the horizon total, and enough context
    (availability, minutes risk, value per million) for the UI and the optimiser
    to work without recomputing anything.
    """
    position = position_of(player)
    team_id = int(player.get("team", 0))
    team_fixtures = fixture_index.get(team_id, {})

    profile = minutes_profile(player, team_games, ownership)
    mins_frac = _clamp(profile["minutes"] / 90.0, 0.0, 1.0)

    xg90 = _per_90(player, "expected_goals_per_90", "expected_goals", team_games)
    xa90 = _per_90(player, "expected_assists_per_90", "expected_assists", team_games)
    xgc90 = _per_90(
        player, "expected_goals_conceded_per_90", "expected_goals_conceded", team_games
    )
    saves90 = _per_90(player, "saves_per_90", "saves", team_games)
    dc90 = _per_90(
        player,
        "defensive_contribution_per_90",
        "defensive_contribution",
        team_games,
    )

    # Underlying stats can be missing entirely in early seasons; fall back to
    # actual returns so the model degrades rather than zeroing everyone out.
    if xg90 <= 0 and team_games > 0 and profile["minutes"] > 0:
        xg90 = _f(player.get("goals_scored")) / team_games / max(mins_frac, 0.01)
    if xa90 <= 0 and team_games > 0 and profile["minutes"] > 0:
        xa90 = _f(player.get("assists")) / team_games / max(mins_frac, 0.01)
    if xgc90 <= 0 and team_games > 0 and profile["minutes"] > 0:
        xgc90 = _f(player.get("goals_conceded")) / team_games / max(mins_frac, 0.01)

    # Every rate above is FPL's own total divided by minutes played, so a player
    # with a cameo on record carries rates that are arithmetic, not evidence.
    # Regress them toward the positional median in proportion to the minutes
    # behind them.
    if rate_priors:
        played = _f(player.get("minutes"))
        priors = rate_priors.get(position, {})
        xg90 = shrink_rate(xg90, played, priors.get("expected_goals_per_90", 0.0))
        xa90 = shrink_rate(xa90, played, priors.get("expected_assists_per_90", 0.0))
        xgc90 = shrink_rate(
            xgc90, played, priors.get("expected_goals_conceded_per_90", xgc90)
        )
        saves90 = shrink_rate(saves90, played, priors.get("saves_per_90", 0.0))
        dc90 = shrink_rate(
            dc90, played, priors.get("defensive_contribution_per_90", 0.0)
        )

    games_played = effective_team_games(team_games)
    bonus_pg = _f(player.get("bonus")) / games_played * BONUS_SHRINK
    yellow_pg = _f(player.get("yellow_cards")) / games_played
    red_pg = _f(player.get("red_cards")) / games_played

    per_gameweek: list[dict] = []
    for offset, gw in enumerate(gameweeks):
        fixtures = list(team_fixtures.get(gw, []))
        avail = availability_factor(player, offset)

        components = {
            "appearance": 0.0,
            "goals": 0.0,
            "assists": 0.0,
            "clean_sheet": 0.0,
            "conceded": 0.0,
            "saves": 0.0,
            "bonus": 0.0,
            "defensive_contribution": 0.0,
            "cards": 0.0,
        }

        for fixture in fixtures:
            att_mult, con_mult = _fixture_multipliers(fixture, teams)

            components["appearance"] += profile["p_play"] + profile["p_60"]
            components["goals"] += (
                xg90 * mins_frac * att_mult * GOAL_POINTS.get(position, 4)
            )
            components["assists"] += xa90 * mins_frac * att_mult * ASSIST_POINTS

            lam = xgc90 * mins_frac * con_mult
            cs_points = CLEAN_SHEET_POINTS.get(position, 0)
            if cs_points:
                # Clean sheet points need 60 minutes, hence the p_60 scaling.
                components["clean_sheet"] += (
                    math.exp(-lam) * cs_points * profile["p_60"]
                )
            if position in ("GKP", "DEF"):
                components["conceded"] -= lam / 2.0
            if position == "GKP":
                components["saves"] += saves90 * mins_frac / 3.0

            threshold = DEFCON_THRESHOLD.get(position, 0)
            if threshold and dc90 > 0:
                p_defcon = defcon_probability(dc90, mins_frac, threshold)
                # Reaching the threshold requires being on the pitch to do it.
                components["defensive_contribution"] += (
                    p_defcon * DEFCON_POINTS * profile["p_play"]
                )

            components["bonus"] += bonus_pg
            components["cards"] -= yellow_pg + 3.0 * red_pg

        # Blend at full strength, then apply availability to the result. Doing
        # it the other way round would let ep_next leak points through for a
        # player we already know is injured, because FPL's own estimate may not
        # have caught up with the news.
        model_total = sum(components.values())
        blended, components = _blend_with_ep_next(
            player, model_total, components, team_games, bool(fixtures), offset
        )

        blended *= avail
        components = {k: avail * v for k, v in components.items()}

        per_gameweek.append(
            {
                "gameweek": gw,
                "xpts": round(blended, 2),
                "fixtures": [
                    {
                        "opponent": _team_short(teams, fx["opponent"]),
                        "home": fx["home"],
                        "fdr": fx["fdr"],
                    }
                    for fx in fixtures
                ],
                "components": {k: round(v, 3) for k, v in components.items()},
            }
        )

    horizon = round(sum(g["xpts"] for g in per_gameweek), 2)
    price = int(player.get("now_cost", 0)) or 1

    return {
        "player_id": int(player["id"]),
        "horizon_xpts": horizon,
        "xpts_next": per_gameweek[0]["xpts"] if per_gameweek else 0.0,
        "per_gameweek": per_gameweek,
        "availability": round(availability_factor(player, 0), 3),
        "minutes_risk": minutes_risk(profile),
        "minutes_basis": minutes_basis(profile),
        "p_start": round(profile["p_start"], 3),
        "xpts_per_million": round(horizon / (price / 10.0), 3),
    }


def ep_next_weight(team_games: int, offset: int) -> float:
    """How much ``ep_next`` counts for the gameweek ``offset`` weeks out.

    Decays geometrically with distance rather than being switched off after the
    first gameweek, so consecutive gameweeks are produced by almost the same
    model and any step between them comes from the fixtures.
    """
    preseason = team_games < EARLY_SEASON_GAMES
    base = EP_NEXT_WEIGHT_EARLY if preseason else EP_NEXT_WEIGHT
    decay = EP_NEXT_DECAY_PRESEASON if preseason else EP_NEXT_DECAY
    return base * (decay**offset)


def _blend_with_ep_next(
    player: Mapping,
    model_total: float,
    components: dict[str, float],
    team_games: int,
    has_fixture: bool,
    offset: int = 0,
) -> tuple[float, dict[str, float]]:
    """Blend one gameweek of the model with FPL's published ``ep_next``.

    ``ep_next`` is a free, independently derived signal. It matters most early in
    a season, when per-90 rates computed over two games are noise — and between
    seasons, when every rate we hold describes a squad that has since changed.

    A blank gameweek is never blended — ``ep_next`` doesn't know the player isn't
    playing, and letting it leak in would reintroduce exactly the bug that made
    FCPS score blanks as easy fixtures.
    """
    if not has_fixture:
        return 0.0, {k: 0.0 for k in components}

    ep_next = _f(player.get("ep_next"))
    if ep_next <= 0:
        return model_total, components

    weight = ep_next_weight(team_games, offset)
    blended = (1.0 - weight) * model_total + weight * ep_next

    # Rescale the components so they still sum to the blended total — the
    # decomposition has to *be* the number, not a plausible story next to it.
    if model_total > 0:
        scale = blended / model_total
        components = {k: v * scale for k, v in components.items()}
    else:
        # Nothing to rescale (a player with no minutes on record). Attribute the
        # whole blended figure to appearance rather than leaving stray component
        # values that wouldn't sum to the total.
        components = {k: 0.0 for k in components}
        components["appearance"] = blended

    return blended, components


def _team_short(teams: Mapping[int, Mapping], team_id: int) -> str:
    team = teams.get(int(team_id))
    return str(team.get("short_name", "UNK")) if team else "UNK"


def team_games_played(teams: Sequence[Mapping], events: Sequence[Mapping]) -> int:
    """How many league games each team has played, on average.

    Used to turn season totals into per-game rates. Prefers the ``played`` field
    on teams; falls back to counting finished events.
    """
    played = [int(t.get("played", 0) or 0) for t in teams]
    played = [p for p in played if p > 0]
    if played:
        return max(1, round(sum(played) / len(played)))
    finished = sum(1 for e in events if e.get("finished"))
    return max(0, finished)


def project_all(
    elements: Sequence[Mapping],
    fixtures: Sequence[Mapping],
    teams: Sequence[Mapping],
    events: Sequence[Mapping],
    from_gameweek: int,
    horizon: int = 5,
) -> dict[int, dict]:
    """Project every player over the horizon. Returns ``player_id -> projection``."""
    gameweeks = _horizon_gameweeks(events, from_gameweek, horizon)
    fixture_index = build_fixture_index(fixtures, from_gameweek)
    team_index = build_team_index(teams)
    games = team_games_played(teams, events)
    priors = position_rate_priors(elements)
    ownership = ownership_baseline(elements)

    return {
        int(p["id"]): project_player(
            p, gameweeks, fixture_index, team_index, games, priors, ownership
        )
        for p in elements
        if POSITIONS.get(int(p.get("element_type", 0))) in ("GKP", "DEF", "MID", "FWD")
    }


def _horizon_gameweeks(
    events: Sequence[Mapping], from_gameweek: int, horizon: int
) -> list[int]:
    """The next ``horizon`` gameweek numbers, respecting the real event list."""
    ids = sorted(
        int(e["id"]) for e in events if int(e.get("id", 0)) >= from_gameweek
    )
    if not ids:
        return list(range(from_gameweek, from_gameweek + horizon))
    return ids[:horizon]
