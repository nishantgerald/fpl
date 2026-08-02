"""FPL squad rules, machine-checked.

This module is the single source of truth for what makes a squad legal. Every
rule the old LLM prompt asked for in English lives here as an assertion instead.

Pure: plain dicts in, plain values out. No I/O, no clock, no randomness.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

# element_type -> short position name. Type 5 ("MGR") exists in some seasons and
# is not part of a normal squad, but we map it so lookups never KeyError.
POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD", 5: "MGR"}

SQUAD_SIZE = 15

# Defaults, used only when bootstrap's element_types are unreadable. The engine
# prefers the upstream values so a rules change doesn't silently invalidate it.
DEFAULT_SQUAD_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
DEFAULT_MAX_PER_CLUB = 3

# Starting XI shape limits: exactly 1 GKP, and these ranges outfield.
FORMATION_LIMITS = {"GKP": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}

# Every legal (DEF, MID, FWD) combination summing to 10. Enumerated once at
# import so the best-XI search is a fixed eight-way loop rather than a solver.
LEGAL_FORMATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (d, m, f)
    for d in range(FORMATION_LIMITS["DEF"][0], FORMATION_LIMITS["DEF"][1] + 1)
    for m in range(FORMATION_LIMITS["MID"][0], FORMATION_LIMITS["MID"][1] + 1)
    for f in range(FORMATION_LIMITS["FWD"][0], FORMATION_LIMITS["FWD"][1] + 1)
    if d + m + f == 10
)

# Statuses that mean a player cannot be relied on to feature at all.
UNAVAILABLE_STATUSES = frozenset({"i", "s", "u", "n"})


def position_of(player: Mapping) -> str:
    """Short position name for a bootstrap element."""
    return POSITIONS.get(int(player.get("element_type", 0)), "UNK")


def squad_quotas(element_types: Sequence[Mapping] | None) -> dict[str, int]:
    """Squad composition quotas, read from bootstrap rather than hard-coded.

    ``element_types[].squad_select`` is FPL telling us how many of each position
    a squad must contain. Falling back to the documented defaults keeps the
    engine working if the field disappears.
    """
    if not element_types:
        return dict(DEFAULT_SQUAD_QUOTAS)
    quotas: dict[str, int] = {}
    for et in element_types:
        name = POSITIONS.get(int(et.get("id", 0)))
        select = et.get("squad_select")
        if name in ("GKP", "DEF", "MID", "FWD") and isinstance(select, int):
            quotas[name] = select
    for name, default in DEFAULT_SQUAD_QUOTAS.items():
        quotas.setdefault(name, default)
    return quotas


def max_per_club(game_settings: Mapping | None) -> int:
    """The 3-per-club cap, read from bootstrap's game settings where present."""
    if game_settings:
        value = game_settings.get("squad_team_limit")
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_MAX_PER_CLUB


def is_available(player: Mapping, doubt_threshold: int = 75) -> bool:
    """Whether a player is safe to *transfer in*.

    Injured, suspended, unavailable and 'not in squad' are hard nos. Doubtful
    players are allowed through only above ``doubt_threshold`` percent, because
    recommending a 25%-chance player is recommending a blank.
    """
    status = str(player.get("status", "a"))
    if status in UNAVAILABLE_STATUSES:
        return False
    if status == "d":
        chance = player.get("chance_of_playing_next_round")
        if chance is None:
            return False
        return int(chance) >= doubt_threshold
    return True


def position_counts(squad: Iterable[Mapping]) -> dict[str, int]:
    counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in squad:
        pos = position_of(p)
        if pos in counts:
            counts[pos] += 1
    return counts


def club_counts(squad: Iterable[Mapping]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for p in squad:
        team = int(p.get("team", 0))
        counts[team] = counts.get(team, 0) + 1
    return counts


def check_squad(
    squad: Sequence[Mapping],
    quotas: Mapping[str, int] | None = None,
    club_limit: int = DEFAULT_MAX_PER_CLUB,
) -> dict[str, bool]:
    """Validate a 15-player squad against every composition rule.

    Returns a per-rule verdict dict rather than a bare bool — the caller renders
    it in the UI as the engine's proof of work, and the tests assert on it.
    """
    quotas = dict(quotas or DEFAULT_SQUAD_QUOTAS)
    counts = position_counts(squad)
    clubs = club_counts(squad)

    size_ok = len(squad) == sum(quotas.values())
    quotas_ok = all(counts.get(pos, 0) == n for pos, n in quotas.items())
    clubs_ok = all(n <= club_limit for n in clubs.values())
    unique_ok = len({int(p["id"]) for p in squad}) == len(squad)
    formation_ok = can_field_legal_xi(counts)

    return {
        "squad_size_ok": size_ok,
        "position_quotas_ok": quotas_ok,
        "club_limit_ok": clubs_ok,
        "unique_players_ok": unique_ok,
        "formation_ok": formation_ok,
        "all_ok": size_ok and quotas_ok and clubs_ok and unique_ok and formation_ok,
    }


def can_field_legal_xi(counts: Mapping[str, int]) -> bool:
    """Whether a squad with these position counts admits at least one legal XI."""
    if counts.get("GKP", 0) < 1:
        return False
    return any(
        counts.get("DEF", 0) >= d and counts.get("MID", 0) >= m and counts.get("FWD", 0) >= f
        for d, m, f in LEGAL_FORMATIONS
    )


def best_xi(
    squad: Sequence[Mapping],
    scores: Mapping[int, float],
) -> tuple[list[int], tuple[int, int, int], float]:
    """Pick the highest-scoring legal starting XI.

    ``scores`` maps player id -> points for the gameweek in question. Returns
    ``(player_ids, (def, mid, fwd), total)``. Bench contributes nothing.

    Every legal formation is evaluated and the best taken. Within a position,
    players are sorted by ``(-score, id)`` so ties break stably and the whole
    function is deterministic.
    """
    by_pos: dict[str, list[int]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for p in squad:
        pos = position_of(p)
        if pos in by_pos:
            by_pos[pos].append(int(p["id"]))
    for pos in by_pos:
        by_pos[pos].sort(key=lambda pid: (-scores.get(pid, 0.0), pid))

    if not by_pos["GKP"]:
        return [], (0, 0, 0), 0.0

    keeper = by_pos["GKP"][0]
    keeper_score = scores.get(keeper, 0.0)

    best_total = float("-inf")
    best_ids: list[int] = []
    best_shape = (0, 0, 0)

    for d, m, f in LEGAL_FORMATIONS:
        if len(by_pos["DEF"]) < d or len(by_pos["MID"]) < m or len(by_pos["FWD"]) < f:
            continue
        picked = by_pos["DEF"][:d] + by_pos["MID"][:m] + by_pos["FWD"][:f]
        total = keeper_score + sum(scores.get(pid, 0.0) for pid in picked)
        # Tie-break on shape so the result is stable across runs.
        if total > best_total or (total == best_total and (d, m, f) < best_shape):
            best_total = total
            best_ids = [keeper] + picked
            best_shape = (d, m, f)

    if not best_ids:
        return [], (0, 0, 0), 0.0
    return best_ids, best_shape, best_total


def best_xi_with_captain(
    squad: Sequence[Mapping],
    scores: Mapping[int, float],
) -> tuple[list[int], tuple[int, int, int], float, int | None]:
    """Best XI plus the captain's doubled contribution.

    The captain is the highest scorer in the XI, which is re-chosen every
    gameweek — correct, because a real manager re-picks the armband weekly.
    """
    xi, shape, total = best_xi(squad, scores)
    if not xi:
        return [], shape, 0.0, None
    captain = max(xi, key=lambda pid: (scores.get(pid, 0.0), -pid))
    return xi, shape, total + scores.get(captain, 0.0), captain


def formation_str(shape: tuple[int, int, int]) -> str:
    return "-".join(str(n) for n in shape)


def hit_cost(n_transfers: int, free_transfers: int, points_per_hit: int = 4) -> int:
    """Points cost of making ``n_transfers`` with ``free_transfers`` banked."""
    return points_per_hit * max(0, int(n_transfers) - int(free_transfers))
