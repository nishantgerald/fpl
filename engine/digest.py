"""The pre-deadline briefing: one message, one decision at a time.

Everything else in this app is pull — the manager has to remember to open it.
This is the one thing that is push, and that changes what belongs in it. A
screen can afford to show forty players and let the eye choose. A message
arriving on a Friday night cannot: it has one chance to say the thing worth
saying, and every extra paragraph makes the useful sentence harder to find.

So a digest is deliberately short: your captain, the transfer worth making (or
an explicit "roll it"), anyone in your squad who has become a problem, and where
you stand in the league that matters. Nothing that could not change a decision
before the deadline.

Pure assembly. Takes already-computed advice and renders it; runs no model and
performs no I/O, so the wording is testable without a network or a clock.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Below this, a transfer is churn. Recommending a move worth a fraction of a
# point invites the manager to burn a free transfer for noise, and doing that
# weekly is how a season is quietly lost.
MIN_WORTHWHILE_GAIN = 0.8


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def captain_line(picks: Sequence[Mapping]) -> str | None:
    """The armband, with the runner-up when it is close.

    A recommendation that hides its own uncertainty is worse than one that
    admits it: the manager who knows two picks are level makes the call on
    something we cannot see, like whether he watched the press conference.
    """
    if not picks:
        return None
    best = picks[0]
    line = (
        f"Captain {best['web_name']} ({best['team']}) — "
        f"{best['xpts_captained']:.1f} projected with the armband."
    )
    if len(picks) > 1:
        second = picks[1]
        margin = best["xpts"] - second["xpts"]
        if margin < 0.4:
            line += (
                f" {second['web_name']} is level with him on the numbers, so "
                "either is defensible."
            )
    if best.get("minutes_risk") == "high":
        line += " Worth checking he starts — the armband doubles a blank too."
    return line


def transfer_line(plans: Sequence[Mapping], free_transfers: int) -> str:
    """The move worth making, or an explicit instruction to hold.

    'No transfer' is a real recommendation and is stated as one. Silence reads
    as an omission, and a manager who thinks the tool had nothing to say will
    go and make a move anyway.
    """
    best = next((p for p in plans or [] if p.get("transfers")), None)
    if best is None or float(best.get("net_gain", 0)) < MIN_WORTHWHILE_GAIN:
        return (
            f"No transfer worth making. Roll your {_plural(free_transfers, 'free transfer')} "
            "— nothing available beats the squad you have by enough to matter."
        )

    moves = ", ".join(
        f"{t['out']['web_name']} → {t['in']['web_name']}" for t in best["transfers"]
    )
    gain = float(best.get("net_gain", 0))
    hit = int(best.get("hit_cost", 0))
    line = f"{moves} — worth {gain:+.1f} points over the horizon"
    if hit:
        line += f", after the -{hit} hit"
    return line + "."


def squad_alerts(squad: Sequence[Mapping]) -> list[str]:
    """Players who have become a problem since the manager last looked.

    Ordered by how bad the news is, because a message is read from the top and
    an injury matters more than a rotation risk.
    """
    injured, doubtful = [], []
    for player in squad or []:
        status = str(player.get("status", "a"))
        name = player.get("web_name", "")
        if status in ("i", "s", "u", "n"):
            injured.append(f"{name} is flagged unavailable")
        elif status == "d":
            chance = player.get("chance_of_playing_next_round")
            suffix = f" ({chance}% chance)" if chance is not None else ""
            doubtful.append(f"{name} is doubtful{suffix}")
    return injured + doubtful


def league_line(analysis: Mapping | None) -> str | None:
    """Where the manager stands, and what that implies for the week."""
    if not analysis:
        return None
    posture = analysis.get("posture") or {}
    if posture.get("stance") in (None, "unknown"):
        return None
    name = (analysis.get("league") or {}).get("name", "your league")
    return f"{name}: {posture['headline']} {posture['advice']}"


def build(
    *,
    manager_name: str,
    gameweek: int,
    deadline: str,
    captain_picks: Sequence[Mapping],
    plans: Sequence[Mapping],
    free_transfers: int,
    squad: Sequence[Mapping],
    league: Mapping | None = None,
) -> dict:
    """Assemble one manager's briefing."""
    alerts = squad_alerts(squad)
    captain = captain_line(captain_picks)
    transfer = transfer_line(plans, free_transfers)
    league_note = league_line(league)

    # The subject is the one line guaranteed to be read, so it carries the
    # most urgent thing rather than a generic label.
    if alerts:
        subject = f"GW{gameweek}: {alerts[0]}"
    elif captain:
        subject = f"GW{gameweek}: captain {captain_picks[0]['web_name']}"
    else:
        subject = f"GW{gameweek} deadline approaching"

    sections = []
    if alerts:
        sections.append(("Needs attention", alerts))
    if captain:
        sections.append(("Captain", [captain]))
    sections.append(("Transfer", [transfer]))
    if league_note:
        sections.append(("Your league", [league_note]))

    return {
        "subject": subject,
        "gameweek": gameweek,
        "deadline": deadline,
        "greeting": f"Hi {manager_name}," if manager_name else "Hi,",
        "sections": [{"title": t, "lines": list(ls)} for t, ls in sections],
    }


def render_text(digest: Mapping) -> str:
    """Plain text. Sent as the primary body: it renders everywhere, survives
    every client, and is what a screen reader reads."""
    out = [digest["greeting"], ""]
    out.append(
        f"Gameweek {digest['gameweek']} deadline: {digest['deadline']}"
    )
    out.append("")
    for section in digest["sections"]:
        out.append(section["title"].upper())
        for line in section["lines"]:
            out.append(f"  {line}")
        out.append("")
    out.append("Projections are estimates. Check team news before the deadline.")
    return "\n".join(out)


def render_html(digest: Mapping) -> str:
    """A minimal HTML alternative. Inline styles only — email clients strip
    stylesheets, and half of them strip <head> entirely."""
    blocks = []
    for section in digest["sections"]:
        items = "".join(
            f'<li style="margin-bottom:6px">{line}</li>' for line in section["lines"]
        )
        blocks.append(
            '<h2 style="font-size:13px;text-transform:uppercase;'
            'letter-spacing:.5px;color:#6b7280;margin:22px 0 6px">'
            f'{section["title"]}</h2>'
            f'<ul style="margin:0;padding-left:18px;font-size:15px;'
            f'line-height:1.55">{items}</ul>'
        )
    return (
        '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Roboto,Helvetica,Arial,sans-serif;max-width:560px;margin:0 auto;'
        'color:#1a1a2e">'
        f'<p style="font-size:15px">{digest["greeting"]}</p>'
        f'<p style="font-size:14px;color:#6b7280">Gameweek '
        f'{digest["gameweek"]} deadline: {digest["deadline"]}</p>'
        + "".join(blocks)
        + '<p style="font-size:12px;color:#6b7280;margin-top:28px;'
        'border-top:1px solid #e6e8ec;padding-top:14px">'
        "Projections are estimates. Check team news before the deadline.</p>"
        "</div>"
    )
