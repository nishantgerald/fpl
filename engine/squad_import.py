"""Read a squad out of a screenshot, and rate it.

The onboarding problem this solves: before the first deadline FPL publishes
nobody's squad, and after it a manager still needs a Team ID they may not know.
Both of our existing routes — Team ID, or a connected FPL session — require the
visitor to already be committed. A stranger who lands on the site and wants to
try the thing has nothing they can do.

A screenshot needs no account, no ID and no credential. It is the lowest-friction
on-ramp available and the only one that works for someone who arrived thirty
seconds ago.

Two halves, deliberately separate. The vision call reads *names* out of an image
and does nothing else — no scoring, no judgement, no arithmetic, because a
language model asked to both read and reason will quietly do neither well. Name
resolution and rating are ordinary code against the bootstrap, and are testable
without a model.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import unicodedata
from typing import Mapping, Sequence

from . import rules

# Screenshots of a full FPL squad are well under this. The cap exists because
# an unbounded upload on an anonymous route is a denial-of-service primitive.
MAX_IMAGE_BYTES = 6 * 1024 * 1024

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}

SQUAD_SIZE = 15

# Below this, a name match is a guess. Better to report a player as unreadable
# and let the user fix one row than to silently put someone else in their squad.
MIN_MATCH_SCORE = 0.72

VISION_PROMPT = """\
This image is a screenshot of a Fantasy Premier League squad.

List every player name you can read, in the order they appear, top to bottom.
Include bench players. There are usually 15.

Return ONLY a JSON object, no other text:
{"players": ["Name", "Name", ...]}

Rules:
- Copy the names exactly as printed. Do not expand abbreviations or correct
  spellings; the caller matches them against the official list.
- If a name is cut off or unreadable, omit it rather than guessing.
- Ignore team names, prices, points, and any interface labels.
"""


class ImportError_(Exception):
    """A user-correctable problem with the upload."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def decode_image(payload: str | bytes, mime: str = "") -> bytes:
    """Validate and decode an uploaded image.

    Checks the magic bytes rather than trusting the declared type: a client can
    claim any MIME it likes, and this route accepts uploads from anyone.
    """
    if isinstance(payload, str):
        # Tolerate a data URL, which is what a browser file reader produces.
        if payload.startswith("data:"):
            _, _, payload = payload.partition(",")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ImportError_("bad_image", "That upload isn't valid image data.") from error
    else:
        raw = payload

    if not raw:
        raise ImportError_("bad_image", "The upload was empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImportError_(
            "image_too_large",
            f"Screenshots must be under {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )
    if not _looks_like_image(raw):
        raise ImportError_(
            "bad_image", "That doesn't look like a PNG, JPEG or WebP image."
        )
    if mime and mime not in ALLOWED_MIME:
        raise ImportError_("bad_image", f"Unsupported image type: {mime}.")
    return raw


def _looks_like_image(raw: bytes) -> bool:
    return (
        raw.startswith(b"\x89PNG\r\n\x1a\n")
        or raw.startswith(b"\xff\xd8\xff")
        or (raw[:4] == b"RIFF" and raw[8:12] == b"WEBP")
    )


def parse_vision_output(text: str) -> list[str]:
    """Pull the name list out of whatever the model returned.

    Models wrap JSON in prose and fences however firmly you ask them not to, so
    the first well-formed object wins rather than the whole response having to
    be clean.
    """
    if not text:
        return []
    match = re.search(r"\{.*?\"players\".*?\}", text, re.S)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    names = parsed.get("players")
    if not isinstance(names, list):
        return []
    return [str(n).strip() for n in names if str(n).strip()][: SQUAD_SIZE * 2]


def _normalise(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value or "")
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z ]+", "", ascii_only.lower()).strip()


def _similarity(a: str, b: str) -> float:
    """Cheap token-aware ratio. Exact and surname matches score highest."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_tokens, b_tokens = set(a.split()), set(b.split())
    if a_tokens & b_tokens:
        overlap = len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))
        # A shared surname is strong evidence; a shared first name much less so.
        if a.split()[-1] == b.split()[-1]:
            overlap = max(overlap, 0.9)
        return overlap
    if a in b or b in a:
        return 0.8
    # Character overlap, as a floor for OCR noise.
    common = len(set(a) & set(b))
    return common / max(len(set(a)), len(set(b))) * 0.6


def resolve_names(
    names: Sequence[str], elements: Sequence[Mapping]
) -> tuple[list[dict], list[str]]:
    """Match read names to real players. Returns ``(matched, unresolved)``.

    A name below :data:`MIN_MATCH_SCORE` is reported as unreadable rather than
    matched to the nearest thing. Putting a player the user does not own into
    their squad and then rating it is worse than admitting one row failed.
    """
    index = []
    for element in elements:
        if rules.position_of(element) not in ("GKP", "DEF", "MID", "FWD"):
            continue
        full = f"{element.get('first_name', '')} {element.get('second_name', '')}"
        index.append(
            (
                _normalise(element.get("web_name", "")),
                _normalise(full),
                element,
            )
        )

    matched: list[dict] = []
    unresolved: list[str] = []
    taken: set[int] = set()

    for raw_name in names:
        target = _normalise(raw_name)
        if not target:
            continue
        best, best_score = None, 0.0
        for web, full, element in index:
            if int(element["id"]) in taken:
                continue
            score = max(_similarity(target, web), _similarity(target, full))
            if score > best_score:
                best, best_score = element, score

        if best is None or best_score < MIN_MATCH_SCORE:
            unresolved.append(raw_name)
            continue
        taken.add(int(best["id"]))
        matched.append(
            {
                "id": int(best["id"]),
                "web_name": best.get("web_name", ""),
                "read_as": raw_name,
                "confidence": round(best_score, 2),
            }
        )

    return matched, unresolved


def rate(
    matched: Sequence[Mapping],
    projections: Mapping[int, Mapping],
    elements: Mapping[int, Mapping],
    optimal_xpts: float | None = None,
) -> dict:
    """Score an imported squad and name its weakest links.

    The score is expressed against the best legal squad rather than on an
    invented 0–100 scale, because "you are 12 points off the optimum" is
    checkable and "your squad scores 74" is not.
    """
    rows = []
    for player in matched:
        element = elements.get(int(player["id"]), {})
        projection = projections.get(int(player["id"]), {})
        rows.append(
            {
                "id": int(player["id"]),
                "web_name": player.get("web_name", ""),
                "position": rules.position_of(element),
                "team_id": int(element.get("team", 0)),
                "price": int(element.get("now_cost", 0)) / 10,
                "xpts": round(float(projection.get("horizon_xpts") or 0.0), 1),
                "status": element.get("status", "a"),
                "minutes_risk": projection.get("minutes_risk", "medium"),
            }
        )

    total = round(sum(r["xpts"] for r in rows), 1)
    cost = round(sum(r["price"] for r in rows), 1)

    # Weakest by projection, but flagged players first — an injured pick is a
    # problem regardless of where his numbers sit.
    flagged = [r for r in rows if r["status"] not in ("a",)]
    weakest = sorted(
        (r for r in rows if r not in flagged), key=lambda r: r["xpts"]
    )[:3]

    verdict = _verdict(total, optimal_xpts, len(rows), flagged)

    return {
        "players": rows,
        "squad_xpts": total,
        "squad_cost": cost,
        "optimal_xpts": (
            round(optimal_xpts, 1) if optimal_xpts is not None else None
        ),
        "gap_to_optimal": (
            round(optimal_xpts - total, 1) if optimal_xpts is not None else None
        ),
        "flagged": flagged,
        "weakest": weakest,
        "verdict": verdict,
    }


def _verdict(
    total: float,
    optimal: float | None,
    count: int,
    flagged: Sequence[Mapping],
) -> str:
    if count < SQUAD_SIZE:
        lead = (
            f"Read {count} of {SQUAD_SIZE} players, so this is a partial "
            "picture — add the missing ones for a full rating."
        )
    else:
        lead = f"This squad projects {total:.1f} points over the next 5 gameweeks."

    if flagged:
        names = ", ".join(f["web_name"] for f in flagged[:3])
        return f"{lead} {names} {'is' if len(flagged) == 1 else 'are'} flagged — fix that first."

    if optimal is None:
        return lead
    gap = optimal - total
    if gap <= 5:
        return f"{lead} That is within {gap:.1f} of the best legal squad — strong."
    if gap <= 15:
        return (
            f"{lead} The best legal squad projects {optimal:.1f}, so there are "
            f"{gap:.1f} points on the table."
        )
    return (
        f"{lead} The best legal squad projects {optimal:.1f} — a gap of "
        f"{gap:.1f} points, which is worth a rebuild rather than a transfer."
    )
