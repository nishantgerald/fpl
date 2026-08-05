"""Reading a squad out of a screenshot.

The onboarding this exists for: Team ID and the session connect both require a
visitor who is already committed. A screenshot needs no account, no ID and no
credential, which makes it the only on-ramp that works for someone who arrived
thirty seconds ago.

The split under test is deliberate. The model reads *names* and nothing else;
resolution and scoring are ordinary code. So these tests need no model — which
is the point, since a feature whose correctness can only be checked by calling
a language model cannot really be checked at all.
"""

import base64

import pytest

from engine import squad_import


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64


def _element(eid, first, second, web, etype=3, cost=50):
    return {
        "id": eid, "first_name": first, "second_name": second, "web_name": web,
        "element_type": etype, "now_cost": cost, "team": 1, "status": "a",
    }


SQUAD = [
    _element(1, "Erling", "Haaland", "Haaland", etype=4, cost=155),
    _element(2, "Bruno", "Fernandes", "B.Fernandes", cost=120),
    _element(3, "Gabriel", "dos Santos Magalhães", "Gabriel", etype=2, cost=80),
    _element(4, "Marc", "Guéhi", "Guéhi", etype=2, cost=60),
    _element(5, "Cole", "Palmer", "Palmer", cost=95),
]


# ---------------------------------------------------------------- uploads


def test_a_data_url_from_a_browser_is_accepted():
    encoded = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert squad_import.decode_image(encoded) == PNG


def test_magic_bytes_are_checked_not_the_declared_type():
    """A client can claim any MIME it likes, and this route accepts uploads
    from anyone."""
    not_an_image = base64.b64encode(b"#!/bin/sh\nrm -rf /").decode()

    with pytest.raises(squad_import.ImportError_) as excinfo:
        squad_import.decode_image(not_an_image, "image/png")

    assert excinfo.value.code == "bad_image"


def test_oversized_uploads_are_refused():
    """An unbounded upload on an anonymous route is a denial-of-service tool."""
    huge = PNG + b"\x00" * squad_import.MAX_IMAGE_BYTES

    with pytest.raises(squad_import.ImportError_) as excinfo:
        squad_import.decode_image(huge)

    assert excinfo.value.code == "image_too_large"


def test_jpeg_and_webp_are_allowed_too():
    assert squad_import.decode_image(JPEG) == JPEG
    webp = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 32
    assert squad_import.decode_image(webp) == webp


# ---------------------------------------------------------------- parsing


def test_json_is_recovered_from_a_model_that_wrapped_it_in_prose():
    """Models fence and preamble their JSON however firmly you ask them not to."""
    messy = 'Sure! Here is the squad:\n```json\n{"players": ["Haaland", "Saka"]}\n```'

    assert squad_import.parse_vision_output(messy) == ["Haaland", "Saka"]


def test_unparseable_output_yields_nothing_rather_than_guessing():
    assert squad_import.parse_vision_output("I could not read that image.") == []
    assert squad_import.parse_vision_output("") == []
    assert squad_import.parse_vision_output('{"players": "not a list"}') == []


# ---------------------------------------------------------------- matching


def test_names_resolve_through_ocr_noise_and_accents():
    matched, unresolved = squad_import.resolve_names(
        ["Haaland", "B.Fernandes", "Guehi"], SQUAD
    )

    assert [m["web_name"] for m in matched] == ["Haaland", "B.Fernandes", "Guéhi"]
    assert unresolved == []


def test_a_full_name_matches_its_web_name():
    matched, _ = squad_import.resolve_names(["Erling Haaland"], SQUAD)
    assert matched[0]["id"] == 1


def test_an_unreadable_name_is_reported_rather_than_guessed():
    """Putting a player the user does not own into their squad and then rating
    it is worse than admitting one row failed."""
    matched, unresolved = squad_import.resolve_names(["Zzzqqxx"], SQUAD)

    assert matched == []
    assert unresolved == ["Zzzqqxx"]


def test_the_same_player_is_never_matched_twice():
    """OCR duplicates a row often enough that this matters."""
    matched, unresolved = squad_import.resolve_names(
        ["Haaland", "Haaland"], SQUAD
    )

    assert len(matched) == 1
    assert len(unresolved) == 1


# ---------------------------------------------------------------- rating


def _projections(overrides=None):
    base = {i: {"horizon_xpts": 20.0, "minutes_risk": "low"} for i in range(1, 6)}
    base.update(overrides or {})
    return base


def test_the_rating_is_expressed_against_the_best_legal_squad():
    """"You are 12 points off the optimum" is checkable; "your squad scores 74"
    is not."""
    matched, _ = squad_import.resolve_names(["Haaland", "Palmer"], SQUAD)
    rating = squad_import.rate(
        matched, _projections(), {e["id"]: e for e in SQUAD}, optimal_xpts=60.0
    )

    assert rating["squad_xpts"] == 40.0
    assert rating["gap_to_optimal"] == 20.0
    assert "60.0" in rating["verdict"]


def test_a_flagged_player_leads_the_verdict_whatever_his_numbers():
    injured = dict(SQUAD[0], status="i")
    elements = {e["id"]: e for e in SQUAD}
    elements[1] = injured

    matched, _ = squad_import.resolve_names(["Haaland", "Palmer"], SQUAD)
    rating = squad_import.rate(matched, _projections(), elements, optimal_xpts=60.0)

    assert rating["flagged"][0]["web_name"] == "Haaland"
    assert "flagged" in rating["verdict"]


def test_a_partial_read_says_so_instead_of_rating_a_partial_squad():
    matched, _ = squad_import.resolve_names(["Haaland"], SQUAD)
    rating = squad_import.rate(matched, _projections(), {e["id"]: e for e in SQUAD})

    assert "1 of 15" in rating["verdict"]


def test_the_weakest_links_are_named():
    matched, _ = squad_import.resolve_names(
        ["Haaland", "Palmer", "Gabriel"], SQUAD
    )
    rating = squad_import.rate(
        matched,
        _projections({3: {"horizon_xpts": 2.0, "minutes_risk": "high"}}),
        {e["id"]: e for e in SQUAD},
    )

    assert rating["weakest"][0]["web_name"] == "Gabriel"
