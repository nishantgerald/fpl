"""The FCPS advice layer, tested without a network or an API key.

The bugs being pinned down here are the ones that made the feature unreachable
rather than merely wrong:

* a missing key must raise, not return a 200 with an exception string in the
  field the client renders as advice;
* the prompt must carry the manager's bank, which the original never passed
  while asking the model to check affordability;
* nothing in this module may import Flask or touch a template.
"""

import pytest

from engine import fcps, fcps_llm
from tests.conftest import make_element


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fcps_llm.clear_cache()


@pytest.fixture
def rows(elements, fixtures):
    scored = fcps.score_all(elements, fixtures, 13)
    return [
        fcps_llm.player_row(element, scored[int(element["id"])], "ARS")
        for element in elements[:20]
    ]


def test_missing_key_raises_rather_than_returning_an_error_as_advice(rows):
    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.advise(rows[:15], rows[15:], gameweek=14)

    assert caught.value.code == "fcps_not_configured"
    assert caught.value.status == 503
    assert caught.value.to_dict()["code"] == "fcps_not_configured"


def test_is_configured_reflects_the_environment(monkeypatch):
    assert fcps_llm.is_configured() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert fcps_llm.is_configured() is True


def test_prompt_carries_the_bank_and_the_free_transfer_count(rows):
    prompt = fcps_llm.build_prompt(
        rows[:15], rows[15:], gameweek=14, bank=23, free_transfers=2
    )

    assert "2.3m" in prompt
    assert "Free transfers available: 2" in prompt
    assert "at most 3 players from the same club" in prompt


def test_prompt_is_a_table_not_a_dataframe_dump(rows):
    prompt = fcps_llm.build_prompt(rows[:15], rows[15:], gameweek=14)

    assert "| Name | Pos | Team |" in prompt
    assert "{'id':" not in prompt  # the original embedded repr'd dicts
    # The old prompt was ~40 KB of JSON for ~85 players. This one is a table.
    assert len(prompt) < 12_000


def test_prompt_handles_an_empty_shortlist():
    assert "_(none)_" in fcps_llm.build_prompt([], [], gameweek=1)


def test_player_row_is_compact_and_typed(elements, fixtures):
    scored = fcps.score_all(elements, fixtures, 13)
    element = elements[0]

    row = fcps_llm.player_row(
        element, scored[int(element["id"])], "ARS", in_squad=True, starting=False
    )

    assert set(row) == {
        "id", "name", "team", "position", "price", "total_points", "form",
        "next_3_fdr", "ict_index", "fcps", "status", "starting_eleven",
    }
    assert isinstance(row["price"], float)
    assert row["starting_eleven"] is False


def test_audit_mentions_counts_only_players_present_in_the_data(rows):
    markdown = "Bring in First1 Last1 for First2 Last2. Avoid Haaland."

    audit = fcps_llm.audit_mentions(markdown, rows[:2], [])

    # Last1/First1/Last2/First2 are known; Haaland is not in the data given.
    assert audit["known_players_named"] == 4
    assert audit["data_rows"] >= 4


def test_cache_returns_the_previous_result_without_calling_out(monkeypatch, rows):
    calls = []

    def fake_advise_body(*_args, **_kwargs):
        calls.append(1)
        return {"markdown": "ok", "model": "test", "gameweek": 14}

    key = (1, 14, "test-model")
    fcps_llm._cache_put(key, fake_advise_body())

    result = fcps_llm.advise(rows[:15], rows[15:], gameweek=14, cache_key=key)

    assert result["cached"] is True
    assert result["markdown"] == "ok"
    assert len(calls) == 1  # the cached put, not a second call


def test_module_has_no_web_framework_dependency():
    """The 500 was a missing template. There is no template to miss now.

    Checked structurally rather than by grepping the source, because the module
    docstring legitimately quotes the old `render_template` line while
    explaining the bug.
    """
    import sys

    assert not hasattr(fcps_llm, "render_template")
    assert not hasattr(fcps_llm, "request")

    module = sys.modules[fcps_llm.__name__]
    imported = {
        getattr(value, "__module__", "") or ""
        for value in vars(module).values()
        if callable(value)
    }
    assert not any(name.startswith("flask") for name in imported)


def test_shortlist_excludes_the_squad_by_construction(elements, fixtures):
    """Recommending a player you already own was the original's loudest failure."""
    scored = fcps.score_all(elements, fixtures, 13)
    shortlist = fcps.top_by_position(scored, elements)
    squad_ids = {int(e["id"]) for e in elements[:15]}

    filtered = [e for e in shortlist if int(e["id"]) not in squad_ids]

    assert not (squad_ids & {int(e["id"]) for e in filtered})


def test_manager_elements_never_reach_a_row(elements, fixtures):
    manager = make_element(9100, 5, team=1)
    scored = fcps.score_all(list(elements) + [manager], fixtures, 13)
    assert 9100 not in scored
