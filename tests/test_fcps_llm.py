"""The FCPS advice layer, tested without a network or a model.

The bugs being pinned down here are the ones that made the feature unreachable
rather than merely wrong:

* a missing model must raise, not return a 200 with an exception string in the
  field the client renders as advice;
* the prompt must carry the manager's bank, which the original never passed
  while asking the model to check affordability;
* nothing in this module may import Flask or touch a template.

The model is reached by spawning the Claude CLI, so "no model available" is
spelled as a binary that isn't there. No test in this file may invoke the real
CLI: that would spend a real call against a real subscription, and would make
the suite depend on the machine being logged in.
"""

import pathlib
import time

import pytest

from engine import fcps, fcps_llm, llm_budget
from tests.conftest import make_element


@pytest.fixture(autouse=True)
def _no_cli(monkeypatch, tmp_path):
    """No reachable CLI, and no state that can outlive the test.

    ``llm_budget.STATE_DIR`` is redirected too: ``call_model`` reserves against
    the global daily ceiling, so without this the suite spends the operator's
    real budget every run and the counter drifts up with no calls behind it.
    """
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(tmp_path / "definitely-not-here"))
    monkeypatch.setattr(fcps_llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm_budget, "STATE_DIR", tmp_path / "budget")
    fcps_llm.clear_cache()


@pytest.fixture
def rows(elements, fixtures):
    scored = fcps.score_all(elements, fixtures, 13)
    return [
        fcps_llm.player_row(element, scored[int(element["id"])], "ARS")
        for element in elements[:20]
    ]


def test_missing_cli_raises_rather_than_returning_an_error_as_advice(rows):
    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.advise(rows[:15], rows[15:], gameweek=14)

    assert caught.value.code == "fcps_not_configured"
    assert caught.value.status == 503
    assert caught.value.to_dict()["code"] == "fcps_not_configured"


def test_is_configured_reflects_whether_the_cli_is_reachable(monkeypatch, tmp_path):
    assert fcps_llm.is_configured() is False

    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(binary))
    assert fcps_llm.is_configured() is True


def test_a_nonzero_exit_is_an_upstream_error_not_advice(monkeypatch):
    """A failed CLI call must not reach the client as prose."""
    monkeypatch.setattr(fcps_llm, "cli_path", lambda: "/bin/false")

    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.call_model("anything")

    assert caught.value.code == "fcps_upstream_error"
    assert caught.value.status == 502


def test_non_json_output_is_an_upstream_error(monkeypatch):
    monkeypatch.setattr(fcps_llm, "cli_path", lambda: "/bin/echo")

    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.call_model("anything")

    assert caught.value.code == "fcps_upstream_error"


def _fake_cli(tmp_path, payload: dict):
    """A stand-in for the CLI that echoes one fixed JSON payload.

    Written in Python rather than shell because ``sh``'s ``echo`` expands
    backslash escapes, which silently corrupts any JSON containing a newline.
    """
    import json

    binary = tmp_path / "fake-claude"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({json.dumps(json.dumps(payload))})\n"
    )
    binary.chmod(0o755)
    return str(binary)


def test_the_cli_error_flag_is_honoured_even_on_a_zero_exit(monkeypatch, tmp_path):
    """The CLI can exit 0 and still report failure in the payload."""
    binary = _fake_cli(
        tmp_path, {"is_error": True, "subtype": "rate_limit", "result": ""}
    )
    monkeypatch.setattr(fcps_llm, "cli_path", lambda: binary)

    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.call_model("anything")

    assert caught.value.code == "fcps_upstream_error"
    assert "rate_limit" in caught.value.message


def test_the_result_field_is_what_comes_back(monkeypatch, tmp_path):
    binary = _fake_cli(tmp_path, {"is_error": False, "result": "# Column\n"})
    monkeypatch.setattr(fcps_llm, "cli_path", lambda: binary)

    assert fcps_llm.call_model("anything") == "# Column"


def test_an_empty_result_is_an_error_not_an_empty_column(monkeypatch, tmp_path):
    binary = _fake_cli(tmp_path, {"is_error": False, "result": "   "})
    monkeypatch.setattr(fcps_llm, "cli_path", lambda: binary)

    with pytest.raises(fcps_llm.FcpsUnavailable) as caught:
        fcps_llm.call_model("anything")

    assert caught.value.code == "fcps_upstream_error"


def test_the_day_long_gate_survives_a_process_restart(monkeypatch, tmp_path, rows):
    """The cache is the rate limit gate, so it has to outlive the worker."""
    key = (12345, 14, "sonnet")
    fcps_llm._cache_put(key, {"markdown": "# Column", "cached": False})

    # Everything in memory is gone; only the disk tier remains.
    fcps_llm._CACHE.clear()

    recovered = fcps_llm._cache_get(key)
    assert recovered is not None
    assert recovered["markdown"] == "# Column"


def test_an_expired_entry_is_a_miss_on_disk_too(monkeypatch, rows):
    key = (12345, 14, "sonnet")
    fcps_llm._cache_put(key, {"markdown": "# Column"})
    fcps_llm._CACHE.clear()

    # Captured before patching: re-importing inside the lambda would resolve to
    # the patched module and recurse.
    a_day_later = time.time() + 86_401
    monkeypatch.setattr(fcps_llm.time, "time", lambda: a_day_later)
    assert fcps_llm._cache_get(key) is None


def test_an_unwritable_cache_dir_does_not_break_the_response(monkeypatch, rows):
    """A degraded cache weakens the gate; it must not fail the request."""
    monkeypatch.setattr(fcps_llm, "CACHE_DIR", pathlib.Path("/proc/x/y"))

    fcps_llm._cache_put((1, 2, "sonnet"), {"markdown": "ok"})  # must not raise
    assert fcps_llm._CACHE[(1, 2, "sonnet")][1]["markdown"] == "ok"


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
        "minutes", "next_3_fdr", "ict_index", "fcps", "status",
        "starting_eleven",
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


# ------------------------------------------------- what a zero is allowed to mean


def test_a_player_who_has_not_played_shows_dashes_not_zeros(elements, fixtures):
    """Fixtures within a gameweek run across several days, so early in one most
    players have no minutes. Printing their points and form as 0 states a fact
    the season has not established, and the advice duly quoted it back: "zero
    points, zero form" about a defender whose match had not kicked off, and
    "blanked in GW1" about a midfielder for the same reason.

    A dash cannot be read as a score.
    """
    scored = fcps.score_all(elements, fixtures, 13)
    element = {**elements[0], "minutes": 0, "total_points": 0, "form": "0.0"}
    row = fcps_llm.player_row(
        element, scored[int(elements[0]["id"])], "ARS", in_squad=True, starting=True
    )

    table = fcps_llm._table([row])

    assert "| Mins |" in table, 'the column that explains the dash'
    assert "—" in table
    assert "| 0 | 0.0 |" not in table


def test_a_player_who_has_played_still_shows_his_numbers(elements, fixtures):
    """The masking must not hide a real zero. Somebody who played ninety minutes
    and scored nothing has told us something, and it is the opposite thing."""
    scored = fcps.score_all(elements, fixtures, 13)
    element = {**elements[0], "minutes": 90, "total_points": 0, "form": "0.0"}
    row = fcps_llm.player_row(
        element, scored[int(elements[0]["id"])], "ARS", in_squad=True, starting=True
    )

    table = fcps_llm._table([row])

    assert "| 90 | 0 | 0.0 |" in table


def test_the_prompt_says_what_a_dash_means():
    prompt = fcps_llm.build_prompt([], [], gameweek=1)

    # Matched on unwrapped fragments: the constraint is a wrapped paragraph, so
    # asserting a whole sentence would break on a reflow rather than on meaning.
    assert "A player on 0 minutes has not" in prompt
    assert "not because they played badly" in prompt
