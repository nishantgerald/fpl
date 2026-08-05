"""The LLM layer must be strictly additive and unable to change a decision.

Narration now reaches the model through the Claude CLI, so "no model" is spelled
as a binary that isn't there. No test here may invoke the real CLI: that would
spend a call against a real subscription and make the suite depend on the machine
being logged in.
"""

import os

import pytest

from engine import fcps_llm, llm_budget, narrative


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No reachable CLI, and budget state that can't outlive the test."""
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(tmp_path / "definitely-not-here"))
    monkeypatch.setattr(llm_budget, "STATE_DIR", tmp_path / "budget")
    narrative.clear_cache()


@pytest.fixture
def _cli(monkeypatch, tmp_path):
    """A stand-in CLI, so is_enabled() can be true without a real model."""
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(binary))
    return binary


def _plan():
    return {
        "n_transfers": 1,
        "hit_cost": 0,
        "net_gain": 6.4,
        "spend": 11,
        "transfers": [
            {
                "out": {"id": 1, "web_name": "Gordon", "name": "Anthony Gordon",
                        "position": "MID", "team": "NEW", "selling_price": 74},
                "in": {"id": 2, "web_name": "Saka", "name": "Bukayo Saka",
                       "position": "MID", "team": "ARS", "now_cost": 85},
            }
        ],
        "reasons": ["Gains 6.4 pts over 5 GWs"],
    }


def test_disabled_by_default(monkeypatch, _cli):
    monkeypatch.delenv("ENABLE_LLM_NARRATIVE", raising=False)
    assert narrative.is_enabled() is False


def test_needs_both_the_flag_and_a_reachable_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(tmp_path / "nope"))
    assert narrative.is_enabled() is False

    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FCPS_CLAUDE_BIN", str(binary))
    assert narrative.is_enabled() is True


def test_only_the_first_plan_is_ever_narrated(monkeypatch, _cli, tmp_path):
    """The amplifier this replaced: five plans used to mean five model calls."""
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")

    calls = []
    monkeypatch.setattr(
        narrative, "_narrate", lambda plan: calls.append(plan) or "Two sentences."
    )

    plans = [_plan() for _ in range(5)]
    narrative.annotate(plans)

    assert len(calls) == 1
    assert "narrative" in plans[0]
    assert all("narrative" not in p for p in plans[1:])


def test_a_spent_budget_drops_the_prose_rather_than_raising(monkeypatch, _cli):
    """Narration is additive; running out of budget must not fail the request."""
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    monkeypatch.setattr(llm_budget, "DAILY_CALL_CEILING", 0)

    plans = [_plan()]
    narrative.annotate(plans)  # must not raise
    assert "narrative" not in plans[0]


def test_would_call_is_false_when_disabled(monkeypatch, _cli):
    monkeypatch.delenv("ENABLE_LLM_NARRATIVE", raising=False)
    assert narrative.would_call([_plan()]) is False


def test_would_call_is_false_once_cached(monkeypatch, _cli):
    """A cache hit costs nothing, so it must not be charged to the caller."""
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    plan = _plan()
    assert narrative.would_call([plan]) is True

    narrative._cache_put(narrative._key_for(plan), "Two sentences.")
    assert narrative.would_call([plan]) is False


def test_the_cache_keys_on_plan_content_not_on_the_requester(monkeypatch, _cli):
    """Two managers with the same problem get the same prose for one call."""
    assert narrative._key_for(_plan()) == narrative._key_for(_plan())

    other = _plan()
    other["transfers"][0]["in"]["web_name"] = "Palmer"
    assert narrative._key_for(other) != narrative._key_for(_plan())


def test_annotate_is_a_no_op_when_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_LLM_NARRATIVE", raising=False)
    plans = [_plan()]
    before = [dict(p) for p in plans]
    narrative.annotate(plans)
    assert plans == before
    assert "narrative" not in plans[0]


def test_prompt_contains_only_the_plan():
    prompt = narrative._prompt(_plan())
    assert "Gordon" in prompt and "Saka" in prompt
    assert len(prompt) < 2048, "the old prompt embedded ~85 player records"


def test_guardrail_accepts_prose_about_the_plan():
    text = (
        "Saka offers a stronger fixture run than Gordon over the next five "
        "gameweeks. The move costs 1.1m and uses your free transfer."
    )
    assert narrative._passes_guardrails(text, _plan()) is True


def test_guardrail_rejects_an_invented_alternative():
    text = "Consider Palmer instead of Saka, who has better underlying numbers."
    assert narrative._passes_guardrails(text, _plan()) is False


def test_guardrail_rejects_an_overlong_response():
    assert narrative._passes_guardrails("word " * 200, _plan()) is False


def test_guardrail_rejects_empty_output():
    assert narrative._passes_guardrails("", _plan()) is False


def test_club_names_are_allowed():
    text = "Saka faces a kind run while Newcastle's fixtures turn against Gordon."
    assert narrative._passes_guardrails(text, _plan()) is True
