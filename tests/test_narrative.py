"""The LLM layer must be strictly additive and unable to change a decision."""

import os

from engine import narrative


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


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LLM_NARRATIVE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert narrative.is_enabled() is False


def test_needs_both_the_flag_and_a_key(monkeypatch):
    monkeypatch.setenv("ENABLE_LLM_NARRATIVE", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert narrative.is_enabled() is False

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert narrative.is_enabled() is True


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
