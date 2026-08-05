"""The pre-deadline briefing.

Everything else in this app is pull — the manager opens it when they remember.
This is the one thing that is push, which changes what belongs in it: a message
has one chance to say the thing worth saying, and every extra line makes the
useful sentence harder to find.

So the properties under test are about restraint and honesty. "No transfer" is
stated rather than omitted, a close captaincy call admits it is close, and an
unconfigured mailer is a routine state rather than a crash.
"""

import pytest

from engine import digest, mailer


def _pick(name, xpts, risk="low", team="ARS"):
    return {
        "web_name": name, "team": team, "xpts": xpts,
        "xpts_captained": xpts * 2, "minutes_risk": risk,
    }


def _plan(gain, out="Gordon", into="Saka", hit=0):
    return {
        "net_gain": gain, "hit_cost": hit,
        "transfers": [{"out": {"web_name": out}, "in": {"web_name": into}}],
    }


# ---------------------------------------------------------------- captain


def test_a_close_captaincy_call_admits_it_is_close():
    """A recommendation that hides its own uncertainty is worse than one that
    admits it — the manager who knows two picks are level decides on something
    we cannot see."""
    line = digest.captain_line([_pick("Haaland", 6.0), _pick("Salah", 5.8)])

    assert "Haaland" in line
    assert "Salah" in line and "defensible" in line


def test_a_clear_captaincy_call_does_not_hedge():
    line = digest.captain_line([_pick("Haaland", 8.0), _pick("Salah", 4.0)])

    assert "Salah" not in line


def test_a_rotation_risk_captain_carries_the_warning():
    line = digest.captain_line([_pick("Someone", 6.0, risk="high")])

    assert "doubles a blank" in line


# ---------------------------------------------------------------- transfer


def test_no_transfer_is_stated_not_omitted():
    """Silence reads as an omission, and a manager who thinks the tool had
    nothing to say will make a move anyway."""
    line = digest.transfer_line([], free_transfers=2)

    assert "Roll your 2 free transfers" in line


def test_a_marginal_gain_is_treated_as_churn():
    """Recommending a move worth a fraction of a point weekly is how a season
    is quietly lost."""
    line = digest.transfer_line([_plan(0.3)], free_transfers=1)

    assert "No transfer worth making" in line
    assert "1 free transfer" in line  # singular


def test_a_worthwhile_transfer_names_the_move_and_the_gain():
    line = digest.transfer_line([_plan(6.4)], free_transfers=1)

    assert "Gordon → Saka" in line
    assert "+6.4" in line


def test_a_hit_is_disclosed_in_the_same_breath_as_the_gain():
    line = digest.transfer_line([_plan(4.0, hit=4)], free_transfers=0)

    assert "-4 hit" in line


# ---------------------------------------------------------------- alerts


def test_injuries_come_before_doubts():
    """A message is read from the top, and an injury matters more."""
    squad = [
        {"web_name": "A", "status": "d", "chance_of_playing_next_round": 50},
        {"web_name": "B", "status": "i"},
    ]

    alerts = digest.squad_alerts(squad)

    assert alerts[0].startswith("B")
    assert "50%" in alerts[1]


def test_a_fit_squad_raises_nothing():
    assert digest.squad_alerts([{"web_name": "A", "status": "a"}]) == []


# ---------------------------------------------------------------- assembly


def _built(**overrides):
    kwargs = dict(
        manager_name="Nishant", gameweek=7, deadline="2026-10-03T10:00:00Z",
        captain_picks=[_pick("Haaland", 8.0)], plans=[_plan(6.4)],
        free_transfers=1, squad=[{"web_name": "A", "status": "a"}], league=None,
    )
    kwargs.update(overrides)
    return digest.build(**kwargs)


def test_the_subject_leads_with_the_most_urgent_thing():
    """The subject is the one line guaranteed to be read."""
    urgent = _built(squad=[{"web_name": "Saka", "status": "i"}])
    routine = _built()

    assert "Saka" in urgent["subject"] and "unavailable" in urgent["subject"]
    assert "captain Haaland" in routine["subject"]


def test_a_transfer_section_is_always_present():
    """Because 'roll it' is a decision the manager still has to make."""
    titles = [s["title"] for s in _built(plans=[])["sections"]]
    assert "Transfer" in titles


def test_league_context_appears_only_when_it_says_something():
    without = _built(league={"posture": {"stance": "unknown"}})
    with_league = _built(
        league={
            "league": {"name": "NBC Sports League"},
            "posture": {"stance": "chase", "headline": "120 behind.",
                        "advice": "Differentiate."},
        }
    )

    assert "Your league" not in [s["title"] for s in without["sections"]]
    assert "Your league" in [s["title"] for s in with_league["sections"]]


def test_both_renderings_contain_the_advice():
    built = _built()
    text = digest.render_text(built)
    html = digest.render_html(built)

    for body in (text, html):
        assert "Haaland" in body
        assert "Gordon → Saka" in body
        assert "estimates" in body


# ---------------------------------------------------------------- mailer


def test_an_unconfigured_mailer_is_routine_not_a_failure(monkeypatch):
    """Every developer machine and every test run lacks SMTP. A scheduled job
    must not die on the first machine that isn't production."""
    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)

    delivery = mailer.send("a@b.co", "Subject", "Body")

    assert delivery.sent is False
    assert delivery.reason == "smtp_not_configured"
    assert mailer.is_configured() is False


def test_a_bad_recipient_is_reported_rather_than_raised(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_FROM", "fpl@example.com")

    assert mailer.send("not-an-email", "S", "B").reason == "invalid_recipient"


def test_a_send_failure_is_returned_never_raised(monkeypatch):
    """A failure partway through a mailing list must not abandon the rest."""
    monkeypatch.setenv("SMTP_HOST", "smtp.invalid.example")
    monkeypatch.setenv("SMTP_FROM", "fpl@example.com")

    def _boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(mailer.smtplib, "SMTP", _boom)
    delivery = mailer.send("a@b.co", "S", "B")

    assert delivery.sent is False
    assert "OSError" in delivery.reason
