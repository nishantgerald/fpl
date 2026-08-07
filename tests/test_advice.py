"""Advice that has to be worth acting on.

Every test here is a defect that shipped, described by the person who found it:

* a star midfielder recommended for sale because his club's run hardened, when
  nothing affordable outscores him;
* a forward recommended as "worth buying" alongside the projection that
  disqualified him — ten points across five gameweeks, which is two a week;
* chip advice about doubles and blanks that never said whether either was
  scheduled.

They share one shape. The engine emitted an item when a condition was true
rather than when acting on it would gain points, so the tests are written the
same way round: they assert on *silence* as much as on output, because the
hardest thing for an advice engine to do is decline to advise.
"""

from engine import advice


def _element(
    element_id,
    name,
    position=3,
    price=7.0,
    team=1,
    status="a",
    form=3.0,
    ownership=20.0,
):
    return {
        "id": element_id,
        "web_name": name,
        "element_type": position,
        "now_cost": int(price * 10),
        "team": team,
        "status": status,
        "form": str(form),
        "selected_by_percent": str(ownership),
    }


def _projection(horizon_xpts, per_gameweek=None):
    return {"horizon_xpts": horizon_xpts, "per_gameweek": per_gameweek or []}


def _run(values, start=1):
    """A per-gameweek projection with the given scores."""
    return [{"gameweek": start + i, "xpts": v} for i, v in enumerate(values)]


TEAMS = {1: {"short_name": "MUN", "name": "Man Utd"}, 2: {"short_name": "ARS", "name": "Arsenal"}}

WORSENING = {
    1: {
        "team": "MUN",
        "team_id": 1,
        "direction": "worsening",
        "from_gameweek": 4,
        "message": "Man Utd's run gets notably harder.",
    }
}


# ----------------------------------------------------------------- selling


def test_an_elite_player_is_not_sold_because_his_clubs_run_hardens():
    """The complaint that started this, almost verbatim.

    A premium's fixtures worsening is a fact about his club. He is still the
    best midfielder the money can buy, so the recommendation had no achievable
    version — every transfer it implied loses points.
    """
    star = _element(1, "Fernandes", price=9.0, team=1)
    # A steep, genuinely material decline — the trigger is not what is wrong.
    projections = {1: _projection(45.0, _run([12.0, 11.0, 6.0, 5.0, 5.0]))}
    elements = {1: star}
    for i, price in enumerate([5.0, 6.5, 8.0, 9.5], start=2):
        elements[i] = _element(i, f"Lesser{i}", price=price, team=2)
        projections[i] = _projection(20.0)

    items = advice.sell_advice(
        [star], elements, projections, WORSENING, {1}, horizon=5, teams=TEAMS
    )

    assert items == []


def test_a_worsening_run_is_a_sell_when_something_affordable_beats_him():
    """The same trigger, the opposite conclusion — because a buyer exists."""
    owned = _element(1, "Mid", price=7.0, team=1)
    replacement = _element(2, "Better", price=7.5, team=2)
    projections = {
        1: _projection(24.0, _run([8.0, 7.0, 3.0, 3.0, 3.0])),
        2: _projection(34.0),
    }

    items = advice.sell_advice(
        [owned],
        {1: owned, 2: replacement},
        projections,
        WORSENING,
        {1},
        horizon=5,
        teams=TEAMS,
    )

    assert len(items) == 1
    action = items[0]
    # The gain is the point. A sell recommendation that does not say what you
    # get instead leaves the whole decision undone.
    assert "Better" in action["headline"]
    assert action["gain"] == 10.0
    assert "GW4" in action["action"]
    assert "£0.5m from the bank" in action["action"]


def test_a_player_the_swing_barely_touches_is_left_alone():
    """The club's run changes; his projection does not. Nothing to do."""
    owned = _element(1, "Steady", price=7.0, team=1)
    replacement = _element(2, "Better", price=7.0, team=2)
    projections = {
        1: _projection(25.0, _run([5.1, 5.1, 4.9, 4.9, 5.0])),
        2: _projection(40.0),
    }

    items = advice.sell_advice(
        [owned],
        {1: owned, 2: replacement},
        projections,
        WORSENING,
        {1},
        horizon=5,
        teams=TEAMS,
    )

    assert items == []


def test_a_replacement_has_to_clear_the_absolute_bar_too():
    """Beating the outgoing player is not enough if both are bench fodder.

    Otherwise a squad full of poor players generates a stream of transfers
    between poor players, each one technically an improvement.
    """
    owned = _element(1, "Bad", price=5.0, team=1)
    slightly_better = _element(2, "AlsoBad", price=5.0, team=2)
    projections = {
        1: _projection(4.0, _run([2.0, 1.5, 0.2, 0.2, 0.1])),
        2: _projection(9.0),  # 1.8 a gameweek — clears MIN_TRANSFER_GAIN, not the bar
    }

    items = advice.sell_advice(
        [owned],
        {1: owned, 2: slightly_better},
        projections,
        WORSENING,
        {1},
        horizon=5,
        teams=TEAMS,
    )

    assert items == []


def test_a_dearer_upgrade_is_not_offered_beyond_the_assumed_bank():
    owned = _element(1, "Mid", price=7.0, team=1)
    unaffordable = _element(2, "Premium", price=12.0, team=2)
    projections = {
        1: _projection(24.0, _run([8.0, 7.0, 3.0, 3.0, 3.0])),
        2: _projection(60.0),
    }

    items = advice.sell_advice(
        [owned],
        {1: owned, 2: unaffordable},
        projections,
        WORSENING,
        {1},
        horizon=5,
        teams=TEAMS,
    )

    assert items == []


# ------------------------------------------------------------------ buying


def _buy_pool():
    """A pool with one of each interesting kind, plus a dud."""
    elements = {
        1: _element(1, "Wright", position=4, price=5.0, team=2, form=1.0, ownership=1.0),
        2: _element(2, "Hot", position=3, price=8.0, team=2, form=7.5, ownership=30.0),
        3: _element(3, "Hidden", position=3, price=6.5, team=2, form=4.0, ownership=2.0),
        4: _element(4, "Cheap", position=2, price=4.5, team=2, form=4.0, ownership=15.0),
    }
    projections = {
        1: _projection(10.0),   # 2.0 a gameweek — the disqualifying number
        2: _projection(35.0),
        3: _projection(28.0),
        4: _projection(21.0),
    }
    return elements, projections


def test_a_two_points_a_gameweek_forward_is_not_worth_buying():
    """Quoting ten points over five gameweeks *as the reason to buy* was the bug.

    He is on no list — not even 'nobody else owns them', where 1% ownership
    would otherwise have put him top.
    """
    elements, projections = _buy_pool()

    lists = advice.buy_shortlists(
        elements, projections, {}, set(), [], horizon=5, teams=TEAMS
    )

    named = {p["player"] for group in lists for p in group["players"]}
    assert "Wright" not in named
    assert named  # and the bar did not empty the whole feature


def test_a_target_names_the_player_it_would_replace_and_the_gain():
    """A transfer is a swap. A name on its own is not a recommendation."""
    elements, projections = _buy_pool()
    owned = _element(9, "Weakest", position=3, price=6.0, team=1)
    projections[9] = _projection(12.0)

    lists = advice.buy_shortlists(
        elements, projections, {}, {9}, [owned], horizon=5, teams=TEAMS
    )

    hot = next(
        p
        for group in lists
        for p in group["players"]
        if p["player"] == "Hot"
    )
    assert hot["replaces"] == "Weakest"
    assert hot["gain"] == 23.0
    assert hot["team"] == "ARS"


def test_the_realistic_swap_is_out_the_worst_not_out_the_best():
    """Gains measured against a squad's best player would all read as losses."""
    elements, projections = _buy_pool()
    best = _element(9, "Star", position=3, price=9.0, team=1)
    worst = _element(10, "Dud", position=3, price=5.0, team=1)
    projections[9] = _projection(50.0)
    projections[10] = _projection(11.0)

    lists = advice.buy_shortlists(
        elements, projections, {}, {9, 10}, [best, worst], horizon=5, teams=TEAMS
    )

    hot = next(
        p for group in lists for p in group["players"] if p["player"] == "Hot"
    )
    assert hot["replaces"] == "Dud"


def test_each_shortlist_answers_a_different_question():
    """Four lists that return the same three names are one list printed four
    times, which is the failure mode this shape exists to avoid."""
    elements, projections = _buy_pool()
    swings = {
        2: {
            "team_id": 2,
            "direction": "improving",
            "from_gameweek": 6,
            "message": "eases",
        }
    }

    lists = advice.buy_shortlists(
        elements, projections, swings, set(), [], horizon=5, teams=TEAMS
    )
    by_key = {group["key"]: group for group in lists}

    assert by_key["form"]["players"][0]["player"] == "Hot"          # form 7.5
    assert by_key["differential"]["players"][0]["player"] == "Hidden"  # 2% owned
    assert by_key["value"]["players"][0]["player"] == "Cheap"       # 21 / 4.5
    assert "GW6" in by_key["fixtures"]["players"][0]["note"]

    # Ownership is what makes 'Hot' ineligible as a differential, not quality.
    assert "Hot" not in {p["player"] for p in by_key["differential"]["players"]}


def test_a_player_already_owned_is_never_a_buy_target():
    elements, projections = _buy_pool()

    lists = advice.buy_shortlists(
        elements, projections, {}, {2}, [], horizon=5, teams=TEAMS
    )

    assert "Hot" not in {p["player"] for group in lists for p in group["players"]}


def test_an_injured_player_is_never_a_buy_target():
    elements, projections = _buy_pool()
    elements[2] = {**elements[2], "status": "i"}

    lists = advice.buy_shortlists(
        elements, projections, {}, set(), [], horizon=5, teams=TEAMS
    )

    assert "Hot" not in {p["player"] for group in lists for p in group["players"]}


# ------------------------------------------------------------ the calendar


def _fixture(event, home, away):
    return {"event": event, "team_h": home, "team_a": away}


def test_an_ordinary_calendar_has_no_doubles_and_no_blanks():
    """Which is the honest answer for most of a season's first half, and the
    one the old advice could not give — it described doubles in the abstract
    and left the reader to assume one was coming."""
    fixtures = [
        _fixture(gw, 1, 2) for gw in range(1, 6)
    ] + [_fixture(gw, 3, 4) for gw in range(1, 6)]

    shape = advice.schedule_shape(fixtures, from_gameweek=1)

    assert shape["doubles"] == []
    assert shape["blanks"] == []
    assert shape["next_double"] is None


def test_a_scheduled_double_is_named_with_its_gameweek_and_clubs():
    fixtures = [
        _fixture(1, 1, 2),
        _fixture(1, 3, 4),
        _fixture(2, 1, 3),
        _fixture(2, 2, 4),
        _fixture(2, 1, 4),  # team 1 and team 4 play twice in GW2
    ]

    shape = advice.schedule_shape(fixtures, from_gameweek=1)

    assert shape["next_double"] == 2
    assert shape["doubles"][0]["teams"] == [1, 4]


def test_a_blank_names_the_clubs_that_are_missing():
    fixtures = [
        _fixture(1, 1, 2),
        _fixture(1, 3, 4),
        _fixture(2, 1, 2),  # 3 and 4 blank
    ]

    shape = advice.schedule_shape(fixtures, from_gameweek=1)

    assert shape["next_blank"] == 2
    assert shape["blanks"][0]["teams"] == [3, 4]


def test_an_unscheduled_fixture_is_not_a_blank():
    """`event: null` means the fixture has no date yet, not that nobody plays.

    Reading it as a blank would invent a Free Hit gameweek out of a scheduling
    gap — advice built on a null.
    """
    fixtures = [
        _fixture(1, 1, 2),
        _fixture(1, 3, 4),
        _fixture(None, 1, 3),
    ]

    shape = advice.schedule_shape(fixtures, from_gameweek=1)

    assert shape["blanks"] == []
    assert shape["unscheduled_fixtures"] == 1


def test_a_gameweek_nobody_plays_is_not_twenty_blanks():
    """Only gameweeks with some fixtures can have clubs missing from them."""
    fixtures = [_fixture(1, 1, 2), _fixture(1, 3, 4), _fixture(3, 1, 2)]

    shape = advice.schedule_shape(fixtures, from_gameweek=1)

    # GW2 has no fixtures at all and so is absent, not a blank for everyone.
    assert [b["gameweek"] for b in shape["blanks"]] == [3]


def test_past_gameweeks_are_not_advice():
    fixtures = [_fixture(1, 1, 2), _fixture(5, 1, 2), _fixture(5, 1, 3)]

    shape = advice.schedule_shape(fixtures, from_gameweek=3)

    assert shape["next_double"] == 5


# ------------------------------------------------------------- chip wording


def test_a_chip_says_no_double_is_scheduled_rather_than_going_quiet():
    """Silence reads as 'no double'; a generic line about doubles reads as
    'there is one'. Only naming which is true is neither."""
    note = advice.chip_schedule_note("3xc", {"doubles": [], "blanks": []}, TEAMS)

    assert "No double gameweek is scheduled" in note


def test_a_chip_names_the_double_when_there_is_one():
    schedule = {"doubles": [{"gameweek": 25, "teams": [1, 2]}], "blanks": []}

    note = advice.chip_schedule_note("bboost", schedule, TEAMS)

    assert "GW25" in note
    assert "MUN" in note and "ARS" in note


def test_free_hit_is_told_about_blanks_not_doubles():
    schedule = {
        "doubles": [{"gameweek": 25, "teams": [1]}],
        "blanks": [{"gameweek": 29, "teams": [1, 2]}],
    }

    note = advice.chip_schedule_note("freehit", schedule, TEAMS)

    assert "GW29" in note
    assert "GW25" not in note


def test_a_long_list_of_clubs_is_summarised_rather_than_printed():
    schedule = {"doubles": [], "blanks": [{"gameweek": 29, "teams": list(range(1, 12))}]}
    teams = {i: {"short_name": f"T{i}"} for i in range(1, 12)}

    note = advice.chip_schedule_note("freehit", schedule, teams)

    assert "and 7 more" in note
