"""Roster legality, bench depth, and the two lineup solvers agreeing."""

import numpy as np
import pytest

from draftsim.config import DST, K, League, POSITIONS, QB, RB, TE, WR
from draftsim.roster import (allowed_positions, bench_depth, lineup_score_one,
                             lineup_scores, starter_need)

LEAGUE = League()


def counts(**kwargs):
    out = np.zeros(len(POSITIONS), dtype=np.int16)
    for name, value in kwargs.items():
        out[POSITIONS.index(name)] = value
    return out


# ---- starter_need ----------------------------------------------------------

def test_empty_roster_needs_every_starting_slot():
    assert starter_need(counts(), LEAGUE) == LEAGUE.mandatory == 10


def test_flex_is_satisfied_by_surplus_not_by_drafting_for_it():
    """Three backs and three receivers covers both flex spots without a flex pick.

    The whole reason the flex is not split by a fixed share: who fills it is decided
    on merit, and a roster that is already deep at receiver has effectively filled it.
    """
    full = counts(QB=1, RB=3, WR=3, TE=1, K=1, DST=1)
    assert starter_need(full, LEAGUE) == 0


def test_a_missing_tight_end_still_counts_as_a_need():
    assert starter_need(counts(QB=1, RB=2, WR=4, K=1, DST=1), LEAGUE) == 1


# ---- allowed_positions ----------------------------------------------------

def test_reserved_rounds_allow_only_the_kicker_then_the_defense():
    """Rounds 15 and 16 are spoken for; nothing else may be taken in them.

    Reserving the rounds rather than merely permitting K and DST late is what
    guarantees every team finishes with exactly one of each, and it is what keeps
    them out of the plan space.
    """
    empty = counts(QB=1, RB=4, WR=5, TE=2)      # 12 picks in, skill phase done
    at_15 = allowed_positions(empty, 14, LEAGUE)
    assert at_15[K] and not at_15[DST] is False   # K legal
    assert not any(at_15[p] for p in (QB, RB, WR, TE))

    with_k = counts(QB=1, RB=4, WR=5, TE=2, K=1)
    at_16 = allowed_positions(with_k, 15, LEAGUE)
    assert at_16[DST] and not at_16[K]
    assert not any(at_16[p] for p in (QB, RB, WR, TE))


def test_a_position_at_its_cap_is_closed():
    at_cap = counts(QB=2, RB=1, WR=1)
    assert not allowed_positions(at_cap, 4, LEAGUE)[QB]


def test_kickers_are_illegal_before_the_reserved_rounds():
    assert not allowed_positions(counts(), 0, LEAGUE)[K]
    assert not allowed_positions(counts(QB=1, RB=2, WR=2, TE=1), 6, LEAGUE)[DST]


def test_starter_guard_forces_the_last_skill_picks_to_fill_holes():
    """With one skill pick left and no tight end, the pick must be a tight end.

    Without this, ADP bots finish the draft with no tight end and start a zero every
    week -- not a mistake a real drafter makes, and it would inflate every plan that
    competes for tight ends.
    """
    # 13 picks made, 14 skill rounds: one skill pick left, and no TE on the roster.
    roster = counts(QB=1, RB=5, WR=7)
    allowed = allowed_positions(roster, 13, LEAGUE)
    assert allowed[TE]
    assert not allowed[QB] and not allowed[RB] and not allowed[WR]


def test_starter_guard_stays_out_of_the_way_early():
    """Round 1 has thirteen picks left and ten holes; nothing should be restricted."""
    allowed = allowed_positions(counts(), 0, LEAGUE)
    assert allowed[QB] and allowed[RB] and allowed[WR] and allowed[TE]


# ---- bench_depth ----------------------------------------------------------

def test_first_players_at_a_position_are_starters():
    assert bench_depth(counts(), RB, LEAGUE) == 0
    assert bench_depth(counts(RB=1), RB, LEAGUE) == 0
    assert bench_depth(counts(QB=0), QB, LEAGUE) == 0


def test_backup_quarterback_is_bench_depth_one():
    """The case that made bench discounting necessary at all.

    Undiscounted value over replacement recommends a second quarterback in round 11:
    the twentieth-best quarterback still clears quarterback replacement, and the
    arithmetic has no idea only one of them can start.
    """
    assert bench_depth(counts(QB=1), QB, LEAGUE) == 1
    assert bench_depth(counts(QB=2), QB, LEAGUE) == 2


def test_flex_eligible_depth_is_measured_per_position_not_pooled():
    """The bug this definition exists to fix.

    Pooling every flex position's surplus made a fifth receiver count as five deep --
    his team's spare backs and tight ends counted against him too -- which drove his
    discount to nearly nothing and let a backup quarterback win the value comparison
    on raw projected points. Legality pools the flex because it is about filling one
    lineup; depth does not, because it is about how often this player takes the field.
    """
    deep = counts(QB=1, RB=3, WR=3, TE=1)
    # RB and WR each get 2 mandatory + 2 flex = 4 startable, so a fourth is startable.
    assert bench_depth(deep, RB, LEAGUE) == 0
    assert bench_depth(deep, WR, LEAGUE) == 0
    # A fifth receiver is one past startable regardless of how deep the backs are.
    assert bench_depth(counts(QB=1, RB=5, WR=4, TE=1), WR, LEAGUE) == 1
    assert bench_depth(counts(QB=1, RB=1, WR=4, TE=1), WR, LEAGUE) == 1
    # Tight ends get 1 mandatory + 2 flex.
    assert bench_depth(counts(TE=2), TE, LEAGUE) == 0
    assert bench_depth(counts(TE=3), TE, LEAGUE) == 1


def test_a_backup_quarterback_is_deeper_than_a_fourth_running_back():
    """The comparison that has to come out this way, or the value fill hoards.

    A fourth back plays -- byes, injuries, the flex. A second quarterback plays only
    when the first cannot, and a discount that ranks him ahead is what put Jordan Love
    on the roster in round 13.
    """
    roster = counts(QB=1, RB=3, WR=3, TE=1)
    assert bench_depth(roster, QB, LEAGUE) > bench_depth(roster, RB, LEAGUE)


# ---- lineup solvers -------------------------------------------------------

def test_lineup_takes_mandatory_slots_before_the_flex():
    """A tight end fills TE before he is allowed to fill FLEX.

    Greedy from best projection down is optimal here, but only because mandatory
    slots are tried first: sending the best tight end to a flex spot would leave the
    TE slot for a worse one and cost points.
    """
    order = [TE, TE, QB, RB, RB, WR, WR, K, DST]
    ppg = [15, 12, 20, 14, 13, 16, 15, 8, 6]
    avail = [True] * len(order)
    # Everyone plays: TE1 in TE, TE2 in a flex, everything else in its own slot.
    assert lineup_score_one(order, avail, ppg, LEAGUE) == sum(ppg)


def test_flex_never_takes_a_quarterback_or_a_kicker():
    """Two spare quarterbacks and a spare kicker must sit, however well they score."""
    order = [QB, QB, K, K, RB, RB, WR, WR, TE, DST]
    ppg = [30, 29, 28, 27, 5, 4, 3, 2, 1, 1]
    avail = [True] * len(order)
    got = lineup_score_one(order, avail, ppg, LEAGUE)
    # QB1 + K1 + RBx2 + WRx2 + TE + DST, and the two flex slots stay empty because
    # nothing flex-eligible is left over.
    assert got == 30 + 28 + 5 + 4 + 3 + 2 + 1 + 1


def test_an_unfillable_slot_scores_zero_rather_than_raising():
    """A bye colliding with an injury is a fact of a season, not an error."""
    order = [RB, RB, WR, WR, TE, K, DST]      # no quarterback at all
    ppg = [10] * len(order)
    avail = [True] * len(order)
    assert lineup_score_one(order, avail, ppg, LEAGUE) == 70


def test_unavailable_players_are_excluded():
    order = [QB, RB, RB, WR, WR, TE, K, DST]
    ppg = [20, 10, 9, 8, 7, 6, 5, 4]
    avail = [False] + [True] * 7
    assert lineup_score_one(order, avail, ppg, LEAGUE) == sum(ppg) - 20


def test_vectorised_solver_matches_the_readable_one(rng):
    """The only reason it is safe to have two implementations.

    lineup_scores is the one the season simulator calls half a million times and it
    is written for speed; lineup_score_one is written to be read. They have to agree
    on arbitrary rosters or the fast one is just a faster wrong answer.
    """
    teams, roster, weeks = 12, 16, 17
    pos = rng.integers(0, len(POSITIONS), size=(teams, roster))
    ppg = rng.uniform(0, 25, size=(teams, roster))
    # Both solvers require descending projected PPG along the roster axis.
    order = np.argsort(-ppg, axis=1)
    pos = np.take_along_axis(pos, order, axis=1)
    avail = rng.random((teams, roster, weeks)) > 0.2
    points = rng.uniform(0, 40, size=(teams, roster, weeks))

    fast = lineup_scores(pos, avail, points, LEAGUE)
    for t in range(teams):
        for w in range(weeks):
            slow = lineup_score_one(pos[t], avail[t, :, w], points[t, :, w], LEAGUE)
            assert fast[t, w] == pytest.approx(slow)
