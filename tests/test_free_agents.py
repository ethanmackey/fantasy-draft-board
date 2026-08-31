"""The free-agent pool: what happens when the bench genuinely cannot cover a slot.

These tests exist because of a specific bug. With bye weeks modelled but no waiver
wire, a roster carrying one quarterback started a *zero* at quarterback in his
starter's bye week -- and the only way the draft could avoid that was to spend a
pick on a backup. The search found that trade and took it, recommending a bench
quarterback in round 8 over a second starting receiver, which no real manager would
ever do: they stream a quarterback for that one week, free.

The bench is not the issue and never was. The lineup solver already draws starters
from all sixteen rostered players. The issue is the week when nothing on the roster
is available at all.
"""

import numpy as np
import pytest

from draftsim.config import DST, K, League, POSITIONS, QB, RB, TE, WR
from draftsim.roster import lineup_score_one, lineup_scores

LEAGUE = League()


def test_an_unfillable_slot_is_streamed_not_forfeited():
    """The bug, in one assertion.

    A roster with one quarterback, unavailable this week. The quarterback slot must
    score what a waiver-wire streamer scores, not zero. Everything else about the
    lineup is filled from the roster as normal.
    """
    order = [QB, RB, RB, WR, WR, TE, K, DST]
    ppg = [20, 10, 9, 8, 7, 6, 5, 4]
    avail = [False] + [True] * 7
    free = {QB: 13.0}
    got = lineup_score_one(order, avail, ppg, LEAGUE, free_agents=free)
    assert got == pytest.approx(sum(ppg) - 20 + 13.0)


def test_the_bench_is_used_before_the_wire():
    """A rostered backup beats a streamer, so the wire must be the last resort.

    If this failed, the free-agent pool would be papering over the bench rather than
    backing it up, and every plan that drafted depth would be undervalued.
    """
    order = [QB, QB, RB, RB, WR, WR, TE, K, DST]
    ppg = [20, 17, 10, 9, 8, 7, 6, 5, 4]
    avail = [False] + [True] * 8
    free = {QB: 13.0}
    got = lineup_score_one(order, avail, ppg, LEAGUE, free_agents=free)
    # QB2 at 17 starts, not the 13-point streamer, and nothing is forfeited.
    assert got == pytest.approx(sum(ppg) - 20)


def test_with_no_wire_supplied_the_slot_still_scores_zero():
    """The old behaviour has to remain reachable, because it is the pessimistic
    bound and the tests that predate the wire assert against it."""
    order = [QB, RB, RB, WR, WR, TE, K, DST]
    ppg = [20, 10, 9, 8, 7, 6, 5, 4]
    avail = [False] + [True] * 7
    assert lineup_score_one(order, avail, ppg, LEAGUE) == pytest.approx(sum(ppg) - 20)


def test_the_wire_fills_an_empty_flex_slot_too():
    """A manager fields a full lineup. An empty flex is streamed like any other slot."""
    order = [QB, RB, RB, WR, WR, TE, K, DST]      # exactly the mandatory slots
    ppg = [20, 10, 9, 8, 7, 6, 5, 4]
    avail = [True] * 8
    free = {RB: 6.0, WR: 5.0, TE: 4.0}
    got = lineup_score_one(order, avail, ppg, LEAGUE, free_agents=free)
    # Two flex slots, both filled from the wire's best flex-eligible option.
    assert got == pytest.approx(sum(ppg) + 2 * 6.0)


def test_vectorised_solver_matches_the_readable_one_with_a_wire(rng):
    """The fast path is what the season actually calls, so it has to agree."""
    teams, roster, weeks = 12, 16, 17
    pos = rng.integers(0, len(POSITIONS), size=(teams, roster))
    ppg = rng.uniform(0, 25, size=(teams, roster))
    order = np.argsort(-ppg, axis=1)
    pos = np.take_along_axis(pos, order, axis=1)
    avail = rng.random((teams, roster, weeks)) > 0.45      # plenty of holes
    points = rng.uniform(0, 40, size=(teams, roster, weeks))
    wire = rng.uniform(2, 12, size=(teams, len(POSITIONS), weeks))
    wire_mean = wire.mean(axis=2)

    fast = lineup_scores(pos, avail, points, LEAGUE, wire=wire, wire_mean=wire_mean)
    for t in range(teams):
        for w in range(weeks):
            slow = lineup_score_one(
                pos[t], avail[t, :, w], points[t, :, w], LEAGUE,
                free_agents={p: wire[t, p, w] for p in range(len(POSITIONS))},
                free_agent_mean={p: wire_mean[t, p] for p in range(len(POSITIONS))})
            assert fast[t, w] == pytest.approx(slow), (t, w)


# ---- replacement level, and the experiment that decided it -------------------

def test_replacement_can_differ_per_seat():
    """Needed for the experiment that settled which definition drafts better: one
    drafter pricing against the wire while eleven price against the board."""
    import numpy as np
    from draftsim.draft import DraftEngine
    from draftsim.pool import load_pool
    pool = load_pool()
    levels = np.broadcast_to(pool.replacement, (LEAGUE.teams, len(POSITIONS))).copy()
    levels[0] = 0.0
    engine = DraftEngine(pool, LEAGUE, replacement=levels)
    assert engine.replacement.shape == (LEAGUE.teams, len(POSITIONS))
    assert (engine.replacement[0] == 0).all()
    assert (engine.replacement[1] == pool.replacement).all()


def test_a_replacement_vector_of_the_wrong_shape_is_rejected():
    import numpy as np
    from draftsim.draft import DraftEngine
    from draftsim.pool import load_pool
    pool = load_pool()
    with pytest.raises(ValueError, match="replacement must be"):
        DraftEngine(pool, LEAGUE, replacement=np.zeros((3, 2)))


def test_calibration_converges_on_the_wire_level():
    """Replacement is a fixed point: the wire depends on how everyone drafts, and how
    everyone drafts depends on the wire. Two passes should land near the measured wire
    and well below the board's 'last starter in the league' at running back."""
    from draftsim.config import SimConfig
    from draftsim.pool import load_pool
    from draftsim.search import calibrate_replacement
    pool = load_pool()
    cfg = SimConfig(slots=(1,))
    levels = calibrate_replacement(pool, cfg)
    assert len(levels) == len(POSITIONS)
    # Backs are drafted far deeper than the board's replacement, so the wire is much
    # lower there; quarterbacks are barely drafted past their starters, so it is not.
    assert levels[RB] < 0.6 * pool.replacement[RB]
    assert levels[QB] > 0.8 * pool.replacement[QB]
    assert all(v >= 0 for v in levels)


def test_board_replacement_is_still_reachable():
    from draftsim.config import SimConfig
    from draftsim.pool import load_pool
    from draftsim.search import calibrate_replacement
    pool = load_pool()
    got = calibrate_replacement(pool, SimConfig(replacement_from_wire=False))
    assert got == tuple(float(x) for x in pool.replacement)


def test_the_replacement_fixed_point_is_actually_converged():
    """Two passes looked converged and was not quite.

    From the board's levels the passes move 7.05, 1.97, 0.12, 0.01 and then nothing, so
    the default of four is past the point where another pass changes anything. Asserted
    by running one extra pass and confirming it barely moves -- if convergence ever
    breaks, the reported replacement level would silently depend on the pass count.
    """
    import numpy as np
    from draftsim.config import (REPLACEMENT_CALIBRATION_PASSES, SimConfig)
    from draftsim.pool import load_pool
    from draftsim.search import calibrate_replacement
    pool = load_pool()
    settled = np.asarray(calibrate_replacement(pool, SimConfig()))

    import draftsim.config as cfgmod
    original = cfgmod.REPLACEMENT_CALIBRATION_PASSES
    try:
        cfgmod.REPLACEMENT_CALIBRATION_PASSES = original + 2
        further = np.asarray(calibrate_replacement(pool, SimConfig()))
    finally:
        cfgmod.REPLACEMENT_CALIBRATION_PASSES = original
    assert np.abs(further - settled).max() < 0.35, (settled, further)
