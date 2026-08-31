"""The search: tallies, intervals, candidate selection, and determinism."""

import math

import numpy as np
import pytest

from draftsim.config import Budget, League, SimConfig, PLAN_ROUNDS, QB, RB, TE, WR
from draftsim.plans import ARCHETYPES, archetype_plans, is_legal
from draftsim.search import (Tally, candidates_for, round_scores, run_baseline,
                             run_stage1, run_stage2, wilson)
from draftsim.season import SeasonResult

TINY = Budget(stage1=120, stage2=40, candidates=10, min_stage1_samples=1)


def cfg_for(slots=(1, 12), jobs=1, metric="playoffs"):
    return SimConfig(league=League(), budget=TINY, metric=metric, slots=slots,
                     jobs=jobs, baseline=60)


# ---- Tally and Wilson -----------------------------------------------------

def result(playoffs=True, title=False, wins=8.0, points=1600.0):
    return SeasonResult(made_playoffs=playoffs, won_title=title, wins=wins,
                        points=points)


def test_tally_averages_what_it_is_given():
    t = Tally()
    t.add(result(True, True, 10, 1800))
    t.add(result(False, False, 4, 1400))
    assert t.n == 2
    assert t.playoff_rate == 0.5
    assert t.title_rate == 0.5
    assert t.mean_wins == 7.0
    assert t.mean_points == 1600.0


def test_an_empty_tally_reports_nan_rather_than_dividing_by_zero():
    t = Tally()
    assert math.isnan(t.playoff_rate)
    assert math.isnan(t.rate("playoffs"))


def test_merging_tallies_is_the_same_as_adding_in_one_place():
    a, b, both = Tally(), Tally(), Tally()
    for r in (result(True), result(False), result(True, True)):
        both.add(r)
    a.add(result(True))
    b.add(result(False))
    b.add(result(True, True))
    a.merge(b)
    assert (a.n, a.playoffs, a.titles) == (both.n, both.playoffs, both.titles)


def test_wilson_stays_inside_zero_and_one():
    """The reason it is not the textbook normal interval: rates here get small, and
    the normal interval cheerfully reports a lower bound below zero."""
    low, high = wilson(0, 500)
    assert low == 0.0 and 0 < high < 0.05
    low, high = wilson(500, 500)
    assert high == pytest.approx(1.0) and 0.98 < low < 1.0
    low, high = wilson(3, 1500)
    assert 0 <= low < 0.002 < high < 0.02


def test_wilson_narrows_as_the_sample_grows():
    narrow = wilson(750, 1500)
    wide = wilson(50, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_on_nothing_is_nan():
    low, high = wilson(0, 0)
    assert math.isnan(low) and math.isnan(high)


# ---- end to end, single process -------------------------------------------

@pytest.fixture(scope="module")
def tiny_run(request):
    from draftsim.pool import load_pool
    pool = load_pool()
    cfg = cfg_for()
    baseline = run_baseline(pool, cfg, jobs=1)
    stage1 = run_stage1(pool, cfg, jobs=1)
    stage2 = run_stage2(pool, cfg, stage1, jobs=1)
    return pool, cfg, baseline, stage1, stage2


def test_stage1_spends_exactly_its_budget(tiny_run):
    _, cfg, _, stage1, _ = tiny_run
    for slot in cfg.slots:
        assert stage1[slot].overall.n == cfg.budget.stage1
        assert sum(t.n for t in stage1[slot].plans.values()) == cfg.budget.stage1
        assert int(stage1[slot].round_n.sum()) == cfg.budget.stage1 * cfg.plan_rounds


def test_stage1_only_ever_sampled_legal_plans(tiny_run):
    _, _, _, stage1, _ = tiny_run
    for result in stage1.values():
        assert all(is_legal(p) for p in result.plans)


def test_round_scores_have_a_value_for_every_position(tiny_run):
    _, cfg, _, stage1, _ = tiny_run
    for slot in cfg.slots:
        scores = round_scores(stage1[slot], cfg)
        assert scores.shape == (cfg.plan_rounds, 6)
        assert np.isfinite(scores[:, :4]).all()


def test_every_named_archetype_gets_a_row(tiny_run):
    """Unconditionally, whether stage 1 liked it or not.

    A reader's first question is always "how does the thing I already do score", and a
    winner nobody has a name for is far more convincing beside a measured Zero-RB than
    on its own.
    """
    _, cfg, _, stage1, stage2 = tiny_run
    wanted = set(archetype_plans(cfg.plan_rounds).values())
    for slot in cfg.slots:
        measured = {r.plan for r in stage2[slot]}
        assert wanted <= measured, wanted - measured


def test_candidates_are_legal_distinct_and_capped(tiny_run):
    _, cfg, _, stage1, _ = tiny_run
    for slot in cfg.slots:
        picks = candidates_for(stage1[slot], cfg)
        assert len(picks) == len(set(picks))
        assert all(is_legal(p) for p in picks)
        assert len(picks) >= len(archetype_plans(cfg.plan_rounds))


def test_stage2_spends_its_budget_on_each_candidate(tiny_run):
    _, cfg, _, _, stage2 = tiny_run
    for slot in cfg.slots:
        for result in stage2[slot]:
            assert result.tally.n == cfg.budget.stage2


def test_stage2_is_ranked_best_first(tiny_run):
    _, cfg, _, _, stage2 = tiny_run
    for slot in cfg.slots:
        rates = [r.tally.rate(cfg.metric) for r in stage2[slot]]
        assert rates == sorted(rates, reverse=True)


def test_anchors_account_for_every_round_of_every_draft(tiny_run):
    """The anchor counts are what turns an abstract plan into names a drafter can use.
    If they do not sum to the drafts spent, some rounds are missing from the report."""
    _, cfg, _, _, stage2 = tiny_run
    for slot in cfg.slots:
        for result in stage2[slot]:
            assert sum(result.anchors.values()) == cfg.budget.stage2 * cfg.league.rounds
            rounds = {r for r, _ in result.anchors}
            assert rounds == set(range(cfg.league.rounds))


def test_the_baseline_is_a_coin_flip_averaged_over_slots(tiny_run):
    """The run's own calibration check, in the run's own output.

    Twelve identical ADP drafters must average a 50% playoff rate. The baseline
    measures all twelve slots from every draft, so this is exact, not statistical.
    """
    _, cfg, baseline, _, _ = tiny_run
    assert set(baseline) == set(range(1, cfg.league.teams + 1))
    total = sum(t.n for t in baseline.values())
    made = sum(t.playoffs for t in baseline.values())
    assert made / total == pytest.approx(
        cfg.league.playoff_teams / cfg.league.teams, abs=1e-9)
    for t in baseline.values():
        assert t.n == cfg.baseline


# ---- determinism ----------------------------------------------------------

def test_the_same_seed_gives_the_same_answer(pool):
    cfg = cfg_for(slots=(4,))
    a = run_stage1(pool, cfg, jobs=1)
    b = run_stage1(pool, cfg, jobs=1)
    assert a[4].plans.keys() == b[4].plans.keys()
    assert np.array_equal(a[4].round_hit, b[4].round_hit)
    assert a[4].overall.playoffs == b[4].overall.playoffs


def test_a_different_seed_gives_a_different_answer(pool):
    one = run_stage1(pool, cfg_for(slots=(4,)), jobs=1)
    other_cfg = SimConfig(league=League(), budget=Budget(**{**TINY.__dict__,
                                                           "seed": TINY.seed + 1}),
                          slots=(4,), jobs=1, baseline=60)
    two = run_stage1(pool, other_cfg, jobs=1)
    assert not np.array_equal(one[4].round_hit, two[4].round_hit)


def test_results_do_not_depend_on_how_many_processes_ran(pool):
    """Seeding is keyed to each task's own coordinates, not to a shared counter, so a
    run is reproducible at any --jobs. Without that, every result would be unrepeatable
    on a different machine.
    """
    cfg_one = cfg_for(slots=(2,), jobs=1)
    cfg_many = cfg_for(slots=(2,), jobs=3)
    a = run_baseline(pool, cfg_one, jobs=1)
    b = run_baseline(pool, cfg_many, jobs=3)
    assert [a[s].playoffs for s in sorted(a)] == [b[s].playoffs for s in sorted(b)]
    assert [round(a[s].points, 6) for s in sorted(a)] == \
           [round(b[s].points, 6) for s in sorted(b)]


def test_the_candidate_budget_is_actually_spent(tiny_run):
    """It was not, and that was a silent waste of a third of stage 2.

    At a full budget almost no individual plan clears the stage-1 sample floor -- 12,000
    drafts over 25,174 plans is half a sample each -- so the source that was supposed
    to fill the last third of the list contributed nothing and the run quietly measured
    16 candidates instead of 24.
    """
    _, cfg, _, stage1, stage2 = tiny_run
    for slot in cfg.slots:
        picks = candidates_for(stage1[slot], cfg)
        assert len(picks) == cfg.budget.candidates, len(picks)
        assert len(stage2[slot]) == cfg.budget.candidates


def test_a_shape_nomination_really_has_that_shape(tiny_run):
    """Source 3 conditions on a whole opening multiset, so the plan it returns has to
    have it -- otherwise it is just source 2 with extra steps."""
    from draftsim.search import SHAPE_ROUNDS, best_plan_for_shape
    _, cfg, _, stage1, _ = tiny_run
    slot = cfg.slots[0]
    scores = round_scores(stage1[slot], cfg)
    for shape in list(stage1[slot].shapes)[:6]:
        plan = best_plan_for_shape(shape, scores, cfg.plan_rounds)
        if plan is None:
            continue
        assert tuple(sorted(plan[:SHAPE_ROUNDS])) == tuple(sorted(shape))
        assert is_legal(plan)


def test_an_impossible_shape_returns_nothing_rather_than_raising():
    """Three quarterbacks in four rounds cannot open a legal plan."""
    from draftsim.search import best_plan_for_shape
    scores = np.zeros((PLAN_ROUNDS, 6))
    assert best_plan_for_shape((QB, QB, QB, QB), scores, PLAN_ROUNDS) is None


def test_grading_one_plan_reproduces_the_row_a_full_run_would_print(pool):
    """--plan has its own entry point but must not be its own measurement.

    It shares stage 2's worker function and its seeding, so grading a plan directly
    has to give bit-identical numbers to that plan's row in a full run. Otherwise
    "grade my plan" and "search for the best plan" would quietly disagree about the
    same plan.
    """
    from draftsim.search import run_plan
    cfg = cfg_for(slots=(5,))
    plan = ARCHETYPES["Robust-RB"]
    stage1 = run_stage1(pool, cfg, jobs=1)
    stage2 = run_stage2(pool, cfg, stage1, jobs=1)
    from_search = next(r for r in stage2[5] if r.plan == plan)
    direct = run_plan(pool, cfg, plan, jobs=1)[5][0]
    assert direct.tally.n == from_search.tally.n
    assert direct.tally.playoffs == from_search.tally.playoffs
    assert direct.tally.titles == from_search.tally.titles
    assert direct.anchors == from_search.anchors


# ---- stage 3 and the tie test ---------------------------------------------

def tally(hits, n):
    t = Tally()
    t.n, t.playoffs = n, hits
    return t


def test_identical_rates_are_indistinguishable():
    from draftsim.search import indistinguishable
    assert indistinguishable(tally(500, 1000), tally(500, 1000))


def test_a_large_gap_is_distinguished():
    from draftsim.search import indistinguishable
    assert not indistinguishable(tally(600, 1000), tally(450, 1000))


def test_more_drafts_separate_a_gap_that_fewer_could_not():
    """The entire argument for stage 3, in one assertion.

    A four-point difference is invisible at a few hundred drafts and clear at several
    thousand. Stage 2 sits on the wrong side of that line for the gaps between the
    leading plans, which is why it screens rather than decides.
    """
    from draftsim.search import indistinguishable
    assert indistinguishable(tally(270, 500), tally(250, 500))
    assert not indistinguishable(tally(2700, 5000), tally(2500, 5000))


def test_the_difference_test_is_less_conservative_than_overlapping_intervals():
    """The claim the docstring makes, checked rather than asserted in prose.

    Comparing whether two plans' own intervals overlap behaves like a test at roughly
    the 0.83 level, so it calls ties that a proper test on the difference separates.
    Finding one such case proves the two tests are not interchangeable and that the
    conservative one would have hidden a real difference.
    """
    from draftsim.search import indistinguishable, wilson
    # A 2.4-point gap at 5,000 drafts each. The individual half-widths are ~1.39
    # points, so the intervals still touch; the standard error of the difference is
    # ~1.0 point, so 2.4 clears 1.96 of them. Everything between 1.96 and 2.77 points
    # lands in this window where the two tests disagree.
    a, b = tally(2560, 5000), tally(2440, 5000)
    a_low, a_high = wilson(a.playoffs, a.n)
    b_low, b_high = wilson(b.playoffs, b.n)
    overlap = a_low <= b_high and b_low <= a_high
    assert overlap, "expected the individual intervals to overlap here"
    assert not indistinguishable(a, b), "yet the difference is significant"


def test_a_tally_with_no_drafts_ties_with_anything():
    from draftsim.search import indistinguishable
    assert indistinguishable(Tally(), tally(900, 1000))


def test_the_leading_group_always_contains_the_leader(tiny_run):
    from draftsim.search import leading_group
    _, cfg, _, _, stage2 = tiny_run
    for slot in cfg.slots:
        group = leading_group(stage2[slot])
        assert group and group[0] is stage2[slot][0]
        assert len(group) <= len(stage2[slot])


def test_the_leading_group_is_measured_against_the_leader_not_transitively():
    """A chain of pairwise ties can run arbitrarily far down a list -- plan 1 ties 2
    ties 3 while 1 and 3 are plainly separated. 'Tied with the best' is the claim a
    reader needs, so the comparison is always against the leader."""
    from draftsim.search import Stage2Result, leading_group
    def result(hits, n):
        r = Stage2Result(slot=1, plan=(0,))
        r.tally = tally(hits, n)
        return r
    chain = [result(3000, 5000), result(2900, 5000), result(2800, 5000),
             result(2600, 5000)]
    group = leading_group(chain)
    assert chain[0] in group
    assert chain[-1] not in group


def test_stage3_measures_only_the_finalists_and_at_full_depth(pool):
    from draftsim.search import run_stage3
    cfg = SimConfig(league=League(),
                    budget=Budget(stage1=100, stage2=30, stage3=90, candidates=9,
                                  finalists=3, min_stage1_samples=1),
                    slots=(3,), jobs=1, baseline=24)
    stage1 = run_stage1(pool, cfg, jobs=1)
    stage2 = run_stage2(pool, cfg, stage1, jobs=1)
    final = run_stage3(pool, cfg, stage2, jobs=1)
    assert len(final[3]) == cfg.budget.finalists
    assert {r.plan for r in final[3]} <= {r.plan for r in stage2[3]}
    for r in final[3]:
        assert r.tally.n == cfg.budget.stage3


def test_stage3_redraws_rather_than_accumulating_on_stage2(pool):
    """Fresh draws, and the reason matters.

    Stage 2 promoted these plans *because* their observed rates were high, so its
    counts for exactly those plans are biased upward by the selection. Adding them in
    would carry the winner's curse into the final number. Checked by measuring the
    same plan at the same budget under both stage tags: the results must differ.
    """
    from draftsim.search import _measure_chunk, _init_worker
    cfg = SimConfig(league=League(),
                    budget=Budget(stage1=50, stage2=40, stage3=40, candidates=9,
                                  finalists=3, min_stage1_samples=1),
                    slots=(3,), jobs=1, baseline=24)
    _init_worker(pool, cfg)
    task = (3, ARCHETYPES["Robust-RB"], 40, cfg.budget.seed, 0)
    screen = _measure_chunk(task, 2)
    decide = _measure_chunk(task, 3)
    assert screen.tally.n == decide.tally.n == 40
    assert screen.anchors != decide.anchors


def test_recommend_names_one_plan_and_the_group_it_came_from(tiny_run):
    from draftsim.search import recommend
    _, cfg, _, stage1, stage2 = tiny_run
    for slot in cfg.slots:
        best, group, reason = recommend(stage2[slot], cfg)
        assert best in group
        assert reason
        if len(group) == 1:
            assert best is stage2[slot][0]


def test_recommend_on_an_empty_slot_does_not_raise():
    from draftsim.search import recommend
    best, group, reason = recommend([], cfg_for(slots=(1,)))
    assert best is None and group == []


def test_a_tie_is_broken_on_wins_not_on_confounded_marginals():
    """The bug that produced a bench quarterback in round 8.

    The old tie-break summed stage-1 per-round marginals, and those are confounded:
    P(playoffs | QB in round 8) is high because such a plan spent rounds 1-7 on skill
    players, not because a round-8 quarterback is good. Mean wins is measured on the
    same drafts, is continuous rather than binary, and is the same objective -- a berth
    is won on wins -- rather than a proxy for it.
    """
    from draftsim.search import Stage2Result, recommend

    def result(plan, hits, n, wins):
        r = Stage2Result(slot=1, plan=plan)
        r.tally = Tally(n=n, playoffs=hits, wins=wins * n)
        return r

    # Two plans a coin-toss apart on playoff rate, clearly apart on wins.
    lead = result((RB,) * 8, 2510, 5000, 7.10)
    other = result((WR,) * 8, 2500, 5000, 7.35)
    best, group, reason = recommend([lead, other])
    assert len(group) == 2
    assert best is other, "the tie must go to the plan that wins more games"
    assert "wins" in reason


def test_points_is_not_the_tie_break():
    """Points feed wins and wins feed berths, but the chain is not tight: two plans with
    identical mean points can differ in week-to-week variance and so in how often those
    points land as wins. Wins already contains that, so points must not override it."""
    from draftsim.search import Stage2Result, recommend

    def result(plan, hits, n, wins, points):
        r = Stage2Result(slot=1, plan=plan)
        r.tally = Tally(n=n, playoffs=hits, wins=wins * n, points=points * n)
        return r

    fewer_points_more_wins = result((RB,) * 8, 2500, 5000, 7.40, 1700.0)
    more_points_fewer_wins = result((WR,) * 8, 2500, 5000, 7.00, 1800.0)
    best, _, _ = recommend([fewer_points_more_wins, more_points_fewer_wins])
    assert best is fewer_points_more_wins


def test_mutations_are_legal_single_round_swaps():
    from draftsim.plans import is_legal, mutations
    plan = ARCHETYPES["Robust-RB"]
    got = mutations(plan)
    assert got
    for m in got:
        assert is_legal(m)
        assert len(m) == len(plan)
        assert sum(1 for a, b in zip(plan, m) if a != b) == 1
    assert plan not in got
    assert len(set(got)) == len(got)


def test_the_mutation_pass_measures_plans_the_nomination_missed(pool):
    """The other half of the same bug: the better plan was never even a candidate.

    At slot 1 a plan two swaps from the screen leader beat it by 2.2 points and was
    never nominated, because an additive score cannot see a pair of swaps that only
    pays off together. The mutation pass walks the neighbourhood out from plans that
    already screened well.
    """
    from draftsim.search import mutation_candidates, run_mutations
    # mutate_passes=1 on purpose. The arithmetic below counts one pass's worth of
    # neighbours; at the default of 2 a second pass runs whenever the first one
    # promotes a new leader, which adds rows this assertion does not account for.
    # Whether that happens is a property of the rankings export, not of the code,
    # so leaving it at the default made the test pass or fail with the week's data.
    cfg = SimConfig(league=League(),
                    budget=Budget(stage1=80, stage2=40, stage2b=40, stage3=40,
                                  candidates=9, mutate_top=1, finalists=3,
                                  mutate_passes=1, min_stage1_samples=1),
                    slots=(1,), jobs=1, baseline=24)
    stage1 = run_stage1(pool, cfg, jobs=1)
    screen = run_stage2(pool, cfg, stage1, jobs=1)
    fresh = mutation_candidates(screen, cfg)[1]
    assert fresh, "the leader must have legal neighbours not already screened"
    assert all(p not in {r.plan for r in screen[1]} for p in fresh)

    merged = run_mutations(pool, cfg, screen, jobs=1)
    assert len(merged[1]) == len(screen[1]) + len(fresh)
    rates = [r.tally.rate(cfg.metric) for r in merged[1]]
    assert rates == sorted(rates, reverse=True), "merged screen must stay ranked"


def test_mutate_top_zero_skips_the_pass(pool):
    from draftsim.search import run_mutations
    cfg = SimConfig(league=League(),
                    budget=Budget(stage1=80, stage2=40, stage2b=40, stage3=40,
                                  candidates=9, mutate_top=0, finalists=3,
                                  min_stage1_samples=1),
                    slots=(1,), jobs=1, baseline=24)
    stage1 = run_stage1(pool, cfg, jobs=1)
    screen = run_stage2(pool, cfg, stage1, jobs=1)
    assert run_mutations(pool, cfg, screen, jobs=1) is screen
