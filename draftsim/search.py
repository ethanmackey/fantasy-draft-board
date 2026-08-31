"""The three-stage search, and the process pool that runs it.

The plan space has 25,174 legal members and a full run can afford about a million
drafts. Those numbers are why this narrows in stages rather than measuring
everything: a budget that thin cannot rank individual plans, but it can locate the
good *regions* of the space, and a short list drawn from those regions can then be
measured properly.

**Stage 1 explores.** Random legal plans, thousands per slot. Its output is not a
ranking -- at half a sample per plan a ranking would be a list of whichever plans
got lucky once. Its output is marginals: how a slot does when it takes a receiver in
round 2, or when its first four rounds are two backs and two receivers. Marginals
pool thousands of drafts behind every number, which is what makes them the part of
stage 1 worth believing.

**Stage 2 screens.** Two dozen candidates per slot -- the regions stage 1 liked,
plus every named archetype as a reference row -- each measured well enough to be
sorted into a plausible order. Not well enough to name a winner: at this budget the
interval is wider than the gaps between the leading plans.

**Stage 3 decides.** The best eight per slot, re-measured with four times the
drafts, which narrows the interval to about a point. Fresh drafts rather than an
accumulation on top of stage 2, because stage 2 promoted these plans *for* having
high observed rates and adding its counts in would carry that selection bias -- the
winner's curse -- straight into the final number.

Even then the answer is usually a group rather than a plan, and ``leading_group``
says which. Two failure modes are worth naming because the code is arranged around
them: stage 1's nomination score is additive over rounds, which treats rounds as
independent when they are not -- harmless, because stage 2 re-measures whatever it
is handed -- and a table that names one best plan per slot when six are
indistinguishable is reporting noise as a decision, which is what stage 3 and the
tied-group reporting exist to prevent.
"""

import math
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import POSITIONS
from .draft import DraftEngine
from .plans import (archetype_plans, label_plan, legal_plans, mutations,
                    sample_plan, top_plans_by_round_scores)
from .season import SeasonSimulator

# The first N rounds whose position multiset is reported as a "shape". Four,
# because that is the horizon people actually argue about -- "two backs and two
# receivers to start" is a recognisable claim and an eight-round multiset is not.
SHAPE_ROUNDS = 4


# --------------------------------------------------------------------------
# Accumulators
# --------------------------------------------------------------------------

@dataclass
class Tally:
    """Outcomes for one plan, or one marginal. Mergeable across processes."""

    n: int = 0
    playoffs: int = 0
    titles: int = 0
    points: float = 0.0
    wins: float = 0.0

    def add(self, result):
        self.n += 1
        self.playoffs += bool(result.made_playoffs)
        self.titles += bool(result.won_title)
        self.points += result.points
        self.wins += result.wins

    def merge(self, other):
        self.n += other.n
        self.playoffs += other.playoffs
        self.titles += other.titles
        self.points += other.points
        self.wins += other.wins
        return self

    def rate(self, metric):
        """The metric's value, or nan when nothing has been observed yet."""
        if not self.n:
            return float("nan")
        if metric == "playoffs":
            return self.playoffs / self.n
        if metric == "title":
            return self.titles / self.n
        return self.points / self.n

    @property
    def playoff_rate(self):
        return self.playoffs / self.n if self.n else float("nan")

    @property
    def title_rate(self):
        return self.titles / self.n if self.n else float("nan")

    @property
    def mean_points(self):
        return self.points / self.n if self.n else float("nan")

    @property
    def mean_wins(self):
        return self.wins / self.n if self.n else float("nan")


def indistinguishable(a, b, z=None):
    """Were these two plans actually separated by the drafts spent on them?

    True when a confidence interval on the **difference** of their playoff rates
    contains zero. That is the honest test, and it is not the one everybody reaches
    for: comparing whether the two plans' own intervals *overlap* is far too
    conservative -- it behaves like a test at roughly the 0.83 level rather than 0.05
    -- so it declares ties between plans that are in fact distinguished. Overlapping
    intervals are a reason to look closer, not a verdict.

    Independent samples, so the variances add:

        se = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)

    Two plans with no drafts between them are trivially indistinguishable, which is
    the right answer rather than a division by zero.
    """
    from .config import CONFIDENCE_Z
    z = CONFIDENCE_Z if z is None else z
    if not a.n or not b.n:
        return True
    p1, p2 = a.playoff_rate, b.playoff_rate
    se = math.sqrt(p1 * (1 - p1) / a.n + p2 * (1 - p2) / b.n)
    if se == 0:
        return p1 == p2
    return abs(p1 - p2) < z * se


def leading_group(results, z=None):
    """The results at the top that nobody can tell apart, best first.

    Every plan indistinguishable from the leader, the leader included. This is what
    the report prints instead of a single winner: at these sample sizes the top few
    plans at every slot sit within a couple of points of each other, and naming one
    of them the best is reporting noise as a decision.

    Measured against the leader only, not transitively. A chain of pairwise ties can
    stretch arbitrarily far down a list -- plan 1 ties plan 2 ties plan 3 while 1 and
    3 are clearly separated -- and "tied with the best" is the claim a reader
    actually needs.
    """
    if not results:
        return []
    leader = results[0]
    return [r for r in results if indistinguishable(leader.tally, r.tally, z)]


def wilson(hits, n, z=1.96):
    """Wilson score interval for a proportion. (low, high), both nan if n == 0.

    Wilson rather than the textbook normal interval because the rates here are
    sometimes small -- a title rate near 0.08 on 1,500 draws -- and the normal
    interval is badly behaved near the ends, happily reporting a lower bound below
    zero. Wilson stays inside [0, 1] and is close to exact at these sample sizes.
    """
    if not n:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Stage1Result:
    """One stage-1 chunk. Per-plan tallies plus the marginals that motivate stage 2."""

    slot: int
    plans: dict = field(default_factory=dict)
    # (plan_rounds, positions) counts and outcome sums, for the per-round marginals.
    round_n: np.ndarray = None
    round_hit: np.ndarray = None
    round_metric: np.ndarray = None
    shapes: dict = field(default_factory=dict)
    overall: Tally = field(default_factory=Tally)

    def merge(self, other):
        for plan, tally in other.plans.items():
            self.plans.setdefault(plan, Tally()).merge(tally)
        for shape, tally in other.shapes.items():
            self.shapes.setdefault(shape, Tally()).merge(tally)
        self.round_n = other.round_n if self.round_n is None else self.round_n + other.round_n
        self.round_hit = (other.round_hit if self.round_hit is None
                          else self.round_hit + other.round_hit)
        self.round_metric = (other.round_metric if self.round_metric is None
                             else self.round_metric + other.round_metric)
        self.overall.merge(other.overall)
        return self


@dataclass
class Stage2Result:
    """One candidate plan, measured. ``anchors`` is who actually filled each round."""

    slot: int
    plan: tuple
    tally: Tally = field(default_factory=Tally)
    anchors: dict = field(default_factory=dict)   # (round, pool index) -> count

    def merge(self, other):
        self.tally.merge(other.tally)
        for key, count in other.anchors.items():
            self.anchors[key] = self.anchors.get(key, 0) + count
        return self


# --------------------------------------------------------------------------
# Worker state.
#
# One engine and one season simulator per process, built once in the initializer.
# Both hold scratch buffers sized to the pool, and rebuilding them per task would
# cost more than the tasks do.
# --------------------------------------------------------------------------

_WORKER = {}


def _init_worker(pool, cfg):
    _WORKER["pool"] = pool
    _WORKER["cfg"] = cfg
    _WORKER["engine"] = DraftEngine(pool, cfg.league,
                                    sigma_base=cfg.adp_sigma_base,
                                    sigma_rate=cfg.adp_sigma_rate,
                                    bench_decay=cfg.bench_decay,
                                    room=cfg.room,
                                    perception_sigma=cfg.perception_sigma,
                                    free_agent_depth=cfg.free_agent_depth,
                                    replacement=cfg.replacement or None)
    _WORKER["season"] = SeasonSimulator(pool, cfg.league,
                                        weekly_cv=cfg.weekly_cv,
                                        miss_any_prob=cfg.miss_any_prob,
                                        miss_mean_weeks=cfg.miss_mean_weeks,
                                        projection_sigma=cfg.proj_sigma)


def _wire(engine, cfg):
    """The streaming levels this draft left on the board, or None if disabled.

    Read after the draft and before the next one, because it is derived from the taken
    mask. Cheap: a scan down each position's PPG-sorted list until the sixth undrafted
    name.
    """
    if not cfg.free_agents:
        return None
    return engine.free_agents(cfg.free_agent_depth)


def _rng_for(seed, *tags):
    """A generator keyed by the run seed and the task's identity, not by order.

    Seeding from the task's own coordinates rather than from a shared counter is
    what makes a run reproducible at any ``--jobs``: task (slot 7, candidate 3,
    chunk 2) draws the same numbers whether it ran first or last.
    """
    return np.random.default_rng([int(seed)] + [int(t) for t in tags])


def _stage1_chunk(task):
    """Run one chunk of stage 1. Returns a Stage1Result."""
    slot, n_drafts, seed, chunk = task
    cfg = _WORKER["cfg"]
    engine, season = _WORKER["engine"], _WORKER["season"]
    rounds = cfg.plan_rounds
    rng = _rng_for(seed, 1, slot, chunk)

    out = Stage1Result(slot=slot)
    out.round_n = np.zeros((rounds, len(POSITIONS)), dtype=np.int64)
    out.round_hit = np.zeros((rounds, len(POSITIONS)), dtype=np.int64)
    out.round_metric = np.zeros((rounds, len(POSITIONS)), dtype=np.float64)

    hero = slot - 1
    for _ in range(n_drafts):
        plan = sample_plan(rng, rounds)
        rosters = engine.run(slot, plan, rng)
        result = season.simulate(rosters, hero, rng, _wire(engine, cfg))

        out.plans.setdefault(plan, Tally()).add(result)
        out.overall.add(result)
        shape = tuple(sorted(plan[:SHAPE_ROUNDS]))
        out.shapes.setdefault(shape, Tally()).add(result)
        for r, pos in enumerate(plan):
            out.round_n[r, pos] += 1
            out.round_hit[r, pos] += bool(result.made_playoffs)
            out.round_metric[r, pos] += _metric_of(result, cfg.metric)
    return out


def _baseline_chunk(task):
    """Drafts in which nobody follows a plan, scored from all twelve seats at once.

    The number every other number in the report has to be read against. A plan that
    reaches the playoffs 78% of the time from slot 4 is either a triumph or a
    non-event depending on what an ordinary ADP drafter gets from slot 4, and until
    this is measured there is no way to tell which.

    Cheap, too, and for a structural reason: a room of twelve identical ADP drafters
    is twelve baseline samples per draft, one per slot. The whole baseline costs a
    twelfth of what measuring it slot by slot would.
    """
    n_drafts, seed, chunk = task
    cfg = _WORKER["cfg"]
    engine, season = _WORKER["engine"], _WORKER["season"]
    rng = _rng_for(seed, 0, chunk)

    out = {slot: Tally() for slot in range(1, cfg.league.teams + 1)}
    cut = cfg.league.playoff_teams
    from .season import SeasonResult
    for _ in range(n_drafts):
        rosters = engine.run(0, (), rng)
        wins, points, seeds, champion = season.play(rosters, rng, _wire(engine, cfg))
        made = set(int(t) for t in seeds[:cut])
        for team in range(cfg.league.teams):
            out[team + 1].add(SeasonResult(
                made_playoffs=team in made,
                won_title=champion == team,
                wins=float(wins[team]),
                points=float(points[team])))
    return out


def calibrate_replacement(pool, cfg, verbose=None):
    """Replacement level as a fixed point: what is freely available, given how
    everyone drafts, given they price against what is freely available.

    Returns a (positions,) tuple to install on ``cfg.replacement``.

    Two definitions of replacement compete for this job. The board's -- the last player
    at a position who starts somewhere in the league, which is what
    ``draft_tiers.starter_depths`` computes -- is right for comparing positions on a
    cheat sheet. The other is the waiver wire: the best player available for nothing.
    On this data they diverge by a factor of three at running back, because fifty-eight
    backs get drafted and RB29 is nowhere near free.

    Which one drafts better was settled by experiment rather than argument: one
    wire-priced drafter in a room of eleven board-priced ones, every seat taking its
    turn, beats its own all-board control by 2.0 +/- 0.4 points of playoff rate and
    eleven points a season. So the wire wins, and by enough that using the board's
    levels left the no-plan baseline a measurably weak opponent -- which inflated every
    edge the report printed by about two points.

    It has to be a fixed point because the wire level depends on how everyone drafts,
    and how everyone drafts depends on the wire level. Draft with the board's levels,
    measure what is left, re-draft pricing against that, measure again. It converges
    fast -- the second pass moves the numbers by tenths -- so two passes is plenty.
    """
    levels = np.asarray(pool.replacement, dtype=np.float64)
    if not cfg.replacement_from_wire or not cfg.free_agents:
        return tuple(float(x) for x in levels)
    from .config import REPLACEMENT_CALIBRATION_DRAFTS, REPLACEMENT_CALIBRATION_PASSES
    for step in range(REPLACEMENT_CALIBRATION_PASSES):
        engine = DraftEngine(pool, cfg.league, sigma_base=cfg.adp_sigma_base,
                             sigma_rate=cfg.adp_sigma_rate,
                             bench_decay=cfg.bench_decay, room=cfg.room,
                             perception_sigma=cfg.perception_sigma,
                             free_agent_depth=cfg.free_agent_depth,
                             replacement=levels)
        rng = _rng_for(cfg.budget.seed, 9, step)
        acc = np.zeros(len(POSITIONS))
        for _ in range(REPLACEMENT_CALIBRATION_DRAFTS):
            engine.run(0, (), rng)
            acc += engine.free_agents(cfg.free_agent_depth)
        levels = acc / REPLACEMENT_CALIBRATION_DRAFTS
        if verbose:
            verbose(f"  pass {step + 1}: " + ", ".join(
                f"{POSITIONS[i]} {levels[i]:.1f}" for i in range(len(POSITIONS))))
    return tuple(float(x) for x in levels)


def run_baseline(pool, cfg, jobs=None, progress=None):
    """Measure the ADP baseline. Returns {slot: Tally}."""
    jobs = job_count(cfg) if jobs is None else jobs
    tasks = [(n, cfg.budget.seed, chunk)
             for chunk, n in enumerate(_chunks(cfg.baseline))]
    results = _run_tasks(tasks, _baseline_chunk, pool, cfg, jobs, progress)
    merged = {}
    for result in results:
        for slot, tally in result.items():
            merged.setdefault(slot, Tally()).merge(tally)
    return merged


def _metric_of(result, metric):
    if metric == "playoffs":
        return float(result.made_playoffs)
    if metric == "title":
        return float(result.won_title)
    return result.points


def _measure_chunk(task, stage):
    """Run one chunk of drafts for one named plan. Returns a Stage2Result.

    ``stage`` is folded into the seed, which is what makes stage 3's drafts *fresh*
    rather than a continuation of stage 2's. That matters more than it looks: stage 2
    promotes plans because their observed rates were high, so its counts for exactly
    those plans are biased upward by the selection. Re-drawing under a different tag
    keeps the winner's curse out of the final number.
    """
    slot, plan, n_drafts, seed, chunk = task
    cfg = _WORKER["cfg"]
    engine, season = _WORKER["engine"], _WORKER["season"]
    rng = _rng_for(seed, stage, slot, chunk, *plan)

    out = Stage2Result(slot=slot, plan=plan)
    hero = slot - 1
    for _ in range(n_drafts):
        rosters = engine.run(slot, plan, rng)
        result = season.simulate(rosters, hero, rng, _wire(engine, cfg))
        out.tally.add(result)
        for r, index in enumerate(rosters[hero]):
            key = (r, int(index))
            out.anchors[key] = out.anchors.get(key, 0) + 1
    return out


def _stage2_chunk(task):
    """One chunk of the screening pass."""
    return _measure_chunk(task, 2)


def _stage3_chunk(task):
    """One chunk of the deciding pass. Same work, different seed stream."""
    return _measure_chunk(task, 3)


def _mutation_chunk(task):
    """One chunk of the mutation screen. Its own seed stream, like every stage."""
    return _measure_chunk(task, 4)


# --------------------------------------------------------------------------
# Driving it
# --------------------------------------------------------------------------

# How many chunks a batch of drafts is cut into. Fixed, and deliberately not a
# function of the number of workers -- that was a real bug: chunk boundaries decide
# which seeds each chunk draws, so scaling the split with --jobs meant a run at
# --jobs 1 and the same run at --jobs 4 sampled different drafts and reported
# different numbers. A result that cannot be reproduced on a machine with a
# different core count is not a result.
#
# 64 is comfortably more than any plausible core count, so there is still plenty of
# work to steal when one chunk runs long, and small enough that per-chunk overhead
# stays invisible.
CHUNKS = 64


def _chunks(total):
    """Split ``total`` draws into at most CHUNKS pieces, as evenly as possible."""
    pieces = max(1, min(total, CHUNKS))
    base, extra = divmod(total, pieces)
    return [base + (1 if i < extra else 0) for i in range(pieces)]


def job_count(cfg):
    if cfg.jobs and cfg.jobs > 0:
        return cfg.jobs
    # Leave a core for the OS and for whatever the user is doing while this runs.
    return max(1, (os.cpu_count() or 2) - 2)


def _run_tasks(tasks, fn, pool, cfg, jobs, progress=None):
    """Map ``fn`` over tasks, in this process when jobs == 1 and in a pool otherwise.

    The single-process path is not just a fallback: it is what the determinism test
    compares against, and it is the only way to get a readable traceback out of a
    worker.
    """
    if jobs <= 1:
        _init_worker(pool, cfg)
        results = []
        for i, task in enumerate(tasks):
            results.append(fn(task))
            if progress:
                progress(i + 1, len(tasks))
        return results

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(jobs, initializer=_init_worker, initargs=(pool, cfg)) as mpool:
        for i, result in enumerate(mpool.imap_unordered(fn, tasks, chunksize=1)):
            results.append(result)
            if progress:
                progress(i + 1, len(tasks))
    return results


def run_stage1(pool, cfg, jobs=None, progress=None):
    """Explore. Returns {slot: Stage1Result}."""
    jobs = job_count(cfg) if jobs is None else jobs
    tasks = []
    for slot in cfg.slots:
        for chunk, n in enumerate(_chunks(cfg.budget.stage1)):
            tasks.append((slot, n, cfg.budget.seed, chunk))
    results = _run_tasks(tasks, _stage1_chunk, pool, cfg, jobs, progress)

    merged = {}
    for result in results:
        if result.slot in merged:
            merged[result.slot].merge(result)
        else:
            merged[result.slot] = result
    return merged


def round_scores(stage1, cfg):
    """(plan_rounds, positions) of the metric conditioned on that round's position.

    Positions never sampled in a round -- impossible here, since sampling is uniform
    over legal plans, but possible with a tiny --quick budget -- fall back to the
    slot's overall mean so they are neither favoured nor excluded.
    """
    fallback = stage1.overall.rate(cfg.metric)
    n = np.maximum(stage1.round_n, 1)
    scores = stage1.round_metric / n
    scores[stage1.round_n == 0] = fallback if not math.isnan(fallback) else 0.0
    return scores


def best_plan_for_shape(shape, scores, rounds):
    """The best legal plan whose first ``len(shape)`` rounds are that multiset, or None.

    A "shape" is what people actually argue about -- "two backs and two receivers to
    start" -- and it is the one thing stage 1 measures with real precision: there are
    only a few dozen shapes, so thousands of drafts sit behind each one, where an
    individual plan gets less than a single draft. Turning a good shape back into a
    concrete plan means choosing the order within it and the rounds after it, and the
    per-round marginals are what decides both.
    """
    best, best_score = None, -np.inf
    target = tuple(sorted(shape))
    depth = len(target)
    for plan in legal_plans(rounds):
        if tuple(sorted(plan[:depth])) != target:
            continue
        total = sum(scores[r][pos] for r, pos in enumerate(plan))
        if total > best_score:
            best, best_score = plan, total
    return best


def candidates_for(stage1, cfg):
    """The plans stage 2 should measure for this slot, in the order they earn a row.

    Four sources, and the order is the argument for each:

    1. **Every named archetype**, unconditionally. A table whose winner is a plan
       nobody has a name for is far more convincing beside a measured Zero-RB than on
       its own, and a reader's first question is always "how does the thing I already
       do score".
    2. **Plans built from the per-round marginals** -- the best legal plan under
       stage 1's scores, and its runners-up. This is the part that can find a shape
       nobody thought to test.
    3. **Plans built from the best opening shapes.** Source 2 scores rounds
       independently, which they are not; conditioning on a whole four-round multiset
       captures some of what that misses, and shapes are measured precisely enough to
       trust because there are only dozens of them.
    4. **Plans that did well in stage 1 on their own**, if any cleared the sample
       floor. At a full budget almost none do -- 12,000 drafts over 25,174 plans is
       half a sample each -- so this source is really for small runs. It is last, and
       stage 2 re-measures whatever it nominates, so its noisiness costs nothing but
       simulations.

    Whatever room is left over after all four goes back to source 2, so the candidate
    budget is always spent rather than silently truncated.
    """
    budget = cfg.budget
    picked, seen = [], set()

    def take(plan):
        plan = tuple(plan)
        if plan in seen or len(picked) >= budget.candidates:
            return
        seen.add(plan)
        picked.append(plan)

    for plan in archetype_plans(cfg.plan_rounds).values():
        take(plan)

    scores = round_scores(stage1, cfg)
    room = max(0, budget.candidates - len(picked))
    from_marginals = top_plans_by_round_scores(scores, max(room, 1) * 2, cfg.plan_rounds)
    for plan in from_marginals[:max(room // 2, 1)]:
        take(plan)

    shapes = sorted((s for s, t in stage1.shapes.items() if t.n),
                    key=lambda s: -stage1.shapes[s].rate(cfg.metric))
    for shape in shapes:
        if len(picked) >= budget.candidates:
            break
        plan = best_plan_for_shape(shape, scores, cfg.plan_rounds)
        if plan is not None:
            take(plan)

    empirical = sorted(
        (p for p, t in stage1.plans.items() if t.n >= budget.min_stage1_samples),
        key=lambda p: -stage1.plans[p].rate(cfg.metric))
    for plan in empirical:
        if len(picked) >= budget.candidates:
            break
        take(plan)

    # Top up from the marginal list rather than leaving the budget unspent.
    for plan in from_marginals:
        if len(picked) >= budget.candidates:
            break
        take(plan)

    return picked


def _merge_by_slot(results, cfg):
    """Fold chunk results into {slot: [Stage2Result]}, ranked best first."""
    merged = {}
    for result in results:
        key = (result.slot, result.plan)
        if key in merged:
            merged[key].merge(result)
        else:
            merged[key] = result

    by_slot = {}
    for (slot, _plan), result in merged.items():
        by_slot.setdefault(slot, []).append(result)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda r: -r.tally.rate(cfg.metric))
    return by_slot


def run_stage2(pool, cfg, stage1, jobs=None, progress=None):
    """Screen. Returns {slot: [Stage2Result, ...]} ranked best first by the metric.

    A screening pass, not a verdict: at ``stage2`` drafts the interval is wider than
    the differences between the leading plans, so its job is only to sort two dozen
    candidates into a plausible order so that stage 3 knows which ones to pay for.
    """
    jobs = job_count(cfg) if jobs is None else jobs
    tasks = []
    for slot in cfg.slots:
        for plan in candidates_for(stage1[slot], cfg):
            for chunk, n in enumerate(_chunks(cfg.budget.stage2)):
                tasks.append((slot, plan, n, cfg.budget.seed, chunk))
    results = _run_tasks(tasks, _stage2_chunk, pool, cfg, jobs, progress)
    return _merge_by_slot(results, cfg)


def mutation_candidates(screen, cfg):
    """{slot: [plan]} -- single-round neighbours of each slot's screen leaders.

    A local search step, and the answer to a real failure. The nomination score is
    additive over rounds, so it cannot see that swapping two rounds together is worth
    more than either swap alone; at slot 1 that blindness left a plan worth 2.2 extra
    points of playoff rate unmeasured. Walking one swap out from plans that already
    screened well finds them.

    Already-screened plans are excluded, so this only ever adds work that is new.
    """
    out = {}
    for slot, results in screen.items():
        seen = {r.plan for r in results}
        fresh = []
        for leader in results[:cfg.budget.mutate_top]:
            for plan in mutations(leader.plan, cfg.plan_rounds):
                if plan not in seen:
                    seen.add(plan)
                    fresh.append(plan)
        out[slot] = fresh
    return out


def _mutation_pass(pool, cfg, screen, step, jobs, progress):
    """One walk of the neighbourhood. Returns the screen with the new plans merged in."""
    wanted = mutation_candidates(screen, cfg)
    tasks = [(slot, plan, n, cfg.budget.seed + step, chunk)
             for slot, plans in wanted.items()
             for plan in plans
             for chunk, n in enumerate(_chunks(cfg.budget.stage2b))]
    if not tasks:
        return screen, 0
    results = _run_tasks(tasks, _mutation_chunk, pool, cfg, jobs, progress)
    extra = _merge_by_slot(results, cfg)

    merged = {}
    for slot in set(screen) | set(extra):
        rows = list(screen.get(slot, [])) + list(extra.get(slot, []))
        rows.sort(key=lambda r: -r.tally.rate(cfg.metric))
        merged[slot] = rows
    return merged, len(tasks)


def run_mutations(pool, cfg, screen, jobs=None, progress=None):
    """Hill-climb the plan space from the screen leaders. Returns the merged screen.

    Repeated ``mutate_passes`` times, each pass starting from whatever the previous one
    promoted to the top. One pass only reaches plans a single swap from a screen
    leader, and the plan that exposed the nomination bug was **two** swaps away -- it
    beat the leader by 2.2 points of playoff rate and was never measured. One pass is
    therefore demonstrably not enough.

    Cost falls with each pass rather than compounding: once a leader stops moving, its
    whole neighbourhood is already screened and the pass finds nothing new to run.
    Measured at ``stage2b`` drafts, below stage 2's, because these plans only need
    ranking against the leaders they came from -- stage 3 re-measures whatever survives.
    """
    jobs = job_count(cfg) if jobs is None else jobs
    for step in range(cfg.budget.mutate_passes):
        screen, added = _mutation_pass(pool, cfg, screen, step, jobs, progress)
        if not added:
            break
    return screen


def run_stage3(pool, cfg, stage2, jobs=None, progress=None):
    """Decide. Re-measures the top ``finalists`` plans per slot at full precision.

    Returns {slot: [Stage2Result]} -- the same shape as stage 2, ranked best first,
    but with ``stage3`` drafts behind each row instead of ``stage2``.

    This stage exists because the report was being over-read. The gap between the
    best few plans at a slot is one to three points of playoff rate, and a
    1,200-draft interval is nearly three points wide either side, so a table naming
    one best plan per slot was presenting noise as a decision. Concentrating the
    budget on the eight plans that could plausibly win narrows the interval to about
    a point, which is the resolution the question needs.

    Note it re-measures rather than accumulating on top of stage 2. Adding the two
    together would be more sample-efficient and quietly wrong: stage 2 selected these
    plans *because* their observed rates were high, so their stage-2 counts are biased
    upward by the selection, and the winner's-curse inflation would carry straight
    into the final number. Fresh drafts, keyed to a different stage tag, cost 20% more
    simulations and buy an unbiased estimate.
    """
    jobs = job_count(cfg) if jobs is None else jobs
    tasks = []
    for slot in cfg.slots:
        for plan in [r.plan for r in stage2.get(slot, [])[:cfg.budget.finalists]]:
            for chunk, n in enumerate(_chunks(cfg.budget.stage3)):
                tasks.append((slot, plan, n, cfg.budget.seed, chunk))
    if not tasks:
        return {}
    results = _run_tasks(tasks, _stage3_chunk, pool, cfg, jobs, progress)
    return _merge_by_slot(results, cfg)


def recommend(results, cfg=None):
    """One plan out of a tied leading group, with the reason it was chosen.

    Returns ``(result, group, reason)``. When the group has one member there is nothing
    to decide. When it has several, they are by construction indistinguishable on the
    metric the search ranks by, so choosing among them on that metric is choosing on
    noise.

    The tie-break is **mean wins**, measured on the very same drafts. It is continuous
    where the playoff flag is binary, so it carries far more signal per draft, and
    unlike mean points it is the same objective: a playoff berth is won on wins, so
    ranking a tie by wins is measuring the thing itself more precisely rather than
    substituting a proxy for it.

    Points was the first attempt and it is subtly the wrong quantity. Points feed wins
    and wins feed berths, but the chain is not tight: two plans with identical mean
    points can differ in week-to-week variance and so in how often those points land as
    wins. Wins already contains that.

    It used to be the sum of stage-1 per-round marginals, and that was wrong -- wrong
    enough to produce the bug that prompted this rewrite. Those marginals are
    confounded: P(playoffs | quarterback in round 8) is high not because a round-8
    quarterback is good but because a plan that takes its quarterback in round 8 spent
    rounds 1-7 on skill players, while a plan with a receiver there mostly spent an
    earlier pick on a quarterback. The marginal measures *placement*, not the pick. Add
    eight of them up and the score is maximised by stuffing quarterbacks into every
    late round up to the cap -- which is exactly the "bench quarterback ahead of a
    second starting receiver" recommendation that could not possibly be right.
    """
    group = leading_group(results)
    if len(group) <= 1:
        return (group[0] if group else None), group, "clear best"
    best = max(group, key=lambda r: r.tally.mean_wins)
    return best, group, f"{len(group)} tied; most wins"


def run_plan(pool, cfg, plan, jobs=None, progress=None):
    """Measure one named plan from every slot. Returns {slot: [Stage2Result]}.

    Its own entry point rather than a one-candidate search, because stage 1 would be
    pure waste when the candidate list is already known. Shares stage 2's worker
    function and its seeding, so `--plan X` reproduces exactly the row a full run
    would have printed for X at the same seed and budget.
    """
    jobs = job_count(cfg) if jobs is None else jobs
    plan = tuple(plan)
    tasks = [(slot, plan, n, cfg.budget.seed, chunk)
             for slot in cfg.slots
             for chunk, n in enumerate(_chunks(cfg.budget.stage2))]
    results = _run_tasks(tasks, _stage2_chunk, pool, cfg, jobs, progress)

    merged = {}
    for result in results:
        key = (result.slot, result.plan)
        merged[key] = merged[key].merge(result) if key in merged else result
    by_slot = {}
    for (slot, _plan), result in merged.items():
        by_slot.setdefault(slot, []).append(result)
    return by_slot


def anchor_names(result, pool, rounds=None, top=2):
    """Per round, the players this plan most often actually ended up with.

    A plan says "receiver in round 3"; this says it was Tee Higgins 41% of the time.
    The abstraction is what generalises and the names are what a drafter can use, so
    the report carries both.
    """
    rounds = rounds or max(r for r, _ in result.anchors) + 1
    out = []
    for r in range(rounds):
        counts = Counter({idx: c for (rr, idx), c in result.anchors.items() if rr == r})
        total = sum(counts.values()) or 1
        out.append([(str(pool.name[idx]), c / total) for idx, c in counts.most_common(top)])
    return out


def summarise(cfg, final, stage1=None):
    """{slot: (recommended result, tied group, reason)} -- the headline table's rows.

    The recommendation, not "the winner". Where several plans could not be told apart
    the group is carried alongside it, and the report prints both, because a reader
    who takes one row as the answer when six plans tie has been misled by the table.
    """
    out = {}
    for slot, results in final.items():
        if not results:
            continue
        out[slot] = recommend(results, cfg)
    return out
