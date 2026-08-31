#!/usr/bin/env python3
"""Find the best draft plan for every slot by simulating thousands of drafts.

Usage:
    python simulate_drafts.py                    # full run, all 12 slots
    python simulate_drafts.py --quick            # smoke test, ~30 seconds
    python simulate_drafts.py --slots 1,6,12     # only these slots
    python simulate_drafts.py --adp-model espn   # untouched ESPN ADP
    python simulate_drafts.py --room adp         # opponents draft off ADP instead
    python simulate_drafts.py --injuries         # switch injuries back on
    python simulate_drafts.py --plan RB-WR-WR-TE-RB-WR-QB-RB   # grade one plan

The draft board (draft_tiers.py) prices players and says whether one survives to
your next pick. It cannot say what shape of roster to aim for from the slot you
drew, because that depends on what eleven other drafters do -- and the only honest
way to find out is to draft against them a few hundred thousand times.

Reads the same rankings export and projections the board reads, and prices players
through the board's own code, so the two agree about what a player is worth. Writes
tables to the console and CSVs plus a markdown report to --out.

See docs/superpowers/specs/2026-08-19-draft-simulator-design.md for the model and
its limits, which are real and worth reading before betting a first-round pick on
a 1.4-point difference in playoff rate.
"""

import argparse
import os
import sys
import time

from draftsim import report, search
from draftsim.config import (DEFAULT_METRIC, DEFAULT_ROOM, FULL_BUDGET, METRICS,
                             POS_INDEX, POSITIONS, QUICK_BUDGET, ROOMS, Budget,
                             League, SimConfig)
from draftsim.plans import label_plan, parse_plan, plan_string
from draftsim.pool import DEFAULT_PROJECTIONS, DEFAULT_RANKINGS, load_pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "sim_out")


def parse_starters(text):
    """'QB1,RB2,WR2,TE1,K1,DST1' -> the starters tuple in POSITIONS order."""
    counts = [0] * len(POSITIONS)
    for token in text.split(","):
        token = token.strip().upper()
        if not token:
            continue
        digits = "".join(c for c in token if c.isdigit())
        name = "".join(c for c in token if not c.isdigit())
        if name == "DEF":
            name = "DST"
        if name not in POS_INDEX:
            raise argparse.ArgumentTypeError(f"unknown position {name!r} in --starters")
        counts[POS_INDEX[name]] = int(digits or 1)
    return tuple(counts)


def build_parser():
    p = argparse.ArgumentParser(
        description="Simulate drafts and report the best plan from each slot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group("data")
    src.add_argument("--rankings", default=DEFAULT_RANKINGS,
                     help="Draft-rankings export CSV")
    src.add_argument("--projections", default=DEFAULT_PROJECTIONS,
                     help="component projections CSV, read only for TE receptions")
    src.add_argument("--pool-skill", type=int, default=FULL_BUDGET.pool_skill,
                     help="cap the skill-position pool at this overall rank")
    src.add_argument("--adp-model", choices=("premium", "espn"), default="premium",
                     help="'premium' shifts ADP for the 1.5 PPR tight end market; "
                          "'espn' uses the export's raw ADP")
    src.add_argument("--no-te-premium", action="store_true",
                     help="score tight ends at standard PPR")

    lg = p.add_argument_group("league")
    lg.add_argument("--teams", type=int, default=12)
    lg.add_argument("--rounds", type=int, default=16)
    lg.add_argument("--starters", type=parse_starters, default="QB1,RB2,WR2,TE1,K1,DST1",
                    help="mandatory starting slots, excluding the flex")
    lg.add_argument("--flex", type=int, default=2)
    lg.add_argument("--regular-weeks", type=int, default=14)
    lg.add_argument("--total-weeks", type=int, default=17)
    lg.add_argument("--playoff-teams", type=int, default=6)

    run = p.add_argument_group("search")
    run.add_argument("--slots", default="",
                     help="comma-separated draft slots; default every slot")
    run.add_argument("--stage1", type=int, default=None,
                     help=f"exploration drafts per slot (default {FULL_BUDGET.stage1})")
    run.add_argument("--stage2", type=int, default=None,
                     help=f"confirmation drafts per candidate (default {FULL_BUDGET.stage2})")
    run.add_argument("--candidates", type=int, default=None,
                     help=f"candidate plans per slot (default {FULL_BUDGET.candidates})")
    run.add_argument("--metric", choices=METRICS, default=DEFAULT_METRIC,
                     help="what 'best' means")
    run.add_argument("--baseline", type=int, default=None,
                     help="drafts spent measuring what an ADP drafter gets from each "
                          "slot; every plan's edge is read against it. 0 skips it")
    run.add_argument("--stage3", type=int, default=None,
                     help=f"drafts per finalist, on fresh draws "
                          f"(default {FULL_BUDGET.stage3})")
    run.add_argument("--stage2b", type=int, default=None,
                     help=f"drafts per mutation-screen plan "
                          f"(default {FULL_BUDGET.stage2b})")
    run.add_argument("--mutate-passes", type=int, default=None,
                     help=f"times the neighbourhood is walked; 1 only reaches plans "
                          f"one swap out (default {FULL_BUDGET.mutate_passes})")
    run.add_argument("--board-replacement", action="store_true",
                     help="price picks against the last starter in the league, as the "
                          "draft board does, instead of against the waiver wire. "
                          "Measurably worse drafting -- kept for comparison")
    run.add_argument("--no-projection-error", action="store_true",
                     help="score the season off the projections themselves, so nobody "
                          "is ever wrong about a player. The old behaviour")
    run.add_argument("--no-injuries", action="store_true",
                     help="nobody misses a game; bye weeks still apply")
    run.add_argument("--mutate-top", type=int, default=None,
                     help=f"screen leaders whose single-round neighbourhood is "
                          f"explored; 0 skips it (default {FULL_BUDGET.mutate_top})")
    run.add_argument("--free-agent-depth", type=int, default=None,
                     help="the streamer who fills an unfillable slot is this deep in "
                          "the undrafted pool at his position")
    run.add_argument("--no-free-agents", action="store_true",
                     help="forfeit an unfillable starting slot instead of streaming "
                          "it. The old behaviour, and the reason a bench QB used to "
                          "look worth a round-8 pick")
    run.add_argument("--finalists", type=int, default=None,
                     help=f"candidates promoted to stage 3 (default "
                          f"{FULL_BUDGET.finalists})")
    run.add_argument("--room", choices=ROOMS, default=DEFAULT_ROOM,
                     help="'value': all seats draft best value available, differing "
                          "only in their private view of each player. 'adp': the "
                          "opponents draft off ADP + noise instead, which is the "
                          "comparison that shows how much of an edge was projection "
                          "arbitrage rather than strategy")
    run.add_argument("--perception-sigma", type=float, default=None,
                     help="spread of a drafter's private view of a player, as a "
                          "lognormal on projected PPG. 0 makes the room deterministic")
    run.add_argument("--bench-decay", type=float, default=None,
                     help="value multiplier per bench slot deep a pick lands")
    run.add_argument("--plan", default=None,
                     help="skip the search and grade this one plan from every slot, "
                          "e.g. RB-WR-WR-TE-RB-WR-QB-RB")
    run.add_argument("--seed", type=int, default=FULL_BUDGET.seed)
    run.add_argument("--jobs", type=int, default=0,
                     help="worker processes; 0 picks cpu_count - 2")
    run.add_argument("--quick", action="store_true",
                     help="tiny budget: proves the pipeline runs, believe nothing in it")

    out = p.add_argument_group("output")
    out.add_argument("--out", default=DEFAULT_OUT, help="directory for CSVs and report.md")
    out.add_argument("--top", type=int, default=10,
                     help="plans shown per slot on the console")
    out.add_argument("--no-files", action="store_true", help="console output only")
    out.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return p


def build_config(args):
    league = League(teams=args.teams, rounds=args.rounds, starters=args.starters,
                    flex=args.flex, regular_weeks=args.regular_weeks,
                    total_weeks=args.total_weeks, playoff_teams=args.playoff_teams)

    base = QUICK_BUDGET if args.quick else FULL_BUDGET
    budget = Budget(
        stage1=args.stage1 if args.stage1 is not None else base.stage1,
        stage2=args.stage2 if args.stage2 is not None else base.stage2,
        stage2b=args.stage2b if args.stage2b is not None else base.stage2b,
        stage3=args.stage3 if args.stage3 is not None else base.stage3,
        mutate_top=(args.mutate_top if args.mutate_top is not None
                    else base.mutate_top),
        mutate_passes=(args.mutate_passes if args.mutate_passes is not None
                       else base.mutate_passes),
        candidates=args.candidates if args.candidates is not None else base.candidates,
        finalists=args.finalists if args.finalists is not None else base.finalists,
        min_stage1_samples=base.min_stage1_samples,
        pool_skill=args.pool_skill,
        seed=args.seed)

    slots = tuple(int(s) for s in args.slots.replace(" ", "").split(",") if s) \
        if args.slots else ()

    kwargs = {}
    if args.perception_sigma is not None:
        kwargs["perception_sigma"] = args.perception_sigma
    if args.free_agent_depth is not None:
        kwargs["free_agent_depth"] = args.free_agent_depth
    if args.no_free_agents:
        kwargs["free_agents"] = False
    if args.board_replacement:
        kwargs["replacement_from_wire"] = False
    if args.no_projection_error:
        kwargs["projection_error"] = False
    if args.bench_decay is not None:
        kwargs["bench_decay"] = args.bench_decay
    if args.baseline is not None:
        kwargs["baseline"] = args.baseline
    elif args.quick:
        # The baseline measures all twelve slots per draft, so it is cheap even here,
        # but --quick promises seconds and this is the one budget that does not shrink
        # with --slots.
        kwargs["baseline"] = 400

    return SimConfig(league=league, budget=budget, metric=args.metric, slots=slots,
                     adp_model=args.adp_model, te_premium=not args.no_te_premium,
                     jobs=args.jobs, room=args.room,
                     injuries=not args.no_injuries, **kwargs)


def _progress(label, quiet):
    """A one-line, rewriting progress indicator. Silent when --quiet or not a tty."""
    if quiet or not sys.stderr.isatty():
        return None
    start = time.perf_counter()

    def tick(done, total):
        elapsed = time.perf_counter() - start
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        sys.stderr.write(f"\r{label}: {done}/{total} chunks, "
                         f"{elapsed:5.1f}s elapsed, {eta:5.1f}s left   ")
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")
    return tick


def grade_one_plan(pool, cfg, plan, args, jobs):
    """--plan: skip the search, measure one plan from every slot.

    Returns an empty stage-1 result alongside, so the reporting path downstream is
    the same one a full run takes; the marginal tables simply have nothing to show.
    """
    if len(plan) != cfg.plan_rounds:
        print(f"note: plan has {len(plan)} rounds, the model uses {cfg.plan_rounds}; "
              f"rounds past it are filled by value.", file=sys.stderr)
    stage2 = search.run_plan(pool, cfg, plan, jobs, _progress("grading", args.quiet))
    return {}, stage2


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = build_config(args)

    started = time.perf_counter()
    pool = load_pool(args.rankings, args.projections, league=cfg.league,
                     adp_model=cfg.adp_model, te_premium=cfg.te_premium,
                     pool_skill=cfg.budget.pool_skill)

    # Replacement level is a fixed point over the room's own behaviour, so it has to
    # be solved before the header can honestly print what the drafters are pricing
    # against.
    if cfg.replacement_from_wire and cfg.free_agents:
        note = (None if args.quiet else
                lambda line: print(line, file=sys.stderr))
        if note:
            print("calibrating replacement level against the waiver wire",
                  file=sys.stderr)
        cfg.replacement = search.calibrate_replacement(pool, cfg, verbose=note)

    report.print_header(pool, cfg)
    jobs = search.job_count(cfg)
    if not args.quiet:
        print(f"running on {jobs} process{'es' if jobs > 1 else ''}\n", file=sys.stderr)

    # The baseline first, and unconditionally: it is the yardstick every plan's
    # number is read against, and it is the run's own calibration check. Twelve
    # identical drafters must average a 50% playoff rate, so if this table is wrong
    # there is no point looking at the rest.
    baseline = {}
    if cfg.baseline:
        baseline = search.run_baseline(pool, cfg, jobs,
                                       _progress("baseline", args.quiet))
        report.print_baseline(cfg, baseline)

    if args.plan:
        plan = parse_plan(args.plan)
        stage1, stage2 = grade_one_plan(pool, cfg, plan, args, jobs)
        final = stage2
        stage2 = {}
        print(f"PLAN {plan_string(plan)}   ({label_plan(plan)})\n")
        report.print_summary(cfg, final, baseline)
    else:
        stage1 = search.run_stage1(pool, cfg, jobs, _progress("stage 1", args.quiet))
        stage2 = search.run_stage2(pool, cfg, stage1, jobs,
                                   _progress("stage 2", args.quiet))
        # Walk one swap out from the screen leaders. The nomination score is additive
        # over rounds and so cannot see a pair of swaps that only pays off together.
        if cfg.budget.mutate_top:
            stage2 = search.run_mutations(pool, cfg, stage2, jobs,
                                          _progress("mutations", args.quiet))
        # Stage 3 re-measures the shortlist on fresh draws. It is what the tables
        # report; stage 2 only decided who got in.
        final = search.run_stage3(pool, cfg, stage2, jobs,
                                  _progress("stage 3", args.quiet)) or stage2
        report.print_summary(cfg, final, baseline, stage1)
        report.print_marginals(cfg, stage1)
        report.print_slot_detail(cfg, pool, final, baseline, stage1, top=args.top)

    elapsed = time.perf_counter() - started
    if not args.no_files:
        paths = report.write_csvs(args.out, cfg, pool, stage1, stage2, final, baseline)
        paths.append(report.write_markdown(args.out, cfg, pool, stage1, final,
                                           baseline, elapsed))
        print("wrote:")
        for path in paths:
            print(f"  {path}")
    print(f"\ndone in {elapsed / 60:.1f} min")
    return 0


if __name__ == "__main__":
    # Required on Windows: the worker pool uses the spawn start method, which
    # re-imports this module in every child. Without the guard each child would
    # start its own search.
    sys.exit(main())
