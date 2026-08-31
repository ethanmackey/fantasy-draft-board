"""Turning tallies into something a person can act on at a draft table.

Three audiences, three outputs. The console is for the person who just ran it and
wants the answer. The CSVs are for the person who wants to disagree with it and
needs the raw marginals to do so. The markdown is for the person reading it a week
later, away from the terminal, and it is the only one that carries the caveats.

Three things every table here insists on.

**Intervals.** A plan's number only means something next to how precisely it was
measured, and a table of bare percentages invites exactly the over-reading the
staged search was built to prevent.

**Tied groups, not winners.** At every slot the leading few plans sit within a
couple of points of each other. Where a confidence interval on the *difference*
between two plans contains zero, this report says so and prints the whole group. One
of them is still named as a recommendation -- a reader needs somewhere to start --
but it is labelled as a pick from a tie, not as a measured best.

**The edge column.** Absolute playoff rates are lifted by everything the room does
badly, so they are not comparable across configurations. The edge over the slot's
own no-plan baseline is, and it is the number that answers the question the study
was built for.
"""

import csv
import os
from datetime import date

from .config import EXPECTED_GAMES_MISSED, POSITIONS, SKILL_POSITIONS
from .plans import label_plan, plan_string
from .search import (SHAPE_ROUNDS, anchor_names, leading_group, recommend,
                     round_scores, wilson)

BAR = "-" * 78


def _edge(tally, base):
    """A plan's playoff rate minus its slot's baseline, or None if unmeasured."""
    if base is None or not base.n:
        return None
    return tally.playoff_rate - base.playoff_rate


def _fmt_pct(value):
    return "  n/a" if value != value else f"{100 * value:5.1f}"


def _fmt_edge(value):
    return "    -" if value is None else f"{100 * value:+5.1f}"


def _ci(hits, n):
    low, high = wilson(hits, n)
    return f"{_fmt_pct(low)}-{_fmt_pct(high).strip()}"


def _md_pct(value):
    return "n/a" if value != value else f"{100 * value:.1f}"


def _md_ci(hits, n):
    low, high = wilson(hits, n)
    return f"{_md_pct(low)}–{_md_pct(high)}"


def baseline_label(cfg):
    """What the baseline drafter actually is, in words, for every table that cites it.

    It changes with the room, and getting it wrong would misdescribe every edge in the
    report: in the value room the baseline is straight best-available-value, so an
    edge is what the *plan* is worth; in the adp room it is an ADP drafter, and an
    edge conflates the plan with the much larger effect of using projections at all.
    """
    return "BPA" if cfg.room == "value" else "ADP"


def _table(headers, rows, aligns=None):
    """Fixed-width text table. Small enough to hand-roll; no dependency earned."""
    cols = len(headers)
    widths = [len(h) for h in headers]
    body = [[str(c) for c in row] for row in rows]
    for row in body:
        for i in range(cols):
            widths[i] = max(widths[i], len(row[i]))
    aligns = aligns or ["<"] + [">"] * (cols - 1)
    out = ["  ".join(f"{h:{a}{w}}" for h, a, w in zip(headers, aligns, widths)),
           "  ".join("-" * w for w in widths)]
    for row in body:
        out.append("  ".join(f"{c:{a}{w}}" for c, a, w in zip(row, aligns, widths)))
    return "\n".join(out)


PLAN_HEADERS = ["", "Plan", "Archetype", "Plyf%", "95% CI", "edge", "Titl%",
                "Wins", "Points", "n"]
PLAN_ALIGNS = ["<", "<", "<", ">", ">", ">", ">", ">", ">", ">"]


def _plan_row(result, base=None, mark=""):
    t = result.tally
    return [mark, plan_string(result.plan), label_plan(result.plan),
            _fmt_pct(t.playoff_rate), _ci(t.playoffs, t.n),
            _fmt_edge(_edge(t, base)), _fmt_pct(t.title_rate),
            f"{t.mean_wins:.2f}", f"{t.mean_points:.0f}", t.n]


# --------------------------------------------------------------------------
# Console
# --------------------------------------------------------------------------

def print_header(pool, cfg, out=print):
    league = cfg.league
    lineup = ", ".join(f"{POSITIONS[i]}{n}" for i, n in enumerate(league.starters) if n)
    out(BAR)
    out(f"Draft strategy simulation -- {date.today().isoformat()}")
    out(BAR)
    out(f"league      {league.teams} teams, {league.rounds} rounds, "
        f"{lineup} + {league.flex} FLEX, {league.bench} bench")
    out(f"season      {league.regular_weeks} regular weeks, "
        f"{league.playoff_teams}-team playoff through week {league.total_weeks}")
    out(f"scoring     PPR{', tight ends 1.5' if pool.te_premium else ' (no TE premium)'}")
    out(f"pool        {pool.size} players, ADP model '{pool.adp_model}'")
    if cfg.room == "value":
        out(f"room        all {league.teams} seats draft best value available, each "
            f"with its own")
        out(f"            view of every player (lognormal sigma "
            f"{cfg.perception_sigma:.2f}); ADP does not")
        out(f"            influence who is picked when")
    else:
        out(f"room        {league.teams - 1} opponents draft off ADP + noise "
            f"(sigma {cfg.adp_sigma_base:.1f} + {cfg.adp_sigma_rate:.2f}/pick)")
    out(f"injuries    " + ("on, " + ", ".join(
            f"{POSITIONS[i]} {g:.1f}" for i, g in enumerate(EXPECTED_GAMES_MISSED))
            + " expected games missed" if cfg.injuries else "OFF -- byes only"))
    out(f"proj error  " + ("on, sigma " + ", ".join(
            f"{POSITIONS[i]} {x:.2f}" for i, x in enumerate(cfg.projection_sigma))
            + " rising with rank" if cfg.projection_error
            else "OFF -- projections scored as truth"))
    if cfg.replacement:
        out(f"replacement " + ", ".join(f"{POSITIONS[i]} {v:.1f}"
                                        for i, v in enumerate(cfg.replacement))
            + ("  (waiver wire, fixed point)" if cfg.replacement_from_wire
               else "  (board)"))
    out(f"waiver wire {'depth ' + str(cfg.free_agent_depth) + ' of the undrafted pool'
        if cfg.free_agents else 'OFF -- an unfillable slot forfeits'}"
        + ("  (last resort, after the bench)" if cfg.free_agents else ""))
    out(f"weekly CV   " + ", ".join(f"{POSITIONS[i]} {cv:.2f}"
                                    for i, cv in enumerate(cfg.weekly_cv)))
    out(f"budget      baseline {cfg.baseline:,}  |  stage 1 {cfg.budget.stage1:,}/slot"
        f"  |  stage 2 {cfg.budget.stage2:,} x {cfg.budget.candidates}"
        f"  |  mutations {cfg.budget.stage2b:,} x top {cfg.budget.mutate_top}"
        f"  |  stage 3 {cfg.budget.stage3:,} x {cfg.budget.finalists}")
    out(f"metric      {cfg.metric}    seed {cfg.budget.seed}")
    out("")


def print_baseline(cfg, baseline, out=print):
    """What a drafter with no plan gets from each slot -- and the calibration check.

    Printed before any plan is considered, because it is the yardstick for all of
    them. It also self-tests the simulator: twelve identical drafters in a
    twelve-team league must average a 50% playoff rate, so a mean that drifts is a
    bug in the schedule, the standings or the bracket rather than a finding about
    draft position. What the table legitimately shows is the size of the snake's own
    slot effect.
    """
    if not baseline:
        return
    rows = []
    for slot in sorted(baseline):
        t = baseline[slot]
        rows.append([slot, _fmt_pct(t.playoff_rate), _ci(t.playoffs, t.n),
                     _fmt_pct(t.title_rate), f"{t.mean_wins:.2f}",
                     f"{t.mean_points:.0f}", t.n])
    kind = baseline_label(cfg)
    what = ("best-available-value drafter, no plan" if kind == "BPA"
            else "ordinary ADP drafter")
    out(f"{kind} BASELINE: a {what}, by slot")
    out(_table(["Slot", "Plyf%", "95% CI", "Titl%", "Wins", "Points", "n"], rows,
               aligns=[">"] * 7))
    total_n = sum(t.n for t in baseline.values())
    mean = sum(t.playoffs for t in baseline.values()) / max(1, total_n)
    out(f"  mean {100 * mean:.1f}% -- must sit near 50.0 by symmetry; a drift is a "
        f"bug, not a finding.")
    out("")


def print_summary(cfg, final, baseline=None, stage1=None, out=print):
    """The one table most readers will look at: the recommended plan per slot.

    Carries a 'tied' column, which is the whole point. Where it reads 6, five other
    plans measured the same as this one and the row is a pick from a tie rather than
    a finding -- and the per-slot sections below name them.
    """
    picks, rows = {}, []
    for slot in sorted(final):
        if not final[slot]:
            continue
        best, group, reason = recommend(final[slot], cfg)
        picks[slot] = (best, group, reason)
        t = best.tally
        base = (baseline or {}).get(slot)
        rows.append([slot, plan_string(best.plan), label_plan(best.plan),
                     _fmt_pct(t.playoff_rate), _ci(t.playoffs, t.n),
                     _fmt_pct(base.playoff_rate) if base else "  n/a",
                     _fmt_edge(_edge(t, base)),
                     len(group), _fmt_pct(t.title_rate), f"{t.mean_points:.0f}"])
    kind = baseline_label(cfg)
    out("RECOMMENDED PLAN BY DRAFT SLOT")
    out(_table(["Slot", "Plan", "Archetype", "Plyf%", "95% CI", f"{kind}base", "edge",
                "tied", "Titl%", "Points"], rows,
               aligns=[">", "<", "<", ">", ">", ">", ">", ">", ">", ">"]))
    out("")
    out(f"  {kind}base is what a no-plan drafter gets from that slot; edge is the "
        f"plan's gain")
    out( "  over it. 'tied' counts the plans that could NOT be distinguished from "
         "this one")
    out( "  (a 95% interval on the difference of the two rates contains zero). Where "
         "tied")
    out( "  is above 1 the row is a pick from a tie, broken on mean wins.")
    out("")
    return picks


def print_marginals(cfg, stage1, out=print):
    """Per-round position marginals -- the most reliable table in the report.

    Printed for the first three rounds only. Later rounds are in the CSV; by round 5
    the marginals converge, because a plan's later rounds are mostly compensating for
    its earlier ones, and printing eight nearly-identical tables buries the three
    that say something.
    """
    for r in range(min(3, cfg.plan_rounds)):
        rows = []
        for slot in sorted(stage1):
            scores = round_scores(stage1[slot], cfg)
            row = [slot] + [_fmt_pct(scores[r, pos]) for pos in SKILL_POSITIONS]
            row.append(POSITIONS[max(SKILL_POSITIONS, key=lambda p: scores[r, p])])
            rows.append(row)
        out(f"ROUND {r + 1}: {cfg.metric} rate by the position taken "
            f"(stage 1, {cfg.budget.stage1:,} drafts/slot)")
        out(_table(["Slot"] + [POSITIONS[p] for p in SKILL_POSITIONS] + ["best"], rows,
                   aligns=[">"] * (len(SKILL_POSITIONS) + 1) + ["<"]))
        out("")


def print_slot_detail(cfg, pool, final, baseline=None, stage1=None, top=10, out=print):
    """Per slot: the finalists, with the tied group flagged, then the anchor players."""
    for slot in sorted(final):
        results = final[slot]
        if not results:
            continue
        base = (baseline or {}).get(slot)
        group = {id(r) for r in leading_group(results)}
        best, _, reason = recommend(results, cfg)
        out(BAR)
        out(f"SLOT {slot}   (first picks "
            f"{', '.join(str(p) for p in _slot_picks(cfg, slot))})")
        out(BAR)
        rows = []
        for r in results[:top]:
            mark = "*" if r is best else ("=" if id(r) in group else "")
            rows.append(_plan_row(r, base, mark))
        out(_table(PLAN_HEADERS, rows, PLAN_ALIGNS))
        out(f"  * recommended ({reason})   = indistinguishable from the leader")
        out("")
        out(f"  who actually filled the rounds under {plan_string(best.plan)}:")
        for r, names in enumerate(anchor_names(best, pool, cfg.league.rounds, top=2)):
            if names:
                out(f"    R{r + 1:02d}  " + ", ".join(f"{n} {100 * f:.0f}%"
                                                      for n, f in names))
        out("")


def _slot_picks(cfg, slot, n=4):
    from .draft import slot_picks
    return slot_picks(slot, cfg.league.teams, cfg.league.rounds)[:n]


# --------------------------------------------------------------------------
# CSVs -- for the reader who wants to disagree with the tables
# --------------------------------------------------------------------------

def _write_plan_csv(path, cfg, by_slot, baseline, stage_name):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "rank", "stage", "plan", "archetype", "drafts",
                    "playoff_rate", "playoff_ci_low", "playoff_ci_high",
                    "edge_vs_baseline", "tied_with_leader", "title_rate",
                    "title_ci_low", "title_ci_high", "mean_wins", "mean_points"])
        for slot in sorted(by_slot):
            base = (baseline or {}).get(slot)
            group = {id(r) for r in leading_group(by_slot[slot])}
            for i, result in enumerate(by_slot[slot]):
                t = result.tally
                low, high = wilson(t.playoffs, t.n)
                tlow, thigh = wilson(t.titles, t.n)
                edge = _edge(t, base)
                w.writerow([slot, i + 1, stage_name, plan_string(result.plan),
                            label_plan(result.plan), t.n,
                            f"{t.playoff_rate:.4f}", f"{low:.4f}", f"{high:.4f}",
                            "" if edge is None else f"{edge:.4f}",
                            int(id(result) in group),
                            f"{t.title_rate:.4f}", f"{tlow:.4f}", f"{thigh:.4f}",
                            f"{t.mean_wins:.3f}", f"{t.mean_points:.1f}"])


def write_csvs(outdir, cfg, pool, stage1, stage2, final=None, baseline=None):
    os.makedirs(outdir, exist_ok=True)
    final = final if final is not None else stage2
    paths = []

    if baseline:
        path = os.path.join(outdir, "baseline.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["slot", "baseline_kind", "drafts", "playoff_rate",
                        "playoff_ci_low", "playoff_ci_high", "title_rate",
                        "mean_wins", "mean_points"])
            for slot in sorted(baseline):
                t = baseline[slot]
                low, high = wilson(t.playoffs, t.n)
                w.writerow([slot, baseline_label(cfg), t.n, f"{t.playoff_rate:.4f}",
                            f"{low:.4f}", f"{high:.4f}", f"{t.title_rate:.4f}",
                            f"{t.mean_wins:.3f}", f"{t.mean_points:.1f}"])
        paths.append(path)

    if stage1:
        path = os.path.join(outdir, "stage1_marginals.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["slot", "round", "position", "drafts", "playoff_rate",
                        "metric_mean"])
            for slot in sorted(stage1):
                result = stage1[slot]
                scores = round_scores(result, cfg)
                for r in range(cfg.plan_rounds):
                    for pos in SKILL_POSITIONS:
                        n = int(result.round_n[r, pos])
                        w.writerow([slot, r + 1, POSITIONS[pos], n,
                                    f"{result.round_hit[r, pos] / n:.4f}" if n else "",
                                    f"{scores[r, pos]:.4f}"])
        paths.append(path)

        path = os.path.join(outdir, "stage1_shapes.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["slot", f"first_{SHAPE_ROUNDS}_rounds", "drafts",
                        "playoff_rate", "title_rate", "mean_points"])
            for slot in sorted(stage1):
                shapes = stage1[slot].shapes
                for shape in sorted(shapes, key=lambda s: -shapes[s].playoff_rate):
                    t = shapes[shape]
                    w.writerow([slot, "+".join(POSITIONS[p] for p in shape), t.n,
                                f"{t.playoff_rate:.4f}", f"{t.title_rate:.4f}",
                                f"{t.mean_points:.1f}"])
        paths.append(path)

    if stage2:
        path = os.path.join(outdir, "stage2_screen.csv")
        _write_plan_csv(path, cfg, stage2, baseline, "screen")
        paths.append(path)

    path = os.path.join(outdir, "finalists.csv")
    _write_plan_csv(path, cfg, final, baseline, "final")
    paths.append(path)

    path = os.path.join(outdir, "anchors.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "plan", "round", "player", "position", "share"])
        for slot in sorted(final):
            if not final[slot]:
                continue
            best = final[slot][0]
            for r, names in enumerate(anchor_names(best, pool, cfg.league.rounds, top=5)):
                for name, share in names:
                    index = pool.index_of(name)
                    w.writerow([slot, plan_string(best.plan), r + 1, name,
                                POSITIONS[pool.pos[index]] if index is not None else "",
                                f"{share:.4f}"])
    paths.append(path)
    return paths


# --------------------------------------------------------------------------
# Markdown -- the version that has to survive being read a week later
# --------------------------------------------------------------------------

def write_markdown(outdir, cfg, pool, stage1, final, baseline=None, elapsed=None):
    """The report someone reads later, caveats included.

    The caveats are in it because the numbers are seductive and the model is a model.
    A reader who takes "RB-WR-WR-TE beats WR-WR-RB-TE by 1.4 points of playoff rate"
    as a fact about football, rather than as a fact about these projections under this
    opponent model, has been misled by the report rather than informed by it.
    """
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "report.md")
    lines = []
    add = lines.append
    league = cfg.league
    kind = baseline_label(cfg)

    lineup = ", ".join(f"{POSITIONS[i]}{n}" for i, n in enumerate(league.starters) if n)
    add(f"# Draft strategy by slot — {date.today().isoformat()}")
    add("")
    add(f"{league.teams} teams, {league.rounds} rounds, {lineup} + {league.flex} FLEX, "
        f"{league.bench} bench. "
        f"PPR{', tight ends at 1.5' if pool.te_premium else ', no TE premium'}. "
        f"ADP model `{pool.adp_model}`.")
    add("")
    add("## How the room was modelled")
    add("")
    if cfg.room == "value":
        add(f"All {league.teams} seats — yours included — run the same policy: take "
            f"the best value over replacement available, subject to roster rules. "
            f"They differ only in what they believe each player is worth: every "
            f"drafter draws a private view of every player, a mean-preserving "
            f"lognormal on the projection with σ = {cfg.perception_sigma:.2f}, once "
            f"per draft. **ADP does not influence who is picked when.**")
        add("")
        add("Your seat differs in exactly one way: for the first "
            f"{cfg.plan_rounds} rounds it is tied to the plan's position. That "
            "symmetry is what makes the baseline a fair yardstick — an edge here is "
            "what the *plan* is worth, not what having projections is worth.")
    else:
        add(f"{league.teams - 1} opponents draft off ADP plus noise growing with the "
            f"pick (σ = {cfg.adp_sigma_base:.1f} + {cfg.adp_sigma_rate:.2f} per "
            f"pick). Your seat drafts on value. Note that this asymmetry is larger "
            f"than any real room's, so edges measured here include the effect of "
            f"using projections at all, not just of choosing a plan.")
    add("")
    add(f"**Injuries are {'on' if cfg.injuries else 'off'}.** "
        + ("" if cfg.injuries else
           "Nobody misses a game; bye weeks still apply. Modelling injuries without "
           "modelling the waiver wire is worse than modelling neither — it forces a "
           "manager whose back misses three weeks to start his fifth-best back when "
           "in reality he adds the replacement who just inherited twenty touches, "
           "which gives bench depth a value it does not have. With them off, the "
           "bench is worth what byes make it worth, which is a defensible floor. "
           "Expect this to favour top-heavy rosters."))
    add("")
    add("Weekly scoring is Gamma with these coefficients of variation: "
        + ", ".join(f"{POSITIONS[i]} {cv:.2f}" for i, cv in enumerate(cfg.weekly_cv))
        + ".")
    add("")
    add(f"Stage 1 sampled {cfg.budget.stage1:,} random legal plans per slot; stage 2 "
        f"screened {cfg.budget.candidates} candidates at {cfg.budget.stage2:,} drafts "
        f"each; stage 3 re-measured the top {cfg.budget.finalists} at "
        f"{cfg.budget.stage3:,} drafts each on fresh draws. Ranked by "
        f"**{cfg.metric}**. Seed {cfg.budget.seed}."
        + (f" Ran in {elapsed / 60:.1f} minutes." if elapsed else ""))
    add("")

    if baseline:
        total_n = sum(t.n for t in baseline.values())
        mean = sum(t.playoffs for t in baseline.values()) / max(1, total_n)
        what = ("best-available-value drafter following no plan" if kind == "BPA"
                else "ordinary ADP drafter")
        add(f"## The {kind} baseline")
        add("")
        add(f"What a {what} gets from each slot, over "
            f"{next(iter(baseline.values())).n:,} drafts. **Every plan below is read "
            f"against its own slot's row here.**")
        add("")
        add(f"The mean across slots is **{100 * mean:.1f}%**, which doubles as the "
            "simulator's calibration check: twelve identical drafters in a twelve-team "
            "league must average 50%, so a drifting mean would be a bug in the "
            "schedule, the standings or the bracket rather than a finding.")
        add("")
        add("| Slot | Playoff % | 95% CI | Title % | Wins | Points |")
        add("|---:|---:|---:|---:|---:|---:|")
        for slot in sorted(baseline):
            t = baseline[slot]
            add(f"| {slot} | {_md_pct(t.playoff_rate)} | {_md_ci(t.playoffs, t.n)} | "
                f"{_md_pct(t.title_rate)} | {t.mean_wins:.2f} | {t.mean_points:.0f} |")
        add("")

    add("## Recommended plan by slot")
    add("")
    add("The **tied** column counts the plans that could not be distinguished from "
        "this one — a 95% interval on the *difference* of the two rates contains "
        "zero. Where it is above 1, the row is a pick from a tie and the group is "
        "listed in that slot's section below. Overlapping individual intervals are "
        "not the test used here; that test is far too conservative and would call "
        "almost everything a tie.")
    add("")
    add(f"| Slot | Plan | Archetype | Playoff % | 95% CI | {kind} base | Edge | "
        f"Tied | Title % | Points |")
    add("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for slot in sorted(final):
        if not final[slot]:
            continue
        best, group, _reason = recommend(final[slot], cfg)
        t = best.tally
        base = (baseline or {}).get(slot)
        edge = _edge(t, base)
        cells = [
            str(slot),
            f"`{plan_string(best.plan)}`",
            label_plan(best.plan),
            _md_pct(t.playoff_rate),
            _md_ci(t.playoffs, t.n),
            _md_pct(base.playoff_rate) if base else "—",
            f"{100 * edge:+.1f}" if edge is not None else "—",
            str(len(group)),
            _md_pct(t.title_rate),
            f"{t.mean_points:.0f}",
        ]
        add("| " + " | ".join(cells) + " |")
    add("")

    if stage1:
        add("## Round-by-round marginals")
        add("")
        add("Playoff rate conditioned on the position taken in that round, pooled "
            f"over all {cfg.budget.stage1:,} stage-1 drafts for the slot. This is the "
            "most reliable table in the report: thousands of drafts sit behind every "
            f"cell, where a single finalist's row rests on {cfg.budget.stage3:,}. "
            "When plans tie, this is what breaks the tie.")
        add("")
        for r in range(min(4, cfg.plan_rounds)):
            add(f"### Round {r + 1}")
            add("")
            add("| Slot | " + " | ".join(POSITIONS[p] for p in SKILL_POSITIONS)
                + " | Best |")
            add("|---:|" + "---:|" * len(SKILL_POSITIONS) + "---|")
            for slot in sorted(stage1):
                scores = round_scores(stage1[slot], cfg)
                cells = " | ".join(f"{100 * scores[r, p]:.1f}" for p in SKILL_POSITIONS)
                best_pos = max(SKILL_POSITIONS, key=lambda p: scores[r, p])
                add(f"| {slot} | {cells} | {POSITIONS[best_pos]} |")
            add("")

    add("## Finalists, by slot")
    add("")
    for slot in sorted(final):
        add(f"### Slot {slot}")
        add("")
        add(f"Your picks: {', '.join(str(p) for p in _slot_picks(cfg, slot, 6))}, ...")
        add("")
        base = (baseline or {}).get(slot)
        group = {id(r) for r in leading_group(final[slot])}
        best, tied, reason = recommend(final[slot], cfg)
        add("| | Plan | Archetype | Playoff % | 95% CI | Edge | Title % | Wins | Points |")
        add("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for result in final[slot]:
            t = result.tally
            edge = _edge(t, base)
            mark = "**→**" if result is best else ("=" if id(result) in group else "")
            cells = [
                mark,
                f"`{plan_string(result.plan)}`",
                label_plan(result.plan),
                _md_pct(t.playoff_rate),
                _md_ci(t.playoffs, t.n),
                f"{100 * edge:+.1f}" if edge is not None else "—",
                _md_pct(t.title_rate),
                f"{t.mean_wins:.2f}",
                f"{t.mean_points:.0f}",
            ]
            add("| " + " | ".join(cells) + " |")
        add("")
        if len(tied) > 1:
            add(f"**{len(tied)} plans are indistinguishable here** (marked `=`): "
                + ", ".join(f"`{plan_string(r.plan)}`" for r in tied)
                + f". Recommendation `{plan_string(best.plan)}` is the {reason}.")
        else:
            add(f"`{plan_string(best.plan)}` is separated from every other finalist.")
        add("")
        add(f"Who filled each round under `{plan_string(best.plan)}`:")
        add("")
        for r, names in enumerate(anchor_names(best, pool, league.rounds, top=3)):
            if names:
                add(f"- **R{r + 1}** " + ", ".join(f"{n} ({100 * f:.0f}%)"
                                                   for n, f in names))
        add("")

    add("## How to read this, and how not to")
    add("")
    add("- **Read the edge, not the playoff rate.** Absolute rates move with "
        "everything the room does badly and are not comparable across "
        f"configurations. The edge over the slot's {kind} baseline is what choosing "
        "this plan is worth.")
    add("- **Respect the tied column.** Where several plans tie, the model has not "
        "distinguished them and no amount of staring at the third decimal will. Pick "
        "from the group on the marginals, or on whichever plan you would rather "
        "execute at a live table.")
    add("- A plan is a *shape*, not a script. `RB-WR-WR-TE` means take the best "
        "available back, then the best two receivers, then a tight end — whoever "
        "those turn out to be when your pick comes round. The anchor lists say who "
        "that usually was.")
    add("- **Injuries are off, so the bench is worth only what byes make it worth.** "
        "Expect this report to favour top-heavy rosters more than a real season "
        "would. It will also understate any strategy whose payoff depends on "
        "surviving attrition.")
    add("- **There is no waiver wire.** No streaming, no free agents, no in-season "
        "repair. A strategy that plans to fix its own weaknesses in September cannot "
        "express that here.")
    add("- Title rate is roughly a sixth as common as a playoff berth and needs "
        "roughly six times the samples to separate plans. It is reported, not "
        "ranked on.")
    add("- Projected points are taken as truth. This measures which plan best "
        "exploits *these* projections; a systematic projection error is invisible "
        "to it. What the room disagrees about is each player's value, not whether "
        "the projections themselves are right.")
    add("- One season is simulated per draft, so a single roster's outcome is mostly "
        "noise. Only the aggregates mean anything, and the intervals say how much.")
    add("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
