# Draft Simulator — Design

Date: 2026-08-19
Status: implemented, then revised after an expert audit. This document is the
design *as built*. Sections marked **Changed during implementation** record where
the first build differed from what was approved; sections marked **during
revision** record the four changes made after the audit — injuries off, all seats
drafting on projections, higher weekly variance, and a third search stage with
tied-group reporting.

## Problem

The draft board (`draft_tiers.py` → "2026 Draft Tiers" sheet) tells you what a
player is worth and whether he survives to your next pick. It cannot tell you
what *shape* of roster to aim for from the slot you actually drew. "Take a
running back first from pick 1" and "take a running back first from pick 12" are
different claims, and nothing in the sheet distinguishes them.

This spec covers a Monte Carlo draft simulator that answers the question
empirically: run thousands of drafts from every slot against ESPN-ADP-driven
opponents, play out a season from each resulting roster, and report which draft
plans actually win from which slot.

## Scope

In scope:

- Loading the player pool with the league's real scoring (TE premium) reusing
  `draft_tiers.py`'s pricing machinery, so the simulator and the board agree on
  what a player is worth.
- A 12-team, 16-round snake draft engine in which all twelve seats draft on
  projections, differing only in their private view of each player.
- A 17-week season simulation with weekly variance, bye weeks, a 14-game schedule
  and a 6-team playoff. Injuries are modelled but off by default.
- A three-stage search over draft plans that *discovers* good plans rather than
  only grading a hand-written menu, labels the winners with readable archetype
  names, and reports the statistically tied group rather than a single winner.
- Console tables, CSVs and a markdown report, per slot.

Out of scope:

- Writing anything back to the Google Sheet. The simulator is a
  before-the-draft study, not a live board feature.
- Keepers, auctions, trades, waivers, in-season management beyond setting a
  weekly lineup.
- Calibrating opponent behaviour against this specific league's history. There
  is no such data.
- The waiver wire, and therefore injuries. This is the largest known gap and the
  reason injuries are switched off rather than modelled badly; see *Accepted
  limitations*.
- Projection error. Every drafter disagrees about what a player is worth, but the
  projections themselves are taken as true when the season is scored. The audit
  ranked this second only to waivers; it is not built.

## League configuration

Taken from `draft_tiers.py`, extended with the two roster spots the board does
not draw:

| Setting | Value |
|---|---|
| Teams | 12 |
| Rounds | 16 |
| Starters | QB 1, RB 2, WR 2, TE 1, FLEX 2, K 1, DST 1 (10) |
| Flex eligibility | RB, WR, TE |
| Bench | 6 |
| Scoring | PPR, tight ends 1.5 per reception |
| Regular season | weeks 1–14 |
| Playoffs | weeks 15–17, 6 teams, seeds 1–2 bye in week 15 |

K and DST are startable, which is why the roster is 16 and not 14. They are
drafted only in rounds 15 and 16 (see *Roster rules*), so they never compete
with a skill-position decision and never enter the plan space.

## Player pool

`draftsim/pool.py` builds one immutable `Pool` of parallel numpy arrays:
`name`, `pos` (integer code), `ppg`, `adp`, `rank`, `bye`.

Sources:

- `Draft-rankings-export-2026 (8-21).csv` — overall rank, position, bye week,
  ESPN ADP, projected points.
- `projections (8-21).csv` — receptions, used only by the TE premium.

Pricing reuses `draft_tiers.read_players()` unchanged rather than duplicating
the premium logic. That function drops the bye week, so bye weeks are joined
back on by name from the same CSV. It is called with no rank limit so kickers
and defenses (overall ranks 193+) survive; the pool is then capped at the top
`--pool-skill` (default 300) skill players plus all kickers and defenses.

### ADP model

Two modes, selected with `--adp-model`:

- `premium` (default) — the ADP `read_players()` produces, in which the tight
  end premium has already shifted the market's clock. This is the board's own
  view of the league and the room the user is actually drafting in.
- `espn` — the export's untouched ESPN ADP, i.e. a public standard-scoring
  room.

The default is `premium` because it matches the board the user already trusts;
`espn` exists because the user asked for "ESPN ADP" and it is the honest
baseline when the premium's market model is the thing in doubt.

### ADP holes and the 171 plateau

ESPN reports ADP only out to ~171 and leaves a handful of players null. Two
fixes, both explicit:

- Missing ADP falls back to overall rank (already `read_players()`'s
  behaviour). Ranks run to 512 and picks only to 192, so rank-as-ADP correctly
  reads as "undrafted".
- Dozens of players share an ADP near 170–171 — ESPN's "essentially undrafted"
  plateau. Sampling that plateau uniformly would ignore projection entirely, so
  a deterministic tie-break toward the better-ranked player is folded into the
  sort key.

  **Changed during implementation.** The tie-break was specified as
  `adp + 1e-3 * rank`, which is wrong: overall ranks reach 469 in this pool, so
  the deepest players picked up a nudge of nearly half a pick — five times the
  0.1 that separates two genuinely different reported ADPs. It is now
  `adp + ADP_TIEBREAK_SPAN * rank / worst_rank` with a span of 0.05, so the
  whole nudge is bounded however deep the pool goes. A test asserts the bound
  against the loaded data rather than against the constant.

## Value model

### Weekly points

A player's weekly score is Gamma-distributed with mean equal to his projected
PPG and a position-specific coefficient of variation:

| Pos | CV |
|---|---|
| QB | 0.35 |
| RB | 0.65 |
| WR | 0.75 |
| TE | 0.80 |
| K | 0.45 |
| DST | 0.75 |

Gamma rather than normal because weekly fantasy scores are non-negative and
right-skewed. These are league-agnostic defaults from public weekly-scoring
dispersion, stated as tunable constants in `draftsim/config.py`, not buried
magic numbers. Player weeks are independent: no QB–WR stacking correlation.
That understates the variance of a stacked roster and is accepted as YAGNI.

**Raised during revision** from RB 0.55 / WR 0.60 / TE 0.60. A receiver posting
3 points one week and 28 the next is unremarkable and CV 0.60 does not produce
it, and tight ends are the most volatile skill position per point scored rather
than equal to receivers. Understating weekly variance made the simulator more
confident than a fantasy season ever is, which inflated the apparent gap between
plans and narrowed every interval.

### Injuries — off by default

**Nobody misses a game unless `--injuries` is passed.** Bye weeks still apply, so
a roster still has weeks where somebody cannot be started, but no player is ever
hurt.

Switched off deliberately. Modelling injuries without also modelling the waiver
wire is worse than modelling neither: it forces a manager whose back misses three
weeks to start his fifth-best back, when in reality he adds the replacement who
just inherited twenty touches. That combination gives bench depth a value it does
not have and quietly recommends hoarding — most visibly by making a backup
quarterback look worth a pick, which in a one-quarterback league he is not. With
injuries off, the bench is worth exactly what bye weeks make it worth, which is a
defensible floor rather than an inflated guess.

**Consequence to keep in view: this favours top-heavy rosters,** and it
understates any strategy whose payoff depends on surviving attrition.

The sampler is kept and is correct, so it can be switched back on the day a
free-agent pool exists to make it meaningful. Per player per season, with
probability `INJURY_MISS_ANY_PROB[pos]` he misses a contiguous block of games:

| Pos | P(misses any) |
|---|---|
| QB | 0.38 |
| RB | 0.55 |
| WR | 0.45 |
| TE | 0.45 |
| K | 0.10 |
| DST | 0.00 |

Block length is `Geometric(1 / MISS_MEAN_WEEKS)` — numpy's geometric counts
trials to a first success, so it is already at least one week, has its mode at
one week and a mean of three — placed at a uniform random start week and clipped
to the season. Contiguous rather than per-week Bernoulli because clustering is
the point: three separate one-week absences cost a manager almost nothing, and a
three-week block costs a real starter.

Note that if these are switched back on as they stand they are themselves too
low: they give a running back about 1.6 expected missed games where the real
figure is nearer 3.

### Bye weeks

A player scores nothing in his bye week and is unavailable for lineup
purposes. All byes in this data fall in weeks 5–14, inside the regular season.

## Draft engine

`draftsim/draft.py`. Standard snake: 12 teams, 16 rounds, 192 picks. Odd
rounds ascend, even rounds descend. The user's team occupies one slot, held
fixed for a whole batch of simulations.

### Roster rules

Enforced identically for every team, opponents and user alike:

| Position | Max on roster |
|---|---|
| QB | 2 |
| RB | 6 |
| WR | 7 |
| TE | 3 |
| K | 1 |
| DST | 1 |

K and DST are legal picks only in rounds 15–16. A **starter-completion guard**
applies to everyone: when the number of picks a team has left equals the number
of its unfilled mandatory starter slots, it may only draft positions that fill
one. Without the guard, teams reach the end of the draft with no tight end and
score zero from that slot, which is not a mistake real drafters make and would
distort what falls to the user.

### Opponent policy — the `value` room (default)

**Replaced during revision.** The original model had eleven opponents drafting
purely off ADP while the user's team drafted off projections. That asymmetry is
larger than any real room's — ADP *is* the aggregate of everybody's rankings, and
those rankings come from projections closely correlated with these ones — and it
produced a measured plan edge of about +37 points of playoff rate, most of which
was projection arbitrage rather than strategy.

Now all 12 seats, the user's included, run one policy: take the player whose
bench-discounted value over replacement is highest. They are not clones, because
each drafter brought a different opinion of what each player is worth:

    perceived_ppg = ppg × exp(Normal(−σ²/2, σ))      σ = PERCEPTION_SIGMA = 0.15

drawn **once per drafter per draft**, not per pick — a drafter's opinion is a fixed
thing he brought to the table, and re-rolling it every pick would model indecision
rather than disagreement and average back out to no disagreement at all. The
lognormal is mean-preserving; without the `−σ²/2` shift every drafter would be
systematically optimistic about every player, which would bias every comparison
against a replacement level computed from the unperturbed projections — and the
bench discount is exactly such a comparison.

This disagreement is now the **only** source of draft-to-draft variety. With
`--perception-sigma 0` the room is fully deterministic and every draft from a given
slot is identical, which is a legitimate test fixture and a useless study.

Season scoring and weekly lineups use the **true** projected PPG. Perception is a
draft-day error whose cost shows up in the season; it is not an in-season
handicap, because a manager's weekly decisions are informed by results he has
already seen and so are better informed than his draft was.

ADP no longer influences who is picked when. It still determines pool ordering and,
under `--adp-model premium`, the premium repricing of PPG.

Positional runs still emerge, now from disagreement plus the roster rules rather
than from ADP noise. Runs are the reason slot matters at all, so they must be
emergent and not a tuning knob.

### Opponent policy — the `adp` room

The original model, kept behind `--room adp` for exactly one purpose: running the
study both ways is how you find out how much of a plan's edge was strategy and how
much was having projections at all. Each of the 11 opponents picks the legal
available player minimising

    adp_effective + Normal(0, sigma(p))

where `sigma(p) = ADP_SIGMA_BASE + ADP_SIGMA_RATE * p` over the overall pick
number `p`. With base 4.0 and rate 0.10 that is ±4 picks of noise at the top of
round 1 and ±23 by the end of the draft, which reproduces the 15–25 pick
dispersion through rounds 4–10 that `draft_tiers.py` already documents. Growing
noise, not a constant band, because early ADP is a near-consensus and late ADP is
barely an opinion.

Note that the baseline means something different in each room, and the report says
which: in the `value` room it is a best-available-value drafter following no plan,
so an edge is what the *plan* is worth; in the `adp` room it is an ADP drafter, and
an edge conflates the plan with the much larger effect of using projections.

### User policy: the plan

A **plan** is a tuple of 8 positions over {QB, RB, WR, TE} — one per round for
rounds 1–8. It is the object the search optimises.

- Rounds 1–8: take the best available player at `plan[r-1]`, best by projected
  PPG within that position. If the position is exhausted or illegal, fall back
  to the best value-over-replacement pick available.
- Rounds 9–14: need-aware value fill — the highest value-over-replacement
  player among positions not at their roster cap, subject to the
  starter-completion guard. Replacement level per position comes from
  `draft_tiers.starter_depths()`, so it is the same definition the board uses.
- Round 15: best kicker. Round 16: best defense.

**Added during implementation: bench discounting.** Plain value over replacement
turned out to recommend a second quarterback around round 11 — the twentieth-best
quarterback still clears quarterback replacement, and the arithmetic has no idea
only one of him can start. The value fill now scores a candidate as

    BENCH_DECAY ** depth * ppg - replacement[pos]

where `depth` is how far past *startable* the pick lands at that position and
`BENCH_DECAY` is 0.35. Two details are load-bearing:

- The weight multiplies projected points, and replacement is subtracted
  afterwards. Scaling the finished difference instead would shrink a *negative*
  value toward zero and make hoarding a bench look better than filling a hole.
- `depth` is measured **per position** against `starters[pos] + flex`, not against
  the pooled flex requirement that roster legality uses. The first attempt pooled
  it, which made a fifth receiver count as five deep — his team's spare backs and
  tight ends counted against him too — driving his discount to nearly nothing and
  letting the backup quarterback win on raw points anyway. Legality pools the flex
  because it is about filling one lineup; depth does not, because it is about how
  often this particular player takes the field.

### Legal plans

A plan is legal if it has at most 2 QB, at most 2 TE, at least 1 RB and at
least 1 WR across the eight rounds. The QB and TE caps cut plans no one would
run; the RB and WR floors keep the plan space from filling with rosters that
cannot field a lineup. A true zero-RB-through-round-8 plan is therefore
excluded, but zero-RB-through-round-4 — the strategy people actually mean — is
not.

## Season simulation

`draftsim/season.py`, one season per completed draft. Noise is averaged over
thousands of drafts rather than by replaying each draft many times, which buys
plan coverage with the same compute.

1. Sample availability (injury block ∪ bye) and realised weekly points for
   every drafted player, 17 weeks.
2. Each week, every team fills its lineup from **projected** PPG among
   available players — never from realised points. A manager does not know
   Sunday's score on Saturday, and lineups set from realised points would
   reward luck as if it were roster quality.
3. Fill order: mandatory slots QB, RB, RB, WR, WR, TE, K, DST by projected PPG
   descending, then 2 FLEX from the remaining RB/WR/TE. An unfillable mandatory
   slot scores zero.
4. Schedule: a fixed 11-week round robin plus 3 repeat weeks, with the mapping
   from teams to schedule positions reshuffled every draft so no slot inherits
   a permanent schedule edge.
5. Standings by wins, points for as tiebreak. Top 6 seeded; week 15 is 3v6 and
   4v5, week 16 is 1 vs the lower surviving seed and 2 vs the other, week 17 is
   the final.
6. Returns, for the user's team: made playoffs, won title, total starter
   points.

## Search

`draftsim/search.py`, two stages per slot.

**Stage 1 — explore.** `--stage1` drafts (default 12,000) per slot, each with a
uniformly sampled legal plan. Produces:

- Per-round marginals: playoff rate conditioned on the position taken in each
  of rounds 1–8.
- Per-shape marginals: playoff rate conditioned on the multiset of positions
  taken in rounds 1–4.

**Stage 2 — screen.** Candidates come from four sources, deduplicated in order
and capped at `--candidates` (default 24). Each gets `--stage2` drafts (default
1,200). A screening pass only: at that budget the 95% half-width is about 2.8
points, which is enough to sort two dozen candidates into a plausible order and
nothing like enough to name a winner.

1. Every named archetype, unconditionally, as a reference row.
2. The best legal plans under stage 1's per-round marginals — an exact search over
   the enumerated legal space, not a greedy pick that would need repairing.
3. The best legal plan for each of the best opening four-round *shapes*.
4. Plans that did well in stage 1 on their own and cleared a sample floor.

Whatever room remains after all four is topped up from source 2.

**Changed during implementation.** Source 3 was not in the approved spec, and the
first build measured only 16 of its 24 candidates without saying so. The cause was
source 4: 12,000 drafts spread over 25,174 plans is half a sample per plan, so
essentially nothing clears the floor and the source that was meant to fill the
last third of the list contributed nothing. Shapes fixed it because they are the
one thing stage 1 measures precisely — there are only dozens of them, so thousands
of drafts sit behind each. Source 4 is kept for small runs and moved last.

**Stage 3 — decide.** Added during revision. The top `--finalists` (default 8)
plans per slot are re-measured at `--stage3` drafts each (default 5,000), which
narrows the 95% half-width to about 1.0 point.

It exists because stage 2 was being over-read. The gap between the best few plans
at a slot is one to three points of playoff rate and stage 2's interval is wider
than that, so a table naming one best plan per slot was presenting noise as a
decision.

Stage 3 **re-measures rather than accumulating on top of stage 2**, which costs 20%
more simulations and is the whole point: stage 2 promoted these plans *because*
their observed rates were high, so their stage-2 counts are biased upward by that
selection, and adding them in would carry the winner's curse straight into the
final number. The stage index is folded into each task's seed, so stage 3's draws
are genuinely fresh.

Three stages because the plan space is far larger than any per-plan sample budget
allows. Stage 1 cannot rank individual plans honestly — it can only point at
regions. Stage 2 can order a shortlist but not resolve it. Stage 3 resolves what
can be resolved and the report says plainly what cannot.

**Tied groups.** A plan is called indistinguishable from the leader when a 95%
confidence interval on the **difference** of the two playoff rates contains zero:

    se = sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)      tie if |p1 − p2| < 1.96 · se

Not when the two plans' own intervals overlap. Overlap is the test everybody
reaches for and it is the wrong one — it behaves like a test at roughly the 0.83
level rather than 0.05, so it declares ties between plans that are in fact
separated. A test in the suite exhibits a concrete pair where the two disagree.

Ties are measured against the leader only, never transitively: a chain of pairwise
ties can run arbitrarily far down a list, and "tied with the best" is the claim a
reader needs. Where a group ties, one member is still named as a recommendation —
chosen on the stage-1 marginal score, because thousands of drafts sit behind every
per-round cell where a finalist has a few thousand in total — and it is labelled as
a pick from a tie rather than as a measured best.

**The ADP baseline.** Added during implementation, and the most important thing
that was missing. Before any plan is measured, `--baseline` drafts (default 4,000)
are run in which nobody follows a plan and all twelve seats are ADP drafters. It
does two jobs:

- It is the yardstick. A plan reaching the playoffs 84% of the time means nothing
  until you know an ADP drafter from that slot gets 47%. Every plan is reported
  with its edge over its own slot's baseline, and the report says in three places
  that the edge is the number to act on.
- It is the calibration check. Twelve identical drafters in a twelve-team league
  must average a 50% playoff rate, and because one draft yields twelve baseline
  samples the check is exact rather than statistical. A drift means the schedule,
  the standings or the bracket is broken and no other table is worth reading.

Parallelism: `multiprocessing` over (slot, chunk) in stage 1 and (slot, candidate,
chunk) in stage 2. Every worker seeds from `--seed` and its own task coordinates.

**Changed during implementation.** Chunk sizes were originally derived from the
worker count, which silently broke reproducibility: chunk boundaries decide which
seeds each chunk draws, so the same run at `--jobs 1` and `--jobs 4` sampled
different drafts and reported different numbers. Batches are now cut into a fixed
64 chunks regardless of `--jobs`, and a test asserts the two agree.

## Reporting

`draftsim/report.py`.

Archetype labels are rule-based over the eight-round plan and composable, e.g.
"Hero-RB / Elite-TE / Late-QB":

- Zero-RB: no RB in rounds 1–4
- Hero-RB: RB in round 1 or 2, then none until round 5
- Robust-RB: 2 or more RB in rounds 1–3
- Elite-TE: TE in rounds 1–3
- Early-QB: QB in rounds 1–4; Late-QB: QB in rounds 7–8 or not at all
- WR-heavy: 4 or more WR in the eight

Outputs:

- Console: the ADP baseline table (with its calibration line), a cross-slot
  summary of the best plan per slot with its interval and its edge, the round 1–3
  marginal tables, then per-slot tables of the top plans with their anchors.
- `sim_out/baseline.csv`, `stage1_marginals.csv`, `stage1_shapes.csv`,
  `stage2_plans.csv`, `stage2_anchors.csv`, `report.md`.
- For the best plan in each slot, the players most often drafted at each round
  — the concrete anchors behind the abstract plan.

Headline metric is **playoff rate**, reported both absolutely and as an edge over
the slot's ADP baseline. Title rate is reported alongside but is roughly six times
rarer and needs far more samples to separate plans; points are reported as a
sanity check on both.

## Module layout

| File | Responsibility |
|---|---|
| `draftsim/config.py` | League, CVs, injury constants, room, perception, bench decay, budget |
| `draftsim/pool.py` | CSV → `Pool` arrays; premium pricing; ADP model |
| `draftsim/roster.py` | Roster caps, legality, bench depth, lineup solvers |
| `draftsim/bots.py` | Opponent pick policy |
| `draftsim/plans.py` | Plan legality, sampling, archetypes, labelling |
| `draftsim/draft.py` | Snake order and the draft loop |
| `draftsim/season.py` | Weekly scoring, injuries, schedule, playoffs |
| `draftsim/search.py` | Three-stage search, tie test, parallel execution |
| `draftsim/report.py` | Tables, CSVs, markdown, tied-group reporting |
| `simulate_drafts.py` | CLI |
| `tests/` | pytest suite |

Each module is independently testable and none reaches into another's
internals. `draft.py` is the only module that knows the order of picks;
`season.py` is the only one that knows about weeks.

## CLI

    python simulate_drafts.py                       # full run, all 12 slots (~5 min)
    python simulate_drafts.py --quick               # small sample, for a smoke test
    python simulate_drafts.py --slots 1,6,12        # only these slots
    python simulate_drafts.py --adp-model espn      # untouched ESPN ADP
    python simulate_drafts.py --metric title        # rank by title rate instead
    python simulate_drafts.py --room adp            # the old ADP-only opponents
    python simulate_drafts.py --injuries            # switch injuries back on
    python simulate_drafts.py --perception-sigma 0  # a deterministic room
    python simulate_drafts.py --plan RB-RB-WR-WR-TE-WR-QB-RB   # grade one plan

Flags: `--rankings`, `--projections`, `--pool-skill`, `--adp-model`,
`--no-te-premium`, `--teams`, `--rounds`, `--starters`, `--flex`,
`--regular-weeks`, `--total-weeks`, `--playoff-teams`, `--slots`, `--stage1`,
`--stage2`, `--stage3`, `--candidates`, `--finalists`, `--metric`, `--baseline`,
`--room`, `--perception-sigma`, `--injuries`, `--bench-decay`, `--plan`, `--seed`,
`--jobs`, `--quick`, `--out`, `--top`, `--no-files`, `--quiet`.

`--plan` skips the search entirely and measures one named plan from every slot.
It has its own path rather than being a one-candidate search because stage 1
would be pure waste when the candidate list is already known.

## Testing

Test-driven; pytest in `tests/`.

1. **Snake order** — slot *k* owns picks *k* and *2T+1−k*; 192 picks, each team
   16.
2. **Draft integrity** — no player drafted twice; roster caps respected; every
   team ends with exactly one K and one DST, taken in rounds 15–16 only; every
   team can field all 10 mandatory starters.
3. **Lineup solver** — a hand-built roster yields the known optimal legal
   lineup; unavailable players are excluded; FLEX never takes QB/K/DST; an
   unfillable slot scores zero, not an error. The fast vectorised solver and the
   readable single-team one are cross-checked on random rosters, which is the only
   reason it is safe to have two.
4. **Opponent policy** — in the value room, a draft is byte-identical when the ADP
   key is scrambled (no ADP influence survives), all twelve seats produce a fully
   determined draft when disagreement is switched off (the symmetry the baseline
   depends on), and two seeds diverge when it is switched on (disagreement is the
   only remaining source of variety). Perception is mean-preserving. In the adp
   room with sigma zero, the first pick is the market's first name. A value room
   leaves a plan less than an adp room does.
5. **Injuries** — nobody misses a game by default and byes are the only absence;
   switching them on visibly reduces availability. With them on: missed games are
   contiguous, within 1–17, DST never misses, and the empirical rate matches
   `INJURY_MISS_ANY_PROB` within tolerance.
6. **Season integrity** — total wins equals total games; exactly 6 playoff
   teams; exactly 1 champion.
7. **Plans** — sampled plans are always legal; known plans get the expected
   archetype labels.
8. **Pool** — every pooled player has a bye week; the ADP tie-break orders the
   171 plateau by rank without reordering anyone else.
9. **Determinism** — the same seed gives identical results, a different seed does
   not, and the same seed gives identical results at `--jobs 1` and `--jobs 3`.
10. **Calibration** — twelve identical ADP drafters average exactly a 50% playoff
    rate, and no single slot is far off a coin flip. This is the test that would
    catch a broken schedule, a broken standings tiebreak or a broken bracket, and
    it is the reason the baseline is computed on every run rather than only in the
    test suite.
11. **Bench discounting** — a backup quarterback ranks deeper than a fourth running
    back; the value fill stops hoarding second quarterbacks at decay 0.35 and
    resumes at decay 1.0.
12. **Budget** — every stage spends exactly the drafts it is given, all
    `--candidates` slots are filled, stage 3 measures exactly `--finalists` plans
    drawn from stage 2's list, and anchor counts account for every round of every
    draft.
13. **The tie test** — identical rates tie; a large gap does not; a four-point gap
    is invisible at 500 drafts and clear at 5,000 (the argument for stage 3); a
    concrete pair exists where the individual intervals overlap yet the difference
    is significant (the argument against the overlap test); the leading group always
    contains the leader and is never transitive; stage 3 redraws rather than
    accumulating on stage 2.
14. **Reporting** — the console names every section, shows the edge and not only the
    rate, declares ties rather than a bare winner, and states the model it ran
    (room, injuries, CVs) so a table can never be read without its configuration.

138 tests after the revision; all passing.

## Accepted limitations

Each is a modelling choice, not an oversight. The first two came out of an expert
audit and are the largest known gaps.

- **There is no waiver wire.** No streaming, no free agents, no in-season repair.
  This is the single biggest structural omission, and it is why injuries are off:
  modelling attrition without modelling the response to attrition gives bench depth
  a value it does not have. Any strategy whose payoff depends on fixing its own
  weaknesses in September — Zero-RB above all — cannot express that here, so its
  numbers should be read as a floor.
- **Projected points are taken as truth when the season is scored.** Drafters
  disagree about what a player is worth, but nobody is *wrong*: there is no
  true-talent draw separating a projection from an outcome. So this measures which
  plan best exploits *these* projections. A systematic projection error is invisible
  to it, and the payoff to reaching for a projection outlier is overstated. The fix
  is a per-player lognormal true-talent draw at the start of each simulated season,
  with σ scaled by ADP; it is not built.
- **Injuries are off**, so the bench is worth only what bye weeks make it worth.
  Expect the report to favour top-heavy rosters more than a real season would.
- Opponent disagreement is set by one constant with no data behind it. σ = 0.15 is
  disagreement between people reading the *same* numbers, deliberately smaller than
  disagreement between published projection sets.
- No week-to-week correlation between teammates, so a QB–WR stack is priced as
  independent. Affects variance, not means — matters more for title rate than for
  playoff rate.
- No in-season lineup error: weekly lineups are set from the true projections even
  though the draft was made on a perceived version. A manager's weekly decisions are
  better informed than his draft was, so this is a defensible simplification rather
  than a free lunch, but it is a simplification.
- Your team's draft policy is greedy on value, not survival-aware. A real drafter
  asks "will he last to my next pick?"; this one does not, so the *plan* carries all
  the scarcity-timing work that a live drafter does pick by pick.
- Rounds 9–14 are a fixed heuristic rather than part of the search. It is a good
  heuristic, which means it partly repairs bad plans and compresses the differences
  the study is trying to measure.
- ESPN's projections are compressed — they regress toward the mean and are generous
  to quarterbacks. Compressed projections mechanically favour whichever position has
  the scarcest replacement, so any positional conclusion is worth checking against a
  second source.
- Week 17 is played as an ordinary week; real starters rest. Many leagues end at
  week 16 for that reason.
- One season is simulated per draft, so a single roster's outcome is mostly noise.
  Only the aggregates mean anything, and the reported intervals say how much.
- Playoff rate, not title rate, is the headline, because title rate is roughly a
  sixth as common and needs roughly six times the samples to separate plans.
