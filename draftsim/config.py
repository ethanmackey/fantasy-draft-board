"""League settings and the model constants every other module reads.

Everything here is a number someone could reasonably want to argue with, which is
why it lives in one file with the argument attached. Nothing in this package
hard-codes a rate, a cap or a coefficient of variation anywhere else.

The league half of it is deliberately the same league ``draft_tiers.py`` builds
the board for, plus the two roster spots the board does not draw (K and DST). If
those two files ever disagree about the lineup, the board is the one that is
right and this file is the one that is stale.
"""

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Positions.
#
# Integer codes rather than strings because every hot loop in the package indexes
# arrays by position: a dict lookup on "RB" 200 times per draft times 500,000
# drafts is real time. POSITIONS is the authoritative order; POS_INDEX inverts it.
# --------------------------------------------------------------------------

POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")
POS_INDEX = {name: i for i, name in enumerate(POSITIONS)}

QB, RB, WR, TE, K, DST = range(6)

# The positions a plan can name and the search can reason about. Kickers and
# defenses are excluded on purpose -- see LATE_ONLY below.
SKILL_POSITIONS = (QB, RB, WR, TE)
# Flex eligibility. Tight ends are in it, and in a premium league that is not a
# formality: they win flex slots here that they would never win at 1.0 PPR, which
# is what pushes TE replacement level deeper than the usual TE12.
FLEX_POSITIONS = (RB, WR, TE)
# Drafted only in the last two rounds, in this order, one each. They are startable
# but there is no decision in them: the gap between the best kicker and the
# twelfth is smaller than one week of noise, so letting them compete for an early
# pick would only add variance to the results without adding a strategy.
LATE_ONLY = (K, DST)


# --------------------------------------------------------------------------
# The league.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class League:
    """One league's rules. Frozen because a batch of simulations shares one.

    ``starters`` is per team per position, not counting the flex. ``rounds``
    therefore has to be at least mandatory + flex + one K + one DST or teams
    cannot field a lineup; ``validate`` says so rather than letting the draft
    silently produce zero-scoring slots.
    """

    teams: int = 12
    rounds: int = 16
    # QB 1, RB 2, WR 2, TE 1 -- the board's LINEUP.
    starters: tuple = (1, 2, 2, 1, 1, 1)
    flex: int = 2
    regular_weeks: int = 14
    total_weeks: int = 17
    playoff_teams: int = 6

    # Roster caps, per position, in POSITIONS order. These are not ESPN's limits;
    # they are the limits a drafter who wants to win imposes on themselves. A bot
    # allowed a third quarterback takes one eventually, and every one it takes is
    # a player that should have fallen to somebody.
    caps: tuple = (2, 6, 7, 3, 1, 1)

    def __post_init__(self):
        self.validate()

    @property
    def mandatory(self):
        """Total starting slots including the flex."""
        return sum(self.starters) + self.flex

    @property
    def bench(self):
        return self.rounds - self.mandatory

    @property
    def picks(self):
        return self.teams * self.rounds

    @property
    def playoff_weeks(self):
        return self.total_weeks - self.regular_weeks

    def validate(self):
        if self.teams < 2:
            raise ValueError("a league needs at least two teams")
        if len(self.starters) != len(POSITIONS):
            raise ValueError(f"starters must have {len(POSITIONS)} entries, "
                             f"one per position in {POSITIONS}")
        if len(self.caps) != len(POSITIONS):
            raise ValueError(f"caps must have {len(POSITIONS)} entries")
        if self.rounds < self.mandatory:
            raise ValueError(
                f"{self.rounds} rounds cannot fill {self.mandatory} starting slots")
        for pos, (need, cap) in enumerate(zip(self.starters, self.caps)):
            if cap < need:
                raise ValueError(f"{POSITIONS[pos]} cap {cap} is below its "
                                 f"{need} starting slot(s)")
        if self.playoff_weeks < 1:
            raise ValueError("the season needs at least one playoff week")
        # A 6-team bracket needs three weeks; a 4-team one needs two. Rather than
        # enumerate formats, require enough weeks for a single-elimination bracket
        # over the seeded field.
        rounds_needed = max(1, (self.playoff_teams - 1).bit_length())
        if self.playoff_weeks < rounds_needed:
            raise ValueError(f"{self.playoff_teams} playoff teams need "
                             f"{rounds_needed} playoff weeks, have {self.playoff_weeks}")
        if self.playoff_teams > self.teams:
            raise ValueError("more playoff teams than teams")


DEFAULT_LEAGUE = League()


# --------------------------------------------------------------------------
# Weekly scoring noise.
#
# Coefficient of variation, per position, of a player's weekly PPR score around
# his own mean. Public weekly-scoring dispersion sits in these neighbourhoods:
# quarterbacks are the steadiest thing in fantasy (volume is guaranteed, and a
# passing floor of 15 is normal), defenses the least steady (a return touchdown is
# a third of a week's score and nobody can predict one).
#
# Gamma, not normal: a weekly score cannot be negative and its right tail is long.
# Gamma with shape 1/cv^2 and scale mean*cv^2 has exactly the requested mean and
# CV, and for the CVs below the shape is 1.6-8.2 -- comfortably away from the
# degenerate end.
#
# These decide how often the better roster loses anyway, which sets how many drafts
# are needed to separate two plans. They do not change which roster is better -- but
# understating them makes the simulator more confident than a fantasy season ever
# is, which inflates the apparent gap between plans.
#
# Raised from an earlier, tamer set (RB .55 / WR .60 / TE .60). A receiver posting 3
# points one week and 28 the next is unremarkable and CV 0.60 does not produce it;
# tight ends are the most volatile skill position per point scored, not equal to
# receivers.
# --------------------------------------------------------------------------

WEEKLY_CV = (0.35, 0.65, 0.75, 0.80, 0.45, 0.75)

# A floor on the mean before the gamma is sampled. A player projected at 0.0 PPG
# has an undefined gamma scale; he is also the deep-bench filler whose weekly
# score is genuinely near zero, so a small positive mean is both safe and honest.
MIN_PPG = 0.05


# --------------------------------------------------------------------------
# Injuries -- ON.
#
# They were switched off for one run, and the reason was sound at the time:
# modelling attrition without modelling the response to attrition is worse than
# modelling neither, because it forces a manager whose back misses three weeks to
# start his fifth-best back when in reality he adds the replacement who just
# inherited twenty touches. That hands bench depth a value it does not have.
#
# The waiver wire is the response, and it now exists (see FREE_AGENT_DEPTH). With it
# in place, injuries make the model strictly more realistic rather than less, and
# they restore the bench to meaning something: without them, rounds 9-14 exist only
# to cover bye weeks.
#
# Rates are expressed as expected games missed, not as P(misses any), because that is
# the quantity anyone can check against a season's snap counts and the quantity the
# old numbers got wrong. P(any) of 0.55 with a three-week mean block gave a running
# back 1.65 expected missed games; the real figure is nearer 3, so the model was
# under-injuring by about half.
#
# Backed out to the sampler's parameters: P(any) = expected / mean_block. A running
# back at 3.2 expected games and a 4.5-week mean block misses a block 71% of seasons.
# --------------------------------------------------------------------------

INJURIES_DEFAULT = True
# Expected games missed per season, per position: QB, RB, WR, TE, K, DST.
# Backs take the most contact at the highest snap share; a team defense cannot be
# injured at all, it just plays worse.
EXPECTED_GAMES_MISSED = (2.0, 3.2, 2.3, 2.3, 0.4, 0.0)
# Mean length of a missed-games block, in weeks, given that there is one. Sampled as
# Geometric(1/mean), which counts trials to a first success and so is already at least
# one week, with its mode at one week -- the overwhelmingly common case -- and a tail
# long enough to reach a season-ender.
MISS_MEAN_WEEKS = 4.5
INJURY_MISS_ANY_PROB = tuple(min(1.0, g / MISS_MEAN_WEEKS)
                             for g in EXPECTED_GAMES_MISSED)
NO_INJURIES = (0.0,) * len(POSITIONS)


# --------------------------------------------------------------------------
# Projection error.
#
# The deepest thing the model was missing. Drafters disagreed about what a player was
# worth (see PERCEPTION_SIGMA) but nobody was ever *wrong*: the season was scored off
# the same projections the draft optimised against, so any bias in the source became
# risk-free profit and reaching for a projection outlier was free.
#
# So each simulated season draws a true ability per player,
#
#     true_ppg = projected_ppg * exp(Normal(-s^2/2, s))
#
# mean-preserving, and nobody -- not the drafters, not the lineup setter -- ever sees
# it. Weekly scores are generated from it and lineups are chosen on it (a manager
# works out who is actually good over fourteen weeks, even though he cannot know
# Sunday's score in advance, which is the no-clairvoyance rule this respects).
#
# Lognormal because busts floor out and breakouts have a long right tail; a normal
# would make a 20-PPG back's downside symmetric with his upside, which is not how a
# season goes wrong.
#
# Sigma rises with overall rank, because a first-round pick is far more predictable
# than a tenth-round one: the multiplier runs from 1.0 at rank 1 to
# 1 + PROJECTION_SIGMA_TAIL at PROJECTION_SIGMA_DEPTH and is flat past that.
#
# Expect this to shrink every measured edge. That is the point: what remains is the
# part of a plan's advantage that survives not knowing which projections are right.
# --------------------------------------------------------------------------

PROJECTION_ERROR_DEFAULT = True
# Season-long PPG error, per position, at the very top of the board.
PROJECTION_SIGMA = (0.25, 0.38, 0.33, 0.38, 0.30, 0.35)
PROJECTION_SIGMA_TAIL = 0.5
PROJECTION_SIGMA_DEPTH = 200.0


# --------------------------------------------------------------------------
# The room.
#
# Two opponent models. `value` is the default and the one the study runs on.
#
#   value -- all twelve seats, yours included, run the same policy: take the best
#     value over replacement available. They differ only in what they believe each
#     player is worth (see PERCEPTION_SIGMA) and, for your seat, in being tied to a
#     plan for the opening rounds. ADP does not influence who is picked when at all.
#
#   adp -- the original model: eleven opponents take the market's next name with
#     noise that grows through the draft. Kept because it is the honest description
#     of a public room that has never opened a projection, and because it is the
#     comparison that shows how much of a plan's edge was really projection
#     arbitrage rather than strategy. It is not the default because the asymmetry it
#     creates is larger than any real room's: ADP *is* the aggregate of everybody's
#     rankings, and those rankings come from projections highly correlated with
#     these ones.
# --------------------------------------------------------------------------

ROOMS = ("value", "adp")
DEFAULT_ROOM = "value"

# Spread of a drafter's private opinion of a player, as a lognormal multiplier on
# projected PPG: perceived = ppg * LogNormal(-s^2/2, s), which is mean-preserving so
# nobody is systematically optimistic.
#
# This is where draft-to-draft variety comes from once ADP is out of the picture. A
# room of twelve identical value drafters reading identical numbers produces the
# same draft every time, and a simulator with no draft variance cannot say anything
# about what falls to you.
#
# Drawn once per drafter per draft, not once per pick. A drafter's opinion of a
# player is a fixed thing he brought to the table; re-rolling it every pick would
# model somebody changing his mind sixteen times, which is not disagreement, it is
# indecision -- and it would wash out to no disagreement at all.
#
# 0.15 puts roughly two thirds of opinions within 15% of the projection. That is
# smaller than real disagreement between published projection sets, deliberately:
# it is disagreement between people reading the *same* numbers, not between sources.
PERCEPTION_SIGMA = 0.15

# --------------------------------------------------------------------------
# The market -- only reached in the `adp` room.
#
# Opponents pick the available player with the lowest ADP plus noise. The noise
# grows with the pick number because ADP early is a near-consensus and ADP late is
# barely an opinion: everybody agrees on the first three picks and nobody agrees
# on the 140th.
#
# Calibrated against the dispersion draft_tiers.py already documents for this
# data -- 15-25 picks either side through rounds 4-10. At base 4.0 and rate 0.10
# the standard deviation is 4 picks at 1.01, ~9 at pick 50, ~16 at pick 120 and
# ~23 at the end, so the +/- 1 sigma band through rounds 4-10 (picks 37-120) runs
# 8-16 and the 2-sigma band runs 16-32. That brackets the documented range.
# --------------------------------------------------------------------------

ADP_SIGMA_BASE = 4.0
ADP_SIGMA_RATE = 0.10
# Deterministic tie-break folded into the ADP sort key, spread across the pool as
# `ADP_TIEBREAK_SPAN * rank / worst_rank`. ESPN stops reporting ADP around 171 and
# dozens of players pile up on that plateau; without a tie-break they would be
# drafted in whatever order the CSV listed them, so the late rounds would ignore
# projections entirely -- and the late rounds are where a plan's depth is decided.
#
# Expressed as a total span rather than as a per-rank increment so that it cannot
# grow with the pool. A flat increment per rank was the first attempt and it was
# wrong: at 1e-3 per rank the deepest players picked up a nudge of nearly half a
# pick, which is more than the 0.1 that separates two genuinely different reported
# ADPs. Scaling by the worst rank bounds the whole nudge at this span no matter how
# deep the pool goes.
ADP_TIEBREAK_SPAN = 0.05


# --------------------------------------------------------------------------
# Bench discounting.
#
# Value over replacement answers "how much better than a startable player is he",
# which is the right question for a pick that will start and the wrong one for a
# pick that will not. Undiscounted, VOR happily recommends a second quarterback in
# round 11 -- the twentieth-best quarterback still clears quarterback replacement,
# and the arithmetic does not know that only one of them can start.
#
# So a pick is weighted by how deep on the bench it lands: full credit for a
# starter, BENCH_DECAY for the first backup at a position, its square for the
# second. The weight multiplies projected points and replacement level is then
# subtracted, rather than the whole difference being scaled -- scaling the
# difference would shrink a *negative* value toward zero and make hoarding look
# better than filling a hole, which is the opposite of the intent.
#
# 0.35 is roughly the share of weeks a first backup actually starts once byes and a
# position's injury rate are accounted for. It is not fitted; it is a plausible
# number chosen to be small enough to stop hoarding and large enough that a very
# good backup still beats a bad starter.
#
# With injuries off it is, if anything, generous: a backup's only route into the
# lineup is now the starter's bye week, which is one week in seventeen. It is left
# at 0.35 rather than dropped to 1/17 because the same constant governs how a
# drafter *values* depth, and a drafter who valued his bench at a twentieth of a
# starter would leave real upside on the table. The gap between the two readings is
# a modelling choice worth knowing about.
# --------------------------------------------------------------------------

BENCH_DECAY = 0.35


# --------------------------------------------------------------------------
# The waiver wire.
#
# When nothing on a roster can fill a starting slot -- a singleton starter's bye
# week -- the slot is streamed rather than forfeited. The streamer is the
# FREE_AGENT_DEPTH-th best *undrafted* player at that position in that draft.
#
# This exists because forfeiting was a real, load-bearing bug. Scoring an unfillable
# slot as zero meant the only way a draft could cover its quarterback's bye was to
# spend a pick on a backup, and the search duly recommended a bench quarterback in
# round 8 ahead of a second starting receiver. The tell was unmistakable: the winning
# plans took exactly two quarterbacks and two tight ends, which are exactly the two
# positions you start one of. That is not a strategy, it is bye insurance nobody buys
# with a round-8 pick -- they stream one arm for one week.
#
# Depth rather than a discount multiplier, because depth is a thing you can point at
# on the board. Six is roughly "you do not get first call on the wire": a dozen teams
# pick over the same names, the obvious ones go, and what you actually land is a few
# rows down. It is the one invented number here and it is the right kind of invented
# -- move it and you move the model from "streaming is free" to "streaming is
# hopeless", which brackets the truth.
#
# Note what falls out of this rather than being asserted: the wire is genuinely
# startable at quarterback (only ~13 go, so the sixth undrafted one is close to
# replacement) and genuinely useless at receiver (~60 go, so the sixth undrafted one
# is a deep-bench body). So the model now permits streaming a quarterback, a tight
# end, a kicker and a defense, and refuses to let anyone stream a second receiver --
# which is exactly the real asymmetry, arrived at from the pool rather than typed in.
#
# The wire never covers a slot the bench could have covered; it is the last resort,
# after all sixteen rostered players have been considered. And a rostered starter is
# still worth far more than the wire, because he plays the other thirteen weeks --
# which is why this does not collapse into "never draft a quarterback".
# --------------------------------------------------------------------------

FREE_AGENTS_DEFAULT = True
FREE_AGENT_DEPTH = 6

# Price picks against the wire rather than against "the last player at this position
# who starts somewhere in the league".
#
# Those are two different definitions of replacement and only one can price a draft
# pick. The board's definition -- draft_tiers.starter_depths -- is right for comparing
# positions on a cheat sheet. The marginal value of a *pick* is measured against what
# you can get for nothing, which is the wire, and on this data the two diverge wildly:
#
#     pos   drafted   board repl   wire    ratio
#     QB      14        15.5       14.7    0.95
#     RB      58        10.4        3.3    0.32
#     WR      60        10.4        7.3    0.70
#     TE      36        10.4        4.5    0.43
#
# Fifty-eight running backs get drafted, so RB29 at 10.4 PPG is nothing like freely
# available. Pricing against it made every drafter undervalue back, receiver and tight
# end depth by up to a factor of three -- and that drafter is the baseline, so the
# baseline was a mispriced opponent and every reported edge was inflated by an unknown
# amount.
#
# The wire level depends on how everyone drafts, which depends on the replacement
# level, so it is found by fixed point: draft with the board's levels, measure what is
# left on the wire, re-draft pricing against that, measure again. Two passes is
# plenty; see search.calibrate_replacement.
REPLACEMENT_FROM_WIRE = True
REPLACEMENT_CALIBRATION_DRAFTS = 400
# Four, because it is measurably converged there and not before: from the board's
# levels the passes move 7.05, 1.97, 0.12, 0.01 and then nothing. Two passes lands
# within 0.1 of the fixed point, which is close enough to be tempting and not
# close enough to be provable; the extra two cost about five seconds.
REPLACEMENT_CALIBRATION_PASSES = 4


# --------------------------------------------------------------------------
# The plan space.
#
# A plan names a position for each of the first PLAN_ROUNDS rounds. Eight rounds
# because that is where the strategy is: by round 9 the board is thin enough that
# "best value available" is not just a reasonable policy, it is very close to the
# only one. Extending the plan deeper would multiply the search space by four per
# round to decide questions the data cannot answer.
#
# The caps and floors cut plans nobody would run. Two quarterbacks in the first
# eight rounds is already an aggressive reading of the format and three is not a
# strategy; the RB and WR floors keep the space from filling with rosters that
# cannot field a lineup by round 8. Note the floor of one RB still admits the
# strategy people mean by "zero RB", which is no running back through round 4 or
# 5, not no running back at all.
# --------------------------------------------------------------------------

PLAN_ROUNDS = 8
PLAN_MAX = {QB: 2, TE: 2}
PLAN_MIN = {RB: 1, WR: 1}


# --------------------------------------------------------------------------
# Search budget defaults. Overridable from the command line; these are the
# numbers a full run uses.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Budget:
    """How many drafts to spend, and on what. Three stages, narrowing.

    stage1 explores: random legal plans, enough to measure which *regions* of the
    plan space are good. It cannot rank individual plans -- 12,000 drafts over
    25,174 legal plans is half a sample each.

    stage2 screens: every candidate gets enough drafts to be ranked roughly. At
    1,200 the 95% half-width is about 2.8 points, which is enough to sort a
    two-dozen list into a plausible order and nothing like enough to name a winner.

    stage3 decides: the ``finalists`` best plans from stage 2 are re-measured with
    real precision. At 5,000 drafts the half-width is about 1.0 point.

    The third stage exists because the second one was being over-read. Differences
    between the leading plans are one to three points and stage 2's interval is
    wider than that, so a table naming a single best plan per slot was reporting
    noise as a decision. Stage 3 narrows the interval on the plans that matter, and
    the report then names the whole statistically tied group rather than pretending
    to a winner it cannot see.
    """

    stage1: int = 10_000
    stage2: int = 1_200
    # Screening budget for the mutation neighbourhood. Smaller than stage 2's,
    # because these plans only need ranking against each other and the leaders they
    # were derived from, not resolving.
    stage2b: int = 600
    stage3: int = 5_000
    candidates: int = 24
    # How many screen leaders get their single-round neighbourhood explored. Two,
    # because the leader is itself only a screen result and the runner-up is often a
    # different shape worth walking out from.
    mutate_top: int = 2
    # How many times the neighbourhood is walked. One pass finds plans a single swap
    # from a screen leader; the plan that exposed the nomination bug was two swaps
    # away, so one pass is demonstrably not enough. Later passes start from whatever
    # the previous pass promoted, so the cost per pass falls as it converges.
    mutate_passes: int = 2
    finalists: int = 8
    # A plan needs this many stage-1 samples before its empirical mean is allowed
    # to nominate it. Without a floor the top of the stage-1 list is whichever
    # plan got two lucky drafts.
    min_stage1_samples: int = 8
    # 0 = no cap. The skill pool must include the undrafted tail, because that tail
    # IS the waiver wire; capping it silently emptied the tight end wire.
    pool_skill: int = 0
    seed: int = 20260819

    def __post_init__(self):
        if self.finalists > self.candidates:
            raise ValueError(f"cannot promote {self.finalists} finalists out of "
                             f"{self.candidates} candidates")


FULL_BUDGET = Budget()
# --quick: enough to prove the pipeline runs end to end and produce a report with
# the right shape, not enough to believe any number in it.
QUICK_BUDGET = Budget(stage1=400, stage2=120, stage2b=80, stage3=300, candidates=8,
                      mutate_top=1, mutate_passes=1, finalists=4,
                      min_stage1_samples=2)


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

# Metrics the search can rank plans by. Playoff rate is the default: it is the
# outcome the season is actually played for, and at roughly 50% base rate it is
# the one a fixed number of drafts measures most precisely. Title rate is the
# same question with a sixth of the signal.
METRICS = ("playoffs", "title", "points")
DEFAULT_METRIC = "playoffs"

# Confidence level for the interval printed beside every rate, and for the test that
# decides whether two plans were actually distinguished.
#
# Note which test that is. Two plans are called indistinguishable when a confidence
# interval on the *difference* of their rates contains zero -- not when their
# individual intervals overlap. Overlap is the wrong test and everybody reaches for
# it anyway: it is far too conservative, behaving like a test at roughly the 0.83
# level rather than 0.05, so it declares ties between plans that are in fact
# separated. The difference interval is the honest one.
CONFIDENCE_Z = 1.96


@dataclass
class SimConfig:
    """Everything one run of the search needs, in one object.

    Passed whole to worker processes, so it must stay picklable: plain numbers,
    strings, tuples and frozen dataclasses only.
    """

    league: League = DEFAULT_LEAGUE
    budget: Budget = FULL_BUDGET
    metric: str = DEFAULT_METRIC
    slots: tuple = ()
    adp_model: str = "premium"
    te_premium: bool = True
    jobs: int = 0          # 0 -> pick from cpu_count
    plan_rounds: int = PLAN_ROUNDS
    room: str = DEFAULT_ROOM
    perception_sigma: float = PERCEPTION_SIGMA
    injuries: bool = INJURIES_DEFAULT
    weekly_cv: tuple = WEEKLY_CV
    miss_mean_weeks: float = MISS_MEAN_WEEKS
    adp_sigma_base: float = ADP_SIGMA_BASE
    adp_sigma_rate: float = ADP_SIGMA_RATE
    bench_decay: float = BENCH_DECAY
    free_agents: bool = FREE_AGENTS_DEFAULT
    free_agent_depth: int = FREE_AGENT_DEPTH
    replacement_from_wire: bool = REPLACEMENT_FROM_WIRE
    # Filled in by search.calibrate_replacement; empty means "use the pool's".
    replacement: tuple = ()
    projection_error: bool = PROJECTION_ERROR_DEFAULT
    projection_sigma: tuple = PROJECTION_SIGMA
    # Drafts spent measuring the baseline -- what a drafter with no plan gets from
    # each slot, which is the yardstick every plan's number is read against. One
    # batch measures all twelve slots at once, because a room of twelve identical
    # drafters IS twelve baseline samples.
    #
    # In the `value` room the baseline drafter is straight best-available-value, so
    # the edge reported for a plan is what the *plan* is worth over simply taking
    # value -- which is the comparison the study is for. In the `adp` room it is an
    # ADP drafter, and the edge then conflates plan choice with the much larger
    # effect of using projections at all.
    baseline: int = 4_000

    @property
    def miss_any_prob(self):
        """Injury rates actually in force. All zero if injuries are switched off."""
        return INJURY_MISS_ANY_PROB if self.injuries else NO_INJURIES

    @property
    def proj_sigma(self):
        """Projection-error scale in force. All zero if the error is switched off."""
        return self.projection_sigma if self.projection_error else (0.0,) * len(POSITIONS)

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        if self.adp_model not in ("premium", "espn"):
            raise ValueError("adp_model must be 'premium' or 'espn'")
        if self.room not in ROOMS:
            raise ValueError(f"room must be one of {ROOMS}, got {self.room!r}")
        if self.perception_sigma < 0:
            raise ValueError("perception_sigma cannot be negative")
        if not self.slots:
            self.slots = tuple(range(1, self.league.teams + 1))
        bad = [s for s in self.slots if not 1 <= s <= self.league.teams]
        if bad:
            raise ValueError(f"slots outside 1..{self.league.teams}: {bad}")
        if self.plan_rounds > self.league.rounds - len(LATE_ONLY):
            raise ValueError("the plan cannot reach into the K/DST rounds")
