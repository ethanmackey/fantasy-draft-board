"""Draft plans: the thing being optimised, and the names it goes by.

A **plan** is one position per round for the first ``PLAN_ROUNDS`` rounds -- say
``RB-WR-WR-TE-RB-WR-QB-RB``. It is not a list of players. Which player fills each
slot is decided at the table by who is actually there, which is the only way a
plan can survive contact with eleven opponents; what the plan fixes is the shape
of the roster you are trying to build.

Eight rounds because that is where the strategy is. By round 9 the board is thin
enough that "best value available" is not merely a reasonable policy, it is close
to the only one, and extending the plan would multiply the search space by four
per round to decide questions this data cannot answer.

The legal space is enumerated rather than sampled by rejection. There are 4^8 =
65,536 sequences and roughly a third of them are legal, so enumerating once and
indexing is both faster than rejection sampling and -- more usefully -- it makes
"the best legal plan under these per-round scores" an exact search instead of a
greedy guess that might need repairing.
"""

from functools import lru_cache
from itertools import product

from .config import PLAN_MAX, PLAN_MIN, PLAN_ROUNDS, POSITIONS, QB, RB, SKILL_POSITIONS, TE, WR


def plan_string(plan):
    """``(RB, WR, TE)`` -> ``'RB-WR-TE'``. The form that goes in every table."""
    return "-".join(POSITIONS[p] for p in plan)


def parse_plan(text):
    """``'RB-WR-TE'`` -> ``(RB, WR, TE)``. For --plan on the command line."""
    out = []
    for token in text.replace(",", "-").split("-"):
        token = token.strip().upper()
        if not token:
            continue
        if token == "DEF":
            token = "DST"
        if token not in POSITIONS:
            raise ValueError(f"{token!r} is not a position; expected one of {POSITIONS}")
        out.append(POSITIONS.index(token))
    return tuple(out)


def is_legal(plan, plan_max=None, plan_min=None):
    """Would anyone actually run this plan?

    At most two quarterbacks and two tight ends in the first eight rounds: three
    of either is not a strategy, it is a bug in a random number generator. At
    least one running back and one receiver, so the space does not fill with
    rosters that cannot field a lineup by round 8.

    Note what the floor of one running back still admits: no RB until round 8 is
    legal here. "Zero RB" in the sense anyone means it -- no running back through
    round 4 or 5, then two in a row -- is well inside the space.
    """
    plan_max = PLAN_MAX if plan_max is None else plan_max
    plan_min = PLAN_MIN if plan_min is None else plan_min
    counts = {}
    for pos in plan:
        counts[pos] = counts.get(pos, 0) + 1
    for pos, cap in plan_max.items():
        if counts.get(pos, 0) > cap:
            return False
    for pos, floor in plan_min.items():
        if counts.get(pos, 0) < floor:
            return False
    return True


@lru_cache(maxsize=8)
def legal_plans(rounds=PLAN_ROUNDS):
    """Every legal plan of this length, as a tuple of tuples. Cached.

    Cached because it is built once and then indexed a few hundred thousand times
    across a run, and because every worker process wants the same list.
    """
    return tuple(plan for plan in product(SKILL_POSITIONS, repeat=rounds)
                 if is_legal(plan))


def sample_plan(rng, rounds=PLAN_ROUNDS):
    """A uniformly random legal plan.

    Uniform over *legal* plans rather than over sequences, which matters for the
    marginals stage 1 reports: rejection sampling would leave the per-round
    position counts unbalanced in a way that has nothing to do with the data.
    """
    plans = legal_plans(rounds)
    return plans[int(rng.integers(len(plans)))]


def top_plans_by_round_scores(round_scores, k, rounds=PLAN_ROUNDS):
    """The k legal plans maximising a sum of per-round position scores.

    ``round_scores`` is indexed ``[round][position]`` -- typically stage 1's
    playoff rate conditioned on taking that position in that round.

    Exact, not greedy. Greedily taking the best position in each round produces an
    illegal plan more often than not (the best marginal in seven of eight rounds is
    frequently the same position) and repairing an illegal plan means inventing a
    rule about which round to sacrifice. Enumerating the legal space and scoring it
    is a hundred thousand additions -- cheaper than the argument.

    The additive score is a real assumption: it treats rounds as independent, which
    they are not. It is used only to *nominate* candidates, and stage 2 then
    measures each nominee properly, so a wrong nomination costs simulations and
    never corrupts a result.
    """
    plans = legal_plans(rounds)
    scored = sorted(plans, key=lambda p: -sum(round_scores[r][pos]
                                              for r, pos in enumerate(p)))
    return scored[:k]


def mutations(plan, rounds=None):
    """Every legal plan one round different from this one.

    The neighbourhood a local search walks. It exists because the nomination score --
    a sum of per-round marginals -- is additive over rounds and therefore blind to
    interactions between them, and it demonstrably missed a better plan: at slot 1 the
    plan RB-RB-WR-TE-RB-WR-WR-QB measured 2.2 points of playoff rate above the screen
    leader and was never nominated at all, because the marginals liked a quarterback in
    every late round.

    Walking out one swap at a time from a plan that already screened well finds those,
    and it costs nothing to reason about: eight rounds by three alternatives is at most
    24 neighbours, most of them legal.
    """
    plan = tuple(plan)
    rounds = len(plan) if rounds is None else rounds
    out = []
    for i in range(min(rounds, len(plan))):
        for pos in SKILL_POSITIONS:
            if pos == plan[i]:
                continue
            candidate = plan[:i] + (pos,) + plan[i + 1:]
            if is_legal(candidate):
                out.append(candidate)
    return out


# --------------------------------------------------------------------------
# Named archetypes.
#
# These are reference rows, not candidates in the ordinary sense: they go into
# stage 2 whether or not stage 1 liked them, so that every report shows what the
# strategies people actually talk about are worth from that slot. A table whose
# top line is a plan nobody has a name for is much more convincing next to a
# measured Zero-RB than on its own.
# --------------------------------------------------------------------------

ARCHETYPES = {
    "Robust-RB":  (RB, RB, WR, WR, TE, WR, QB, RB),
    "Zero-RB":    (WR, WR, WR, TE, RB, RB, QB, WR),
    "Hero-RB":    (RB, WR, WR, WR, RB, TE, QB, RB),
    "Elite-TE":   (TE, RB, WR, WR, RB, WR, QB, RB),
    "Early-QB":   (RB, WR, QB, WR, RB, TE, WR, RB),
    "Balanced":   (RB, WR, RB, WR, TE, WR, QB, RB),
    "WR-heavy":   (WR, WR, RB, WR, WR, TE, QB, RB),
    "Double-TE":  (TE, WR, RB, TE, WR, RB, QB, WR),
}


def archetype_plans(rounds=PLAN_ROUNDS):
    """The named plans, trimmed or padded to ``rounds`` and filtered to legal ones.

    Trimming can make a plan illegal -- ``Zero-RB`` cut to five rounds has no
    running back in it -- so the result is filtered rather than assumed. Padding
    repeats the last position, which is only reached if someone raises
    ``--plan-rounds`` past eight and is a placeholder, not a recommendation.
    """
    out = {}
    for label, plan in ARCHETYPES.items():
        if rounds <= len(plan):
            trimmed = plan[:rounds]
        else:
            trimmed = plan + (plan[-1],) * (rounds - len(plan))
        if is_legal(trimmed):
            out[label] = trimmed
    return out


# --------------------------------------------------------------------------
# Labelling.
#
# Rules over the plan, composed with " / " so a plan can be more than one thing at
# once -- which the good ones usually are. The point of a label is that the reader
# recognises the shape without decoding eight position codes, so a plan matching
# nothing gets its shape summarised by position counts instead of a made-up name.
# --------------------------------------------------------------------------

def _first(plan, pos):
    """1-based round of the first pick at this position, or None."""
    for i, p in enumerate(plan):
        if p == pos:
            return i + 1
    return None


def label_plan(plan):
    """A readable archetype label for a plan, e.g. 'Hero-RB / Elite-TE / Late-QB'."""
    exact = {tuple(v): k for k, v in ARCHETYPES.items()}
    if tuple(plan) in exact:
        return exact[tuple(plan)]

    tags = []
    rb_rounds = [i + 1 for i, p in enumerate(plan) if p == RB]
    wr_count = sum(1 for p in plan if p == WR)
    te_rounds = [i + 1 for i, p in enumerate(plan) if p == TE]
    qb_round = _first(plan, QB)

    early_rb = [r for r in rb_rounds if r <= 4]
    if not early_rb:
        tags.append("Zero-RB")
    elif len(early_rb) == 1 and early_rb[0] <= 2 and not any(3 <= r <= 4 for r in rb_rounds):
        tags.append("Hero-RB")
    elif sum(1 for r in rb_rounds if r <= 3) >= 2:
        tags.append("Robust-RB")

    if te_rounds and te_rounds[0] <= 3:
        tags.append("Elite-TE")
    if len(te_rounds) >= 2:
        tags.append("Double-TE")

    if qb_round is None:
        tags.append("No-QB")
    elif qb_round <= 4:
        tags.append("Early-QB")
    elif qb_round >= len(plan) - 1:
        tags.append("Late-QB")

    if wr_count >= len(plan) // 2:
        tags.append("WR-heavy")

    if not tags:
        counts = {POSITIONS[p]: sum(1 for q in plan if q == p) for p in SKILL_POSITIONS}
        tags.append(" ".join(f"{n}{pos}" for pos, n in counts.items() if n))
    return " / ".join(tags)
