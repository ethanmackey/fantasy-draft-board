"""The snake draft itself: who picks when, and what each of them takes.

This is the only module that knows the order of picks. It owns the pick policies
and one piece of state, the board, kept as a "taken" mask over the pool.

Every seat runs the same rule in the default `value` room -- take the player whose
bench-discounted value over replacement is highest, as *you* see him -- so the
policy is written once and your seat differs only in being tied to a plan for the
opening rounds. That symmetry is the point: it is what makes the baseline
("nobody follows a plan") a fair yardstick, and it is what stops a plan's measured
edge from being an information advantage in disguise.

One pass per pick does the whole decision. For every player, score

    bench_weight[his position] * perceived_ppg[this drafter][him] - replacement[his position]

mask out whoever is taken or illegal, take the argmax. Positions do not need to be
considered separately, because the weight and the replacement level are constants
within a position, so a single vector comparison ranks the entire board at once.

An earlier version kept a per-position pointer into a PPG-sorted index list, which
was cheaper still -- amortised O(1) for "best available running back". That trick
dies the moment drafters disagree about PPG: there is no longer one ordering to
point into. The masked-argmax pass costs about what the old ADP argmin cost, and it
replaces two things (the pointers and the 192 x 324 ADP noise matrix) with one.
"""

import numpy as np

from . import bots
from .config import DEFAULT_LEAGUE, LATE_ONLY, POSITIONS
from .roster import allowed_positions, bench_depth


def snake_order(teams, rounds):
    """Team index (0-based) for each overall pick, odd rounds up, even rounds down."""
    order = np.empty(teams * rounds, dtype=np.int16)
    for r in range(rounds):
        row = np.arange(teams) if r % 2 == 0 else np.arange(teams - 1, -1, -1)
        order[r * teams:(r + 1) * teams] = row
    return order


def slot_picks(slot, teams, rounds):
    """1-based overall pick numbers belonging to a 1-based draft slot.

    The snake's whole character in one line: slot 1 owns picks 1 and 24, slot 12
    owns 12 and 13. Used by the report, and by the tests that pin the order down.
    """
    out = []
    for r in range(rounds):
        out.append(r * teams + (slot if r % 2 == 0 else teams - slot + 1))
    return out


class DraftEngine:
    """Runs drafts over one pool. Reusable, and reused: buffers are allocated once.

    One engine per worker process. It holds no per-draft state between calls --
    ``run`` resets everything it touches -- but it does hold the scratch arrays,
    which is the point: a draft is a few hundred small numpy operations, and at a
    million drafts allocation would be a visible fraction of the runtime.
    """

    def __init__(self, pool, league=DEFAULT_LEAGUE, sigma_base=None, sigma_rate=None,
                 bench_decay=None, room=None, perception_sigma=None,
                 free_agent_depth=None, replacement=None):
        from .config import (ADP_SIGMA_BASE, ADP_SIGMA_RATE, BENCH_DECAY,
                             DEFAULT_ROOM, FREE_AGENT_DEPTH, PERCEPTION_SIGMA, ROOMS)
        self.pool = pool
        self.league = league
        self.room = DEFAULT_ROOM if room is None else room
        if self.room not in ROOMS:
            raise ValueError(f"room must be one of {ROOMS}, got {self.room!r}")
        self.sigma_base = ADP_SIGMA_BASE if sigma_base is None else sigma_base
        self.sigma_rate = ADP_SIGMA_RATE if sigma_rate is None else sigma_rate
        self.bench_decay = BENCH_DECAY if bench_decay is None else bench_decay
        self.perception_sigma = (PERCEPTION_SIGMA if perception_sigma is None
                                 else perception_sigma)
        self.free_agent_depth = (FREE_AGENT_DEPTH if free_agent_depth is None
                                 else free_agent_depth)
        self.order = snake_order(league.teams, league.rounds)

        n = pool.size
        self._taken = np.zeros(n, dtype=bool)
        self._work = np.empty(n, dtype=np.float64)
        self._allowed = np.empty(len(POSITIONS), dtype=bool)
        self._weight = np.empty(len(POSITIONS), dtype=np.float64)
        self._counts = np.zeros((league.teams, len(POSITIONS)), dtype=np.int16)
        self._rosters = np.zeros((league.teams, league.rounds), dtype=np.int32)
        # Position code per player, as int64 so `allowed[pos_of]` and
        # `weight[pos_of]` are single fancy-index gathers rather than Python loops.
        self._pos_of = pool.pos.astype(np.int64)
        # Replacement level per team per player, precomputed: the subtrahend of every
        # value comparison. Per *team* so a room can be mixed -- one drafter pricing
        # against the waiver wire while eleven price against the last starter in the
        # league, say -- which is the only way to find out which definition actually
        # drafts better rather than arguing about it. Pass a (positions,) vector for a
        # uniform room, a (teams, positions) one for a mixed room, or nothing for the
        # pool's own levels.
        levels = pool.replacement if replacement is None else np.asarray(
            replacement, dtype=np.float64)
        if levels.ndim == 1:
            levels = np.broadcast_to(levels, (league.teams, len(POSITIONS)))
        if levels.shape != (league.teams, len(POSITIONS)):
            raise ValueError(f"replacement must be ({len(POSITIONS)},) or "
                             f"({league.teams}, {len(POSITIONS)}), got {levels.shape}")
        self.replacement = np.array(levels, dtype=np.float64)
        self._replacement_of = self.replacement[:, self._pos_of]

    # ---- the value policy -----------------------------------------------

    def _value_choice(self, perceived, counts, allowed, replacement_of,
                      restrict_to=None):
        """Index of the player this drafter takes, or -1 if nothing is legal.

        ``perceived`` is this drafter's own row of the perception matrix.
        ``restrict_to`` limits the choice to one position, which is how a plan is
        followed; without it the whole legal board is considered.

        The score is

            decay^depth * perceived_ppg - replacement[pos]

        A startable pick is depth 0 and scores exact value over replacement. A
        backup's perceived points are shrunk *before* replacement is subtracted,
        which is the arrangement that matters: scaling the finished difference
        instead would pull a negative value toward zero and so make stacking a bench
        look better than filling a hole. Without any discount this rule takes a
        second quarterback in the middle rounds -- the twentieth-best quarterback
        clears quarterback replacement, and nothing in the undiscounted arithmetic
        knows he will never start.
        """
        weight = self._weight
        for pos in range(len(POSITIONS)):
            if allowed[pos]:
                weight[pos] = self.bench_decay ** bench_depth(counts, pos, self.league)
            else:
                # Never read: these players are masked out below. Set anyway so the
                # gather cannot pick up a stale value from a previous pick.
                weight[pos] = 0.0

        work = self._work
        np.multiply(weight[self._pos_of], perceived, out=work)
        np.subtract(work, replacement_of, out=work)
        np.copyto(work, -np.inf, where=self._taken)
        if restrict_to is None:
            np.copyto(work, -np.inf, where=~allowed[self._pos_of])
        else:
            np.copyto(work, -np.inf, where=self._pos_of != restrict_to)
        best = int(work.argmax())
        return best if work[best] > -np.inf else -1

    # ---- the draft ------------------------------------------------------

    def run(self, slot, plan, rng, out=None):
        """Draft once. Returns (teams, rounds) of pool indices; rosters[slot-1] is you.

        ``slot`` is 1-based; ``slot=0`` means nobody follows a plan and all twelve
        seats are ordinary drafters. That mode is both the baseline the report
        measures every plan against and the simulator's calibration check: twelve
        identical drafters in a twelve-team league must produce a 50% playoff rate
        overall, and any deviation is a bug in the schedule, the standings or the
        bracket rather than a finding about draft position.

        ``plan`` names a position for each of its own length's worth of opening
        rounds; rounds past it are value fill, and the last two are the kicker and
        the defense.

        The plan is a target, not a promise. If the named position is closed -- at
        its cap, or shut off by the starter-completion guard, or simply exhausted --
        the pick falls back to best value available rather than being skipped. A plan
        that gets overridden often is a plan the report should show performing badly,
        not one the engine should refuse to run.
        """
        pool, league = self.pool, self.league
        taken, counts = self._taken, self._counts
        rosters = self._rosters if out is None else out
        allowed, work = self._allowed, self._work

        taken.fill(False)
        counts.fill(0)
        hero = slot - 1

        perceived = bots.perceived_ppg(rng, pool.ppg, league.teams,
                                       self.perception_sigma)
        noise = None
        if self.room == "adp":
            noise = bots.noise_matrix(rng, league.picks, pool.size,
                                      self.sigma_base, self.sigma_rate)
        picks_made = np.zeros(league.teams, dtype=np.int16)

        for pick, team in enumerate(self.order):
            team = int(team)
            made = int(picks_made[team])
            allowed_positions(counts[team], made, league, out=allowed)

            if team == hero or noise is None:
                wanted = None
                if team == hero and made < len(plan) and allowed[plan[made]]:
                    wanted = plan[made]
                choice = self._value_choice(perceived[team], counts[team], allowed,
                                            self._replacement_of[team],
                                            restrict_to=wanted)
                if choice < 0 and wanted is not None:
                    # The planned position is legal but empty. Fall back rather than
                    # pass; a plan the board cannot honour should show up as a plan
                    # that performs badly.
                    choice = self._value_choice(perceived[team], counts[team], allowed,
                                                self._replacement_of[team])
                if choice < 0:
                    # Nothing legal is left anywhere. Cannot happen with a pool deep
                    # enough for the league -- League.validate and load_pool between
                    # them rule it out -- so it means a hand-built pool, and it is
                    # worth a loud failure rather than a -1 quietly entering a roster.
                    raise RuntimeError(
                        f"pick {pick + 1}: no legal player available for team {team}; "
                        f"pool of {pool.size} is too thin for this league")
            else:
                choice = bots.bot_choice(pool.adp_key, noise[pick], taken,
                                         allowed[self._pos_of], work)

            taken[choice] = True
            rosters[team, made] = choice
            counts[team, pool.pos[choice]] += 1
            picks_made[team] = made + 1

        return rosters


    def free_agents(self, depth=None):
        """(positions,) projected PPG of the streamer available at each position.

        The ``depth``-th best *undrafted* player at that position, from the board this
        draft actually left behind. Must be called before the next ``run``, which
        resets the taken mask.

        Depth rather than the single best undrafted name, because a dozen teams pick
        over the same wire and the obvious claims go. What matters is that the level
        is read off the pool instead of typed in: only about thirteen quarterbacks get
        drafted, so the sixth undrafted one is close to replacement and genuinely
        startable for a week, while sixty receivers go, so the sixth undrafted one is
        a deep-bench body. The model therefore lets a manager stream a quarterback and
        refuses to let him stream a second receiver, which is the real asymmetry.

        A position with nothing left returns 0.0 -- it cannot happen for K or DST, and
        for the skill positions the pool is far deeper than the draft.
        """
        depth = self.free_agent_depth if depth is None else depth
        out = np.zeros(len(POSITIONS), dtype=np.float64)
        for pos in range(len(POSITIONS)):
            column = self.pool.by_pos[pos]          # already sorted by PPG desc
            seen = 0
            for index in column:
                if self._taken[index]:
                    continue
                seen += 1
                if seen == depth:
                    out[pos] = self.pool.ppg[index]
                    break
            else:
                # Fewer than `depth` left: take the worst still available, or nothing.
                spare = [i for i in column if not self._taken[i]]
                out[pos] = float(self.pool.ppg[spare[-1]]) if spare else 0.0
        return out


def draft_once(engine, slot, plan, rng, out=None):
    """Convenience wrapper. Exists so callers read as a sentence, not a method call."""
    return engine.run(slot, plan, rng, out=out)
