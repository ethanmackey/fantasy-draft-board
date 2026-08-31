"""One season out of one draft: seventeen weeks, a schedule, and a bracket.

This is the only module that knows what a week is. It takes twelve finished
rosters and returns what happened to the user's, which is the number the whole
search is built to compare.

One season per draft, not many. Averaging noise by replaying the same draft would
spend the budget learning the variance of a roster the user will never own; the
same simulations spent on fresh drafts learn how a *plan* does against fresh
opponents, which is the question. The cost is that a single row of results is
almost pure noise, and the honest response to that is the confidence interval the
report prints rather than a quieter-looking number.

Three modelling choices carry most of the weight, all of them documented where
they happen: lineups are set from projections and scored on realised points;
missed games come in contiguous blocks; and the schedule is reshuffled every draft
so no slot inherits a permanent opponent.
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_LEAGUE, MIN_PPG, MISS_MEAN_WEEKS, WEEKLY_CV
from .roster import lineup_scores


@dataclass(frozen=True)
class SeasonResult:
    """What happened to the user's team. One row of the search's data."""

    made_playoffs: bool
    won_title: bool
    wins: float
    points: float


def round_robin(teams):
    """(rounds, teams) of opponent positions -- the circle method.

    ``teams`` even: rounds = teams - 1 and every position meets every other
    exactly once. Positions, not teams: the mapping from a draft slot to a
    schedule position is redrawn for every simulated season, so this is computed
    once and reused for the life of the process.
    """
    if teams % 2:
        raise ValueError("round_robin needs an even number of teams")
    rounds = teams - 1
    fixed = teams - 1
    rotating = list(range(teams - 1))
    out = np.empty((rounds, teams), dtype=np.int16)
    for r in range(rounds):
        pairs = [(fixed, rotating[r % fixed])]
        for i in range(1, teams // 2):
            a = rotating[(r + i) % fixed]
            b = rotating[(r - i) % fixed]
            pairs.append((a, b))
        for a, b in pairs:
            out[r, a] = b
            out[r, b] = a
        # Rotate by keeping `fixed` still, which the modular indexing above
        # already does; nothing to mutate.
    return out


def schedule_positions(teams, weeks):
    """(weeks, teams) opponent positions, a round robin extended to ``weeks``.

    A 12-team round robin is 11 weeks and the regular season is 14, so the last
    three weeks repeat the first three. Repeating rather than inventing keeps the
    schedule balanced: every position still plays every other either once or
    twice, and nobody draws the same opponent three times.
    """
    base = round_robin(teams)
    rounds = base.shape[0]
    if weeks <= rounds:
        return base[:weeks].copy()
    extra = weeks - rounds
    if extra > rounds:
        raise ValueError(f"{weeks} weeks needs more than a double round robin "
                         f"for {teams} teams")
    return np.vstack([base, base[:extra]])


class SeasonSimulator:
    """Plays seasons for one pool and one league. One per worker process.

    Holds the per-player distribution parameters and the schedule, both of which
    are functions of the pool and the league rather than of any particular draft,
    so they are computed once here instead of half a million times in the loop.
    """

    def __init__(self, pool, league=DEFAULT_LEAGUE, weekly_cv=WEEKLY_CV,
                 miss_any_prob=None, miss_mean_weeks=MISS_MEAN_WEEKS,
                 projection_sigma=None):
        from .config import (NO_INJURIES, PROJECTION_SIGMA, PROJECTION_SIGMA_DEPTH,
                             PROJECTION_SIGMA_TAIL)
        self.pool = pool
        self.league = league
        self.weeks = league.total_weeks
        self.schedule = schedule_positions(league.teams, league.regular_weeks)

        cv = np.asarray(weekly_cv, dtype=np.float64)[pool.pos]
        mean = np.maximum(pool.ppg, MIN_PPG)
        # Gamma parameterised by mean and CV: shape 1/cv^2, scale mean*cv^2 has
        # exactly that mean and that CV. Gamma rather than normal because a weekly
        # fantasy score cannot be negative and its right tail is long -- a normal at
        # RB's CV of 0.55 would put a fifth of a 12-PPG back's weeks below zero.
        #
        # Note the scale is per *unit of ability*: the season multiplies it by each
        # player's true ability draw, so a bust's weekly variance shrinks with him
        # rather than staying pinned to his projection.
        self.gamma_shape = 1.0 / (cv ** 2)
        self.gamma_unit_scale = (cv ** 2)
        self.projected = mean
        # Kept for callers that want the projection-only behaviour.
        self.gamma_scale = mean * (cv ** 2)

        # Projection error, per player, rising with overall rank: a first-round pick is
        # far more predictable than a tenth-round one.
        sigma = np.asarray(PROJECTION_SIGMA if projection_sigma is None
                           else projection_sigma, dtype=np.float64)[pool.pos]
        depth = np.minimum(pool.rank.astype(np.float64), PROJECTION_SIGMA_DEPTH)
        self.proj_sigma = sigma * (1.0 + PROJECTION_SIGMA_TAIL
                                   * depth / PROJECTION_SIGMA_DEPTH)
        self.projection_error_on = bool(self.proj_sigma.any())
        # Per position rather than per player, for the waiver wire: a streamer is
        # identified by his position and his level, not by a pool index.
        self.cv_by_pos = np.asarray(weekly_cv, dtype=np.float64)
        # Injuries default OFF -- see config. Passing no rates means nobody is ever
        # hurt, and `injured` below skips its sampling entirely rather than drawing
        # random numbers it will multiply by zero.
        miss = NO_INJURIES if miss_any_prob is None else miss_any_prob
        self.miss_prob = np.asarray(miss, dtype=np.float64)[pool.pos]
        self.injuries_on = bool(self.miss_prob.any())
        self.miss_p_geom = 1.0 / max(miss_mean_weeks, 1.0)
        self.week_index = np.arange(self.weeks, dtype=np.int16)

    # ---- true ability ---------------------------------------------------

    def true_ppg(self, rng):
        """(players,) what each player is actually worth this season.

        A mean-preserving lognormal around the projection, drawn once per season, and
        visible to nobody: the draft is made on the projection (perceived through each
        drafter's own error) and the season is played on this.

        Without it the season was scored off the very numbers the draft optimised
        against, so any bias in the source became risk-free profit and reaching for a
        projection outlier was free. That single omission was worth most of the
        implausibly large edges the first version reported.

        Mean-preserving matters for the same reason it does in the perception model:
        an unshifted exp(Normal(0, s)) has mean exp(s^2/2) > 1, so every player would
        quietly outperform his projection and the whole board would inflate.

        Lognormal, not normal, because busts floor out and breakouts have a long right
        tail. Sigma rises with overall rank -- see PROJECTION_SIGMA_DEPTH -- so the
        error is small at the top of the board and large in the rounds where a
        breakout actually comes from.
        """
        if not self.projection_error_on:
            return self.projected
        s = self.proj_sigma
        return self.projected * np.exp(rng.normal(-0.5 * s * s, s))

    # ---- availability ---------------------------------------------------

    def availability(self, idx, rng):
        """(teams, roster, weeks) of "can this player be started this week".

        With injuries off -- the default -- the bye week is the only way to be
        unavailable, and it is deterministic, straight from the export. The bench is
        then worth exactly what byes make it worth, which is a defensible floor.

        Injuries are off because modelling them without modelling the waiver wire is
        worse than modelling neither. It would force a manager whose back misses
        three weeks to start his fifth-best back, when in reality he adds the
        replacement who just inherited twenty touches -- so bench depth acquires a
        value it does not have, and the arithmetic starts recommending that a backup
        quarterback is worth a pick. He is not, in a one-quarterback league.

        With ``--injuries`` on, a player misses a single contiguous block with
        probability ``INJURY_MISS_ANY_PROB`` for his position, starting at a
        uniformly random week. Contiguous rather than scattered because clustering is
        the entire point: three isolated one-week absences cost a manager nothing,
        and one three-week block in October costs him a starter for a fifth of the
        season.
        """
        shape = idx.shape
        flat = idx.reshape(-1)
        n = flat.size
        weeks = self.week_index[None, :]

        out = np.ones((n, self.weeks), dtype=bool)

        bye = self.pool.bye[flat].astype(np.int16)
        np.logical_and(out, weeks != (bye[:, None] - 1), out=out)

        if self.injuries_on:
            hurt = rng.random(n) < self.miss_prob[flat]
            # Geometric with mean MISS_MEAN_WEEKS and a floor of one week: numpy's
            # geometric counts trials to first success, so it is already >= 1 and its
            # mode is 1, which is what a missed-games distribution looks like.
            length = rng.geometric(self.miss_p_geom, size=n).astype(np.int16)
            start = rng.integers(0, self.weeks, size=n).astype(np.int16)
            block = (weeks >= start[:, None]) & (weeks < (start + length)[:, None])
            np.logical_and(out, ~(block & hurt[:, None]), out=out)

        return out.reshape(*shape, self.weeks)

    # ---- a season -------------------------------------------------------

    def wire(self, free_agents, rng):
        """(teams, positions, weeks) realised streamer points, and their projections.

        One draw per team per position per week, because two managers streaming the
        same week are not getting the same player -- they are getting two names of
        similar quality off the same wire.

        A streamer has no bye week. That is the point of him: a manager picking up an
        arm for one week picks one who is playing. He carries the same weekly variance
        as anyone else at his position.
        """
        teams = self.league.teams
        mean = np.maximum(np.asarray(free_agents, dtype=np.float64), MIN_PPG)
        cv = self.cv_by_pos
        shape = 1.0 / (cv ** 2)
        scale = mean * (cv ** 2)
        # No true-ability draw for a streamer. He is not one player -- he is whoever
        # is available that week -- so his error averages out rather than compounding.
        draws = rng.standard_gamma(shape[None, :, None],
                                   size=(teams, len(mean), self.weeks))
        return draws * scale[None, :, None], np.broadcast_to(
            mean, (teams, len(mean))).copy()

    def simulate(self, rosters, hero, rng, free_agents=None):
        """Play one season. ``hero`` is a 0-based team index. Returns SeasonResult."""
        wins, points, seeds, champion = self.play(rosters, rng, free_agents)
        cut = self.league.playoff_teams
        return SeasonResult(
            made_playoffs=bool(hero in seeds[:cut]),
            won_title=bool(champion == hero),
            wins=float(wins[hero]),
            points=float(points[hero]))

    def play(self, rosters, rng, free_agents=None):
        """One season, from every team's point of view: (wins, points, seeds, champion).

        Split out from ``simulate`` because the whole league's outcome is the natural
        unit -- a season has twelve results in it, not one -- and because the
        calibration check that matters most needs all twelve: put twelve identical
        ADP drafters in a room and every slot must come out near a 50% playoff rate.
        Anything else is a bug in the schedule, the bracket or the standings, and
        without this method that check cannot be written.
        """
        pool, league = self.pool, self.league
        teams, roster_size = rosters.shape

        # What every player is actually worth this season. Nobody saw this at the
        # draft; the draft was made on projections.
        truth = self.true_ppg(rng)

        # Descending *true* ability along the roster axis, because that is the order a
        # manager ends up setting his lineup in. Over fourteen weeks he works out who
        # is good -- he cannot know Sunday's score in advance, which is the rule this
        # respects, but he is not still starting a round-three bust in November either.
        # Ordering on projections instead would model a manager who never notices, and
        # would charge a plan twice for the same draft-day error.
        ability = truth[rosters]
        order = np.argsort(-ability, axis=1, kind="stable")
        idx = np.take_along_axis(rosters, order, axis=1)

        roster_pos = pool.pos[idx].astype(np.int64)
        available = self.availability(idx, rng)

        flat = idx.reshape(-1)
        scale = truth[flat] * self.gamma_unit_scale[flat]
        realised = (rng.standard_gamma(self.gamma_shape[flat][:, None],
                                       size=(flat.size, self.weeks))
                    * scale[:, None])
        realised = realised.reshape(teams, roster_size, self.weeks)
        np.multiply(realised, available, out=realised)

        wire = wire_mean = None
        if free_agents is not None:
            wire, wire_mean = self.wire(free_agents, rng)
        scores = lineup_scores(roster_pos, available, realised, league,
                               wire=wire, wire_mean=wire_mean)

        wins, points = self._standings(scores, rng)
        seeds = self._seed(wins, points)
        champion = self._bracket(seeds, scores)
        return wins, points, seeds, champion

    def _standings(self, scores, rng):
        """Wins and points-for over the regular season.

        The team-to-schedule-position mapping is redrawn every season. Without
        that, slot 1 would play a fixed set of opponents in every simulation and
        any slot effect the report found would be part schedule artefact -- which
        would be a bug in exactly the result the simulator exists to produce.
        """
        league = self.league
        teams = league.teams
        perm = rng.permutation(teams)          # team -> schedule position
        team_at = np.argsort(perm)              # schedule position -> team
        opponent = team_at[self.schedule[:, perm]].T   # (teams, regular_weeks)

        played = scores[:, :league.regular_weeks]
        # Each team's opponent's score, week by week, in one gather.
        against = played[opponent, np.arange(league.regular_weeks)[None, :]]

        wins = (played > against).sum(axis=1).astype(np.float64)
        wins += 0.5 * (played == against).sum(axis=1)
        return wins, played.sum(axis=1)

    def _seed(self, wins, points):
        """Team indices best to worst: wins, then points for. Standard tiebreak."""
        return np.lexsort((-points, -wins))

    def _bracket(self, seeds, scores):
        """Single elimination over the seeded field, reseeded each round.

        Byes go to the top seeds when the field is not a power of two -- with six
        teams that is the familiar shape: seeds 1 and 2 rest while 3 plays 6 and 4
        plays 5. Each subsequent round re-pairs highest surviving seed against
        lowest, which is what ESPN does and what makes a first-round bye worth
        having.
        """
        league = self.league
        field = [int(t) for t in seeds[:league.playoff_teams]]
        pow2 = 1 << (len(field) - 1).bit_length()
        byes = pow2 - len(field)
        advanced, playing = field[:byes], field[byes:]
        week = league.regular_weeks

        while len(advanced) + len(playing) > 1:
            winners = []
            half = len(playing) // 2
            for i in range(half):
                a, b = playing[i], playing[len(playing) - 1 - i]
                winners.append(a if scores[a, week] >= scores[b, week] else b)
            survivors = advanced + winners
            # Re-seed: back into the order the regular season established.
            rank = {t: i for i, t in enumerate(field)}
            survivors.sort(key=lambda t: rank[t])
            advanced, playing = [], survivors
            week += 1
        return playing[0] if playing else advanced[0]
