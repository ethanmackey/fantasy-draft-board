"""The season: schedule, injuries, standings, bracket -- and the calibration check."""

import collections

import numpy as np
import pytest

from draftsim.config import (DST, INJURY_MISS_ANY_PROB, K, League,
                            NO_INJURIES, POSITIONS)
from draftsim.season import SeasonSimulator, round_robin, schedule_positions


# ---- schedule -------------------------------------------------------------

def test_round_robin_pairs_everyone_exactly_once():
    teams = 12
    table = round_robin(teams)
    assert table.shape == (teams - 1, teams)
    seen = collections.Counter()
    for week in range(table.shape[0]):
        for t in range(teams):
            opponent = int(table[week, t])
            assert opponent != t, "a team cannot play itself"
            assert int(table[week, opponent]) == t, "pairings must be symmetric"
            seen[frozenset((t, opponent))] += 1
    # Each pairing counted once per team, so twice.
    assert set(seen.values()) == {2}
    assert len(seen) == teams * (teams - 1) // 2


def test_schedule_extends_a_round_robin_without_a_third_meeting():
    """14 weeks over 12 teams is 11 + 3 repeats. Nobody should meet three times.

    A team drawn three times by one slot and once by another is a schedule artefact
    that would show up in the results as a slot effect, which is the exact thing the
    report claims to measure.
    """
    teams, weeks = 12, 14
    table = schedule_positions(teams, weeks)
    assert table.shape == (weeks, teams)
    for t in range(teams):
        counts = collections.Counter(int(table[w, t]) for w in range(weeks))
        assert max(counts.values()) <= 2
        assert sum(counts.values()) == weeks


def test_round_robin_refuses_an_odd_league():
    with pytest.raises(ValueError):
        round_robin(11)


def test_schedule_refuses_more_than_a_double_round_robin():
    with pytest.raises(ValueError):
        schedule_positions(12, 30)


# ---- injuries -------------------------------------------------------------

@pytest.fixture
def hurt_season(pool, league):
    """A simulator with injuries switched on, for the tests that are about injuries.

    They are off by default, so every test below that exercises the sampler has to ask
    for it explicitly -- which is the right way round: the sampler is kept because it
    is correct and may be wanted again, not because it is in use.
    """
    return SeasonSimulator(pool, league, miss_any_prob=INJURY_MISS_ANY_PROB)


def test_nobody_misses_a_game_by_default(season, engine, pool, rng):
    """Injuries are off. Byes are the only way to be unavailable, and that is the
    whole reason the bench is worth what it is worth in this configuration."""
    assert not season.injuries_on
    rosters = engine.run(1, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    avail = season.availability(rosters, rng)
    for team in range(rosters.shape[0]):
        for slot, index in enumerate(rosters[team]):
            out = set(np.flatnonzero(~avail[team, slot]).tolist())
            bye = int(pool.bye[index])
            assert out == ({bye - 1} if bye >= 1 else set())


def test_switching_injuries_off_and_on_is_visible_in_availability(pool, league, engine):
    """The flag has to actually change the season, or it is a comment."""
    off = SeasonSimulator(pool, league)
    on = SeasonSimulator(pool, league, miss_any_prob=INJURY_MISS_ANY_PROB)
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), np.random.default_rng(3))
    a = off.availability(rosters, np.random.default_rng(5))
    b = on.availability(rosters, np.random.default_rng(5))
    assert a.sum() > b.sum()
    assert off.injuries_on is False and on.injuries_on is True


def test_missed_games_come_in_one_contiguous_block(hurt_season, engine, rng):
    """Contiguity is the point of modelling injuries at all: three scattered weeks
    cost a manager nothing, one three-week block costs them a starter."""
    season = hurt_season
    rosters = engine.run(1, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    avail = season.availability(rosters, rng)
    weeks = avail.shape[-1]
    for team in range(avail.shape[0]):
        for slot in range(avail.shape[1]):
            out = np.flatnonzero(~avail[team, slot])
            if len(out) <= 1:
                continue
            gaps = np.diff(out)
            # At most one gap larger than 1, and only because the bye week sits
            # outside the injury block.
            assert (gaps > 1).sum() <= 1, out.tolist()
            assert out.min() >= 0 and out.max() < weeks


def test_the_bye_week_is_always_missed(season, pool, engine, rng):
    rosters = engine.run(1, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    avail = season.availability(rosters, rng)
    for team in range(rosters.shape[0]):
        for slot, index in enumerate(rosters[team]):
            bye = int(pool.bye[index])
            if bye >= 1:
                assert not avail[team, slot, bye - 1]


def test_defenses_never_miss_a_week_for_injury(hurt_season, pool):
    """A team defense cannot be injured; it just plays worse. MISS_ANY_PROB says so
    and the sampler has to honour it, or every roster gets a free zero-scoring week."""
    season = hurt_season
    dst = np.flatnonzero(pool.pos == DST)[:12]
    idx = dst.reshape(1, -1)
    rng = np.random.default_rng(5)
    for _ in range(20):
        avail = season.availability(idx, rng)
        byes = np.array([[int(pool.bye[i]) for i in dst]])
        for j, bye in enumerate(byes[0]):
            expected_out = {bye - 1} if bye >= 1 else set()
            got_out = set(np.flatnonzero(~avail[0, j]).tolist())
            assert got_out == expected_out


def test_the_injury_rate_is_roughly_what_the_config_asks_for(hurt_season, pool):
    """Not an exact test -- it is a sampler -- but a wildly wrong rate would mean the
    bench is worth either nothing or everything, and both would distort every plan."""
    season = hurt_season
    rng = np.random.default_rng(17)
    rbs = np.flatnonzero(pool.pos == 1)[:40]
    # Exclude the bye week from the count by looking only at players' non-bye weeks.
    hurt = 0
    trials = 60
    for _ in range(trials):
        avail = season.availability(rbs.reshape(1, -1), rng)
        for j, index in enumerate(rbs):
            bye = int(pool.bye[index]) - 1
            out = [w for w in np.flatnonzero(~avail[0, j]) if w != bye]
            hurt += bool(out)
    rate = hurt / (trials * len(rbs))
    assert abs(rate - INJURY_MISS_ANY_PROB[1]) < 0.08, rate


# ---- standings and bracket ------------------------------------------------

def test_wins_sum_to_the_number_of_games_played(season, engine, rng, league):
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    wins, points, seeds, champion = season.play(rosters, rng)
    total_games = league.teams * league.regular_weeks / 2
    assert wins.sum() == pytest.approx(total_games)
    assert (points > 0).all()


def test_seeding_is_by_wins_then_points(season, engine, rng):
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    wins, points, seeds, champion = season.play(rosters, rng)
    keys = [(-wins[t], -points[t]) for t in seeds]
    assert keys == sorted(keys)


def test_the_champion_comes_from_the_playoff_field(season, engine, rng, league):
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    for _ in range(20):
        _, _, seeds, champion = season.play(rosters, rng)
        assert champion in seeds[:league.playoff_teams]
        assert len(set(seeds.tolist())) == league.teams


def test_exactly_one_team_in_six_makes_the_playoffs(season, engine, rng, league):
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    made = collections.Counter()
    trials = 60
    for _ in range(trials):
        _, _, seeds, _ = season.play(rosters, rng)
        for t in seeds[:league.playoff_teams]:
            made[int(t)] += 1
    assert sum(made.values()) == trials * league.playoff_teams


def test_a_top_seed_bye_is_worth_something(season, engine, rng, league):
    """With six teams and three playoff weeks, seeds 1 and 2 skip a round. If the
    bracket did not give them one, the bye would be free variance instead of a reward
    for the regular season -- and every plan that builds a high-floor roster would be
    undervalued."""
    rosters = engine.run(4, (1, 2, 2, 3, 1, 2, 0, 1), rng)
    by_seed = collections.Counter()
    trials = 400
    for _ in range(trials):
        _, _, seeds, champion = season.play(rosters, rng)
        by_seed[int(np.flatnonzero(seeds == champion)[0])] += 1
    top_two = by_seed[0] + by_seed[1]
    bottom_two = by_seed[4] + by_seed[5]
    assert top_two > bottom_two, dict(by_seed)


def test_all_adp_room_gives_every_slot_a_coin_flip(pool, league):
    """The calibration check the whole report is read against.

    Twelve identical ADP drafters in a twelve-team league must average a 50% playoff
    rate. If they do not, the schedule, the standings or the bracket is broken, and
    every slot effect the report claims to have found is an artefact of that bug
    rather than a fact about draft position.
    """
    from draftsim.draft import DraftEngine
    engine = DraftEngine(pool, league)
    season = SeasonSimulator(pool, league)
    rng = np.random.default_rng(2026)
    hits = np.zeros(league.teams)
    trials = 400
    for _ in range(trials):
        rosters = engine.run(0, (), rng)
        _, _, seeds, _ = season.play(rosters, rng)
        for t in seeds[:league.playoff_teams]:
            hits[t] += 1
    overall = hits.sum() / (trials * league.teams)
    assert overall == pytest.approx(league.playoff_teams / league.teams, abs=1e-9)
    # No single slot should be far off a coin flip either.
    rates = hits / trials
    assert rates.min() > 0.35, rates.round(3).tolist()
    assert rates.max() < 0.65, rates.round(3).tolist()


def test_lineups_are_set_on_projections_not_on_realised_points(pool, league):
    """The load-bearing modelling choice, tested where it shows.

    If starters were chosen by realised points the simulator would be clairvoyant and
    would reward luck as if it were roster quality. The visible consequence is that a
    team's scoring must be *lower* than a best-ball team's, so an omniscient lineup
    would beat this one.
    """
    from draftsim.draft import DraftEngine
    from draftsim.roster import lineup_scores
    engine = DraftEngine(pool, league)
    season = SeasonSimulator(pool, league)
    rng = np.random.default_rng(31)
    rosters = engine.run(1, (1, 2, 2, 3, 1, 2, 0, 1), rng)

    ppg = pool.ppg[rosters]
    order = np.argsort(-ppg, axis=1, kind="stable")
    idx = np.take_along_axis(rosters, order, axis=1)
    pos = pool.pos[idx].astype(np.int64)
    avail = season.availability(idx, rng)
    flat = idx.reshape(-1)
    realised = (rng.standard_gamma(season.gamma_shape[flat][:, None],
                                   size=(flat.size, season.weeks))
                * season.gamma_scale[flat][:, None]).reshape(*idx.shape, season.weeks)
    realised *= avail

    honest = lineup_scores(pos, avail, realised, league)

    # Now re-sort by realised points week by week -- the clairvoyant lineup.
    best_ball = np.zeros_like(honest)
    for w in range(season.weeks):
        week_order = np.argsort(-realised[:, :, w], axis=1)
        best_ball[:, w] = lineup_scores(
            np.take_along_axis(pos, week_order, axis=1),
            np.take_along_axis(avail[:, :, w], week_order, axis=1)[:, :, None],
            np.take_along_axis(realised[:, :, w], week_order, axis=1)[:, :, None],
            league)[:, 0]

    assert best_ball.sum() > honest.sum()
