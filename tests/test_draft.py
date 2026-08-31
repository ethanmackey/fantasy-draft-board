"""The draft: snake order, roster integrity, and whether the policies do what they say."""

import collections

import numpy as np
import pytest

from draftsim.config import DST, K, POSITIONS, QB, RB, TE, WR
from draftsim.draft import DraftEngine, slot_picks, snake_order
from draftsim.plans import sample_plan

PLAN = (RB, WR, WR, TE, RB, WR, QB, RB)


# ---- order ----------------------------------------------------------------

def test_snake_turns_around_at_the_end_of_each_round():
    order = snake_order(4, 3)
    assert list(order) == [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3]


def test_slot_one_owns_the_turn_picks():
    """The snake's whole character: slot 1 gets 1 and 24, slot 12 gets 12 and 13."""
    assert slot_picks(1, 12, 16)[:2] == [1, 24]
    assert slot_picks(12, 12, 16)[:2] == [12, 13]
    assert slot_picks(6, 12, 16)[:2] == [6, 19]


def test_every_slot_gets_one_pick_per_round(league):
    order = snake_order(league.teams, league.rounds)
    assert len(order) == league.picks
    assert collections.Counter(order.tolist()) == {
        t: league.rounds for t in range(league.teams)}


def test_slot_picks_and_snake_order_agree(league):
    order = snake_order(league.teams, league.rounds)
    for slot in range(1, league.teams + 1):
        from_order = [i + 1 for i, t in enumerate(order) if t == slot - 1]
        assert from_order == slot_picks(slot, league.teams, league.rounds)


# ---- integrity ------------------------------------------------------------

@pytest.fixture
def drafted(engine, rng):
    return engine.run(6, PLAN, rng)


def test_nobody_is_drafted_twice(drafted, league):
    flat = drafted.reshape(-1)
    assert len(set(flat.tolist())) == league.picks


def test_every_team_ends_with_exactly_one_kicker_and_one_defense(drafted, pool, league):
    for team in range(league.teams):
        got = collections.Counter(POSITIONS[pool.pos[p]] for p in drafted[team])
        assert got["K"] == 1, got
        assert got["DST"] == 1, got


def test_kicker_and_defense_come_only_from_the_reserved_rounds(drafted, pool, league):
    """If they leak earlier they are competing with real picks, which is the one thing
    reserving the rounds exists to prevent."""
    reserved = league.rounds - 2
    for team in range(league.teams):
        for r, index in enumerate(drafted[team]):
            code = pool.pos[index]
            if code in (K, DST):
                assert r >= reserved, f"{POSITIONS[code]} in round {r + 1}"
            else:
                assert r < reserved


def test_no_roster_exceeds_a_cap(drafted, pool, league):
    for team in range(league.teams):
        got = collections.Counter(int(pool.pos[p]) for p in drafted[team])
        for code, count in got.items():
            assert count <= league.caps[code], f"{POSITIONS[code]} x{count}"


def test_every_team_can_field_a_full_starting_lineup(drafted, pool, league):
    """The starter-completion guard's whole job, checked on the output rather than the
    rule: no team may reach the end of the draft unable to start somebody at each slot."""
    for team in range(league.teams):
        got = collections.Counter(int(pool.pos[p]) for p in drafted[team])
        for code, need in enumerate(league.starters):
            assert got[code] >= need, f"team {team} short at {POSITIONS[code]}"


def test_all_bot_mode_gives_nobody_a_plan(engine, rng, league, pool):
    """slot=0 is the calibration mode; it must still produce twelve legal rosters."""
    rosters = engine.run(0, (), rng)
    assert len(set(rosters.reshape(-1).tolist())) == league.picks
    for team in range(league.teams):
        got = collections.Counter(int(pool.pos[p]) for p in rosters[team])
        assert got[K] == 1 and got[DST] == 1


# ---- the plan -------------------------------------------------------------

def test_the_plan_is_followed_when_it_can_be(engine, pool, rng):
    """A plan is a target: over many drafts it should be hit nearly every round.

    Not "always", because a position can be closed by a cap or exhausted, and the
    engine falls back to value rather than skipping the pick. But a plan that is
    usually overridden would mean the report is measuring something other than the
    plan it names.
    """
    hits = np.zeros(len(PLAN), dtype=int)
    trials = 40
    for _ in range(trials):
        rosters = engine.run(3, PLAN, rng)
        for r, wanted in enumerate(PLAN):
            hits[r] += pool.pos[rosters[2, r]] == wanted
    assert (hits == trials).all(), dict(enumerate(hits.tolist()))


def test_the_hero_takes_the_best_projected_player_at_the_planned_position(engine, pool, rng):
    """Round 1 from slot 1: nobody has picked, so the plan's position has an
    unambiguous best available and the hero must take exactly him."""
    best_rb = int(pool.by_pos[RB][0])
    rosters = engine.run(1, (RB,) * 8, rng)
    assert rosters[0, 0] == best_rb


def test_the_adp_room_takes_the_lowest_adp_player_first_with_no_noise(pool, league):
    """The `adp` room is kept for comparison, so it still has to behave like one.

    With sigma zero the opponents are a pure ADP queue. Only the first pick is
    checked as an identity, because after that the roster rules legitimately
    intervene -- but that one pick is enough to prove ADP, not value, is driving them:
    the pool is in ADP order, so index 0 is the market's first name and it is not
    generally the highest-value player.
    """
    engine = DraftEngine(pool, league, room="adp", sigma_base=0.0, sigma_rate=0.0,
                         perception_sigma=0.0)
    rosters = engine.run(2, (RB,) * 8, np.random.default_rng(1))
    assert int(rosters[0, 0]) == 0


def test_the_value_room_ignores_adp_entirely(pool, league):
    """Nothing about a value-room draft may depend on the market's ordering.

    Checked by making the ADP key meaningless -- reversed -- and confirming the draft
    is byte-identical. If any ADP influence had survived into the value room, this
    would move players around.
    """
    import dataclasses
    scrambled = dataclasses.replace(pool, adp_key=pool.adp_key[::-1].copy())
    plan = (RB, WR, WR, TE, RB, WR, QB, RB)
    a = DraftEngine(pool, league).run(5, plan, np.random.default_rng(8)).copy()
    b = DraftEngine(scrambled, league).run(5, plan, np.random.default_rng(8)).copy()
    assert np.array_equal(a, b)


def test_every_seat_uses_the_same_policy_in_the_value_room(pool, league):
    """The symmetry the baseline depends on.

    With no plan and no disagreement, all twelve seats run one rule, so the draft is
    fully determined: two runs with different seeds must produce the same rosters.
    Any asymmetry between your seat and the others would break this, and would make
    the baseline an unfair yardstick.
    """
    engine = DraftEngine(pool, league, perception_sigma=0.0)
    a = engine.run(0, (), np.random.default_rng(1)).copy()
    b = engine.run(0, (), np.random.default_rng(999)).copy()
    assert np.array_equal(a, b)


def test_disagreement_is_what_makes_drafts_differ(pool, league):
    """And with it switched on, two seeds must NOT agree.

    This is the only source of draft-to-draft variety once ADP is out of the bot
    policy. If it stopped working the simulator would silently be studying one draft.
    """
    engine = DraftEngine(pool, league, perception_sigma=0.15)
    a = engine.run(0, (), np.random.default_rng(1)).copy()
    b = engine.run(0, (), np.random.default_rng(2)).copy()
    assert not np.array_equal(a, b)


def test_perception_is_mean_preserving(pool, league):
    """A lognormal without the -sigma^2/2 shift makes every drafter systematically
    optimistic, which would bias every comparison against a replacement level computed
    from the unperturbed projections -- and the bench discount is such a comparison."""
    from draftsim.bots import perceived_ppg
    rng = np.random.default_rng(4)
    got = perceived_ppg(rng, pool.ppg, 4000, 0.15)
    assert got.mean(axis=0) == pytest.approx(pool.ppg, rel=0.02)


def test_perception_of_zero_returns_the_projections_untouched(pool):
    from draftsim.bots import perceived_ppg
    got = perceived_ppg(np.random.default_rng(0), pool.ppg, 3, 0.0)
    assert got.shape == (3, pool.size)
    for row in got:
        assert np.array_equal(row, pool.ppg)


def test_bench_discount_stops_the_backup_quarterback_hoard(pool, league, rng):
    """The bug bench discounting exists to kill.

    Undiscounted, the value fill takes a second quarterback in the middle rounds,
    because the twentieth-best quarterback clears quarterback replacement and nothing
    in the arithmetic knows he will never start. With the discount, second
    quarterbacks should be rare rather than routine.
    """
    plan = (RB, WR, WR, TE, RB, WR, QB, RB)

    def qb_rate(decay):
        engine = DraftEngine(pool, league, bench_decay=decay)
        r = np.random.default_rng(9)
        extra = 0
        trials = 25
        for _ in range(trials):
            rosters = engine.run(4, plan, r)
            extra += sum(1 for p in rosters[3] if pool.pos[p] == QB) - 1
        return extra / trials

    assert qb_rate(1.0) > qb_rate(0.35)
    assert qb_rate(0.35) < 0.5


def test_a_value_room_leaves_you_less_than_an_adp_room(pool, league):
    """The whole reason the default changed.

    Opponents who read the projections take the players a value drafter wanted, so the
    same plan must come away with a weaker roster than it does against a room drafting
    off ADP alone. If this did not hold, the +37-point edge the ADP room reported was
    not projection arbitrage after all.
    """
    plan = (RB, WR, WR, TE, RB, WR, QB, RB)
    def total_ppg(room):
        engine = DraftEngine(pool, league, room=room)
        r = np.random.default_rng(5)
        return sum(float(pool.ppg[p]) for _ in range(15)
                   for p in engine.run(6, plan, r)[5])

    assert total_ppg("value") < total_ppg("adp")


def test_an_unknown_room_is_rejected(pool, league):
    with pytest.raises(ValueError, match="room must be"):
        DraftEngine(pool, league, room="auction")


def test_a_reused_engine_gives_the_same_answer_as_a_fresh_one(pool, league):
    """Buffers are reused across drafts; a leak between drafts would be invisible in
    the results and fatal to them."""
    plan = sample_plan(np.random.default_rng(3))
    a = DraftEngine(pool, league).run(7, plan, np.random.default_rng(11)).copy()
    shared = DraftEngine(pool, league)
    shared.run(2, plan, np.random.default_rng(99))       # dirty the buffers
    b = shared.run(7, plan, np.random.default_rng(11)).copy()
    assert np.array_equal(a, b)
