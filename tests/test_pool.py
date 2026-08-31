"""The pool: what came out of the CSVs, and whether it can be drafted from."""

import numpy as np
import pytest

from draftsim.config import (ADP_TIEBREAK_SPAN, DST, K, POSITIONS, QB, RB, SKILL_POSITIONS,
                             TE, WR)
from draftsim.pool import load_pool


def test_every_player_has_a_bye_week(pool):
    """A player with no bye is a player who never sits, which would be a free win.

    read_players drops the bye week, so it is joined back on by name. A join by name
    is exactly the sort of thing that silently half-works, and the season simulator
    would not complain -- it would just quietly give some rosters seventeen playable
    weeks in a fourteen-week season.
    """
    missing = [pool.name[i] for i in range(pool.size) if pool.bye[i] < 1]
    assert not missing, f"no bye week for {missing[:5]}"
    assert pool.bye.max() <= 18


def test_pool_is_in_adp_order(pool):
    """Index order is market order, and everything downstream assumes it."""
    assert np.all(np.diff(pool.adp_key) >= 0)


def test_tied_adps_are_ordered_by_rank(pool):
    """ESPN's undrafted plateau must be ordered by projection, not arbitrarily.

    Around ADP 170 dozens of players share a value. Left alone they would be drafted
    in whatever order the CSV happened to list them, so late rounds would ignore
    projections entirely -- and late rounds are where a plan's depth is decided.
    """
    groups = {}
    for i in range(pool.size):
        groups.setdefault(round(float(pool.adp[i]), 1), []).append(i)
    tied = [members for members in groups.values() if len(members) > 1]
    assert tied, "expected some tied ADPs in this data"
    for members in tied:
        # Pool order is index order, so members come out already in pool order.
        ranks = [int(pool.rank[i]) for i in members]
        assert ranks == sorted(ranks), ranks


def test_the_tiebreak_cannot_reorder_genuinely_different_adps(pool):
    """The tie-break has to be too small to matter, or it is not a tie-break.

    ADP is reported to one decimal, so the smallest real gap is 0.1. The nudge is
    spread across the pool so its total span is fixed however deep the pool goes;
    checked against the data rather than against the constant, because the bug this
    replaces was a per-rank increment that grew with pool depth.
    """
    assert ADP_TIEBREAK_SPAN < 0.1
    nudges = pool.adp_key - pool.adp
    assert nudges.min() >= 0
    assert nudges.max() <= ADP_TIEBREAK_SPAN + 1e-12
    # And the strongest test: pool order never contradicts ADP order by more than a
    # tie, i.e. a player never precedes someone with a strictly smaller ADP.
    assert np.all(np.diff(pool.adp) >= -1e-9)


def test_kickers_and_defenses_survive_the_rank_cap(pool, league):
    """They sit past overall rank 190, so a naive skill cap would eat them.

    The board leaves them out because they are not a decision. The simulator cannot:
    they are startable, and a team with no kicker starts a zero every week.
    """
    for code in (K, DST):
        assert (pool.pos == code).sum() >= league.teams


def test_skill_pool_respects_the_cap():
    small = load_pool(pool_skill=120)
    skill = np.isin(small.pos, list(SKILL_POSITIONS))
    assert small.rank[skill].max() <= 120
    # And the cap must not have touched the kickers.
    assert (small.pos == K).sum() >= 12


def test_replacement_levels_are_ordered_the_way_the_lineup_implies(pool):
    """Positions you start more of run out sooner, so their replacement is lower.

    One flex-eligible starter each at RB and WR versus two mandatory, plus two flex
    spots resolved on merit -- so RB and WR replacement should land close together
    and well below quarterback, where only twelve start in the whole league.
    """
    assert pool.replacement[QB] > pool.replacement[RB]
    assert pool.replacement[QB] > pool.replacement[WR]
    assert abs(pool.replacement[RB] - pool.replacement[WR]) < 3.0
    assert all(pool.replacement[p] > 0 for p in range(len(POSITIONS)))


def test_te_premium_raises_tight_end_scoring():
    """The premium is the league's defining rule; if it does nothing, it is not wired.

    Checked on the tight ends' own points rather than on ranks, because the rank and
    ADP shifts are a model of the market and could legitimately be argued with. The
    extra half point per reception cannot.
    """
    premium = load_pool(te_premium=True)
    standard = load_pool(te_premium=False)
    p_te = np.sort(premium.ppg[premium.pos == TE])[::-1][:12]
    s_te = np.sort(standard.ppg[standard.pos == TE])[::-1][:12]
    assert p_te.sum() > s_te.sum()
    # And nobody else's scoring moved.
    for code in (QB, RB, WR):
        a = np.sort(premium.ppg[premium.pos == code])[::-1][:24]
        b = np.sort(standard.ppg[standard.pos == code])[::-1][:24]
        assert np.allclose(a, b)


def test_espn_adp_model_differs_from_premium_and_keeps_premium_scoring():
    """--adp-model changes the room's clock, never the league's scoring."""
    espn = load_pool(adp_model="espn")
    premium = load_pool(adp_model="premium")
    assert espn.adp_model == "espn"
    # Same players, same points; the premium repriced timing, so ADP must differ.
    by_name_espn = {espn.name[i]: espn.ppg[i] for i in range(espn.size)}
    shared = [n for n in premium.name if n in by_name_espn]
    assert len(shared) > 200
    assert all(abs(by_name_espn[n] - premium.ppg[premium.index_of(n)]) < 1e-9
               for n in shared[:50])
    espn_adp = np.array([espn.adp[espn.index_of(n)] for n in shared])
    prem_adp = np.array([premium.adp[premium.index_of(n)] for n in shared])
    assert not np.allclose(espn_adp, prem_adp)


def test_index_of_and_describe_round_trip(pool):
    name = pool.name[0]
    assert pool.index_of(name) == 0
    assert name in pool.describe(0)
    assert pool.index_of("Nobody At All") is None
