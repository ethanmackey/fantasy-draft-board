"""Plans: legality, sampling, exact search, and the archetype labels."""

import numpy as np
import pytest

from draftsim.config import PLAN_ROUNDS, POSITIONS, QB, RB, TE, WR
from draftsim.plans import (ARCHETYPES, archetype_plans, is_legal, label_plan,
                            legal_plans, parse_plan, plan_string, sample_plan,
                            top_plans_by_round_scores)


def test_plan_string_and_parse_round_trip():
    plan = (RB, WR, WR, TE, RB, WR, QB, RB)
    assert plan_string(plan) == "RB-WR-WR-TE-RB-WR-QB-RB"
    assert parse_plan("RB-WR-WR-TE-RB-WR-QB-RB") == plan
    assert parse_plan("rb,wr,te") == (RB, WR, TE)
    assert parse_plan("DEF") == (POSITIONS.index("DST"),)


def test_parse_rejects_a_non_position():
    with pytest.raises(ValueError):
        parse_plan("RB-WR-PUNTER")


def test_three_quarterbacks_in_eight_rounds_is_not_a_strategy():
    assert not is_legal((QB, QB, QB, RB, WR, WR, TE, RB))
    assert is_legal((QB, QB, RB, WR, WR, TE, RB, WR))


def test_a_plan_with_no_receiver_is_illegal():
    assert not is_legal((RB, RB, RB, RB, TE, TE, QB, RB))


def test_zero_rb_as_people_actually_mean_it_is_legal():
    """The floor of one running back is deliberately loose.

    "Zero RB" in practice means no back through round 4 or 5 and then two in a row --
    which has to be inside the search space, because it is one of the strategies the
    user asked about. What the floor excludes is a plan with no back at all in eight
    rounds, which cannot field a lineup.
    """
    assert is_legal(ARCHETYPES["Zero-RB"])
    assert ARCHETYPES["Zero-RB"][:4].count(RB) == 0


def test_the_legal_space_is_large_but_not_the_whole_space():
    plans = legal_plans(PLAN_ROUNDS)
    assert 0 < len(plans) < 4 ** PLAN_ROUNDS
    assert all(is_legal(p) for p in plans)
    assert len(set(plans)) == len(plans)


def test_sampling_only_ever_returns_a_legal_plan():
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert is_legal(sample_plan(rng))


def test_sampling_is_uniform_enough_to_measure_marginals():
    """Uniform over legal plans, not over sequences.

    The per-round marginals stage 1 reports are only comparable if each position is
    sampled a similar number of times in each round. Rejection sampling over raw
    sequences would leave the counts lopsided for reasons that have nothing to do with
    the data.
    """
    rng = np.random.default_rng(1)
    counts = np.zeros((PLAN_ROUNDS, len(POSITIONS)))
    for _ in range(4000):
        for r, pos in enumerate(sample_plan(rng)):
            counts[r, pos] += 1
    skill = counts[:, :4]
    share = skill / skill.sum(axis=1, keepdims=True)
    assert share.min() > 0.15, share.round(3).tolist()
    assert share.max() < 0.40, share.round(3).tolist()


def test_top_plans_search_is_exact_and_legal():
    """The nomination step must not need repairing.

    Greedily taking the best marginal in each round produces an illegal plan more
    often than not -- the same position is usually best in most rounds -- and repairing
    one means inventing a rule about which round to sacrifice. Enumerating instead is
    both exact and cheaper than the argument.
    """
    # Scores that greedily want eight receivers, which is illegal (no running back).
    scores = np.zeros((PLAN_ROUNDS, len(POSITIONS)))
    scores[:, WR] = 1.0
    scores[:, RB] = 0.9
    best = top_plans_by_round_scores(scores, 3)
    assert all(is_legal(p) for p in best)
    top = best[0]
    assert top.count(WR) == PLAN_ROUNDS - 1
    assert top.count(RB) == 1
    # And it must really be the maximum over the legal space.
    def total(plan):
        return sum(scores[r, pos] for r, pos in enumerate(plan))
    assert total(top) == max(total(p) for p in legal_plans(PLAN_ROUNDS))


def test_top_plans_are_returned_best_first():
    rng = np.random.default_rng(2)
    scores = rng.random((PLAN_ROUNDS, len(POSITIONS)))
    best = top_plans_by_round_scores(scores, 20)
    totals = [sum(scores[r, p] for r, p in enumerate(plan)) for plan in best]
    assert totals == sorted(totals, reverse=True)


def test_archetypes_are_all_legal_and_distinct():
    plans = archetype_plans()
    assert len(plans) == len(ARCHETYPES)
    assert len(set(plans.values())) == len(plans)
    assert all(is_legal(p) for p in plans.values())


def test_trimming_an_archetype_drops_the_ones_it_breaks():
    """Zero-RB cut to four rounds has no running back in it, so it must be filtered
    rather than silently offered as an illegal candidate."""
    short = archetype_plans(4)
    assert "Zero-RB" not in short
    assert all(is_legal(p) for p in short.values())


def test_a_named_archetype_gets_its_own_name_back():
    for label, plan in ARCHETYPES.items():
        assert label_plan(plan) == label


def test_labels_describe_the_shapes_they_claim():
    assert "Zero-RB" in label_plan((WR, WR, TE, WR, RB, RB, QB, RB))
    assert "Hero-RB" in label_plan((RB, WR, WR, WR, WR, RB, QB, RB))
    assert "Robust-RB" in label_plan((RB, RB, WR, TE, WR, WR, QB, RB))
    assert "Elite-TE" in label_plan((TE, RB, WR, RB, WR, WR, QB, WR))
    assert "Early-QB" in label_plan((QB, RB, WR, RB, WR, TE, WR, WR))
    assert "Late-QB" in label_plan((RB, RB, WR, WR, TE, WR, WR, QB))


def test_an_unnameable_plan_gets_its_counts_instead_of_an_invented_name():
    """A label exists so a reader recognises the shape without decoding eight codes.
    Making one up for a shape nobody has a word for would be worse than describing it.
    """
    label = label_plan((QB, TE, QB, TE, RB, WR, WR, WR))
    assert label
    assert all(part.strip() for part in label.split("/"))
