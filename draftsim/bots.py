"""The other eleven drafters: what they believe, and how they pick.

Two rooms live here.

**The `value` room** is the default and the one the study runs on. All twelve
seats -- yours included -- run one policy: take the best value over replacement
available. They are not clones, because each brought a different opinion of what
each player is worth. That opinion is the only thing separating them, and it is
where every bit of draft-to-draft variety now comes from: a room of twelve
identical drafters reading identical numbers produces the identical draft every
time, and a simulator with no draft variance can say nothing about what falls to
you.

Modelling disagreement rather than ADP ignorance is the more honest asymmetry.
ADP *is* the aggregate of everybody's rankings, and those rankings come from
projections closely correlated with these ones -- so a room that has never seen a
projection is a much sloppier opponent than any real room, and a plan measured
against it collects an edge that is projection arbitrage rather than strategy.

**The `adp` room** is the original model, kept for exactly that comparison: eleven
opponents taking the market's next name with noise that grows through the draft.
Running the study both ways is how you find out how much of a plan's edge was
strategy.
"""

import numpy as np


# --------------------------------------------------------------------------
# The value room: private opinions
# --------------------------------------------------------------------------

def perceived_ppg(rng, ppg, teams, sigma):
    """(teams, players) of what each drafter believes each player is worth.

    A mean-preserving lognormal multiplier on the projection:

        perceived = ppg * exp(Normal(-sigma^2/2, sigma))

    Mean-preserving matters. A plain ``exp(Normal(0, sigma))`` has mean
    ``exp(sigma^2/2) > 1``, so every drafter in the room would be systematically
    optimistic about every player -- harmless for the ordering within one drafter's
    board, but it would inflate every value comparison against a replacement level
    computed from the unperturbed projections, and the bench discount is exactly
    such a comparison.

    Lognormal rather than normal because a projection cannot be negative and because
    disagreement is multiplicative: people argue about a 20-PPG back in units of two
    points and about a 6-PPG one in units of half a point.

    Drawn once per drafter per draft. A drafter's opinion is a fixed thing he
    brought to the table -- re-rolling it per pick would model indecision rather
    than disagreement, and over sixteen picks it would average back out to no
    disagreement at all.

    ``sigma`` of 0 returns the projections untouched, which makes the room fully
    deterministic. That is a legitimate configuration and a useful test fixture; it
    is not a useful study.
    """
    if not sigma:
        return np.broadcast_to(ppg, (teams, len(ppg))).copy()
    noise = rng.normal(-0.5 * sigma * sigma, sigma, size=(teams, len(ppg)))
    return ppg[None, :] * np.exp(noise)


# --------------------------------------------------------------------------
# The adp room
# --------------------------------------------------------------------------

def pick_sigma(pick_number, sigma_base, sigma_rate):
    """Standard deviation of ADP noise, in picks, at this overall pick number.

    Linear in the pick number. Against this data -- where ``draft_tiers`` documents
    real dispersion of 15-25 picks either side through rounds 4-10 -- base 4.0 and
    rate 0.10 give a 1-sigma band of 8-16 picks and a 2-sigma band of 16-32 across
    that stretch, which brackets it. Pick 1 keeps a 4-pick sigma rather than zero
    because the first overall pick is not unanimous either.
    """
    return sigma_base + sigma_rate * pick_number


def noise_matrix(rng, picks, pool_size, sigma_base, sigma_rate):
    """(picks, pool_size) of ADP noise for one whole draft, drawn in one call.

    Drawn up front rather than per pick because 192 separate numpy calls cost more
    in call overhead than one call costs in arithmetic, and because it makes a draft
    reproducible from its seed alone: the noise a bot sees at pick 57 does not
    depend on how many random draws the previous 56 picks happened to make.
    """
    sigma = sigma_base + sigma_rate * np.arange(picks, dtype=np.float64)
    return rng.standard_normal((picks, pool_size)) * sigma[:, None]


def bot_choice(adp_key, noise_row, taken, allowed_by_player, work):
    """Index of the player an ADP-room opponent takes.

    ``work`` is a caller-owned scratch buffer, reused across all 192 picks of a
    draft; at half a million drafts the allocations this avoids are worth the
    slightly awkward signature.

    Blocked players are pushed to +inf rather than filtered out, because an argmin
    over the full array is one pass with no index bookkeeping, and the pool is
    small enough (a few hundred) that the pass is cheaper than compacting it.
    """
    np.add(adp_key, noise_row, out=work)
    np.copyto(work, np.inf, where=taken)
    np.copyto(work, np.inf, where=~allowed_by_player)
    return int(work.argmin())
