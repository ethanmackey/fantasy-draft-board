"""Roster rules and the lineup solver.

Two things live here, and they are the same thing seen from either end of the
draft: what a team is *allowed* to draft, and what a team can *start* once it has.
Both are pure functions of a roster's position counts, so both are testable
without a draft or a season anywhere near them.

The lineup solver appears twice on purpose. ``lineup_score_one`` is the obvious
implementation for one team in one week, written to be read. ``lineup_scores`` is
the same rule vectorised over every team and every week at once, written to be
fast, and it is what the season simulator calls half a million times. A test
asserts they agree on random rosters, which is the only reason it is safe to have
two.
"""

import numpy as np

from .config import FLEX_POSITIONS, LATE_ONLY, POSITIONS

# Boolean lookup, indexed by position code: can this position fill a flex slot?
FLEX_ELIGIBLE = np.zeros(len(POSITIONS), dtype=bool)
FLEX_ELIGIBLE[list(FLEX_POSITIONS)] = True
LATE_ONLY_MASK = np.zeros(len(POSITIONS), dtype=bool)
LATE_ONLY_MASK[list(LATE_ONLY)] = True


def starter_need(counts, league):
    """How many more players this roster needs to field a legal starting lineup.

    Mandatory slots first, then the flex: a flex slot is satisfied by any surplus
    flex-eligible player, so a team with three running backs and two receivers has
    already covered one of its two flex spots without drafting for it.

    Kickers and defenses are counted here like anything else. Their rounds are
    reserved separately, so in practice their deficit is only ever the reason the
    guard fires in the last two rounds -- which is exactly when it should.
    """
    need = 0
    flex_surplus = 0
    for pos, required in enumerate(league.starters):
        have = counts[pos]
        if have < required:
            need += required - have
        elif FLEX_ELIGIBLE[pos]:
            flex_surplus += have - required
    return need + max(0, league.flex - flex_surplus)


def allowed_positions(counts, picks_made, league, out=None):
    """Boolean array over positions: what may this team draft with its next pick?

    Three rules, in order of how hard they bite:

    1. **The reserved rounds.** The last ``len(LATE_ONLY)`` rounds are for the
       kicker and the defense, and nothing else is legal in them. Reserving the
       rounds rather than merely allowing K and DST late is what guarantees every
       team ends with exactly one of each -- and it keeps them out of the plan
       space entirely, which is the point: the gap between the best kicker and the
       twelfth is smaller than one week of noise, so a kicker decision is noise
       dressed as a strategy.

    2. **Caps.** A position at its cap is closed. These are not ESPN's limits;
       they are the limits a drafter who wants to win imposes on themselves, and
       they exist here mostly so the eleven opponents do not hoard. Every
       quarterback a bot takes as its third is a player who should have fallen to
       somebody.

    3. **The starter-completion guard.** When a team has exactly as many picks
       left as holes in its starting lineup, it may only draft players who fill
       one. Without this, ADP-driven bots reach the end of the draft with no tight
       end and start a zero every week -- which is not a mistake a real drafter
       makes, and it would quietly inflate every plan that competes for tight ends.

    ``out`` lets a caller reuse one buffer across a whole draft; this is called
    once per pick.
    """
    n = len(POSITIONS)
    if out is None:
        out = np.empty(n, dtype=bool)

    reserved = league.rounds - len(LATE_ONLY)
    if picks_made >= reserved:
        # A reserved round: only the late-only positions still missing, which under
        # rule 1 means K in the first reserved round and DST in the second.
        for pos in range(n):
            out[pos] = LATE_ONLY_MASK[pos] and counts[pos] < league.caps[pos]
        return out

    for pos in range(n):
        out[pos] = (not LATE_ONLY_MASK[pos]) and counts[pos] < league.caps[pos]

    # The guard, measured against the skill phase only: the K and DST rounds are
    # already spoken for, so the picks available to fix a hole at receiver are the
    # ones before the reserved rounds.
    skill_picks_left = reserved - picks_made
    need = starter_need(counts, league)
    late_need = sum(1 for pos in LATE_ONLY if counts[pos] < league.starters[pos])
    skill_need = need - late_need
    if skill_picks_left <= skill_need:
        flex_short = _flex_short(counts, league)
        for pos in range(n):
            if not out[pos]:
                continue
            fills = counts[pos] < league.starters[pos] or (
                flex_short and FLEX_ELIGIBLE[pos])
            out[pos] = bool(fills)
        # A roster can be boxed in only if the caps contradict the lineup, which
        # League.validate already rejects. Falling back to "anything legal" rather
        # than raising keeps a hand-built pool from crashing a simulation.
        if not out.any():
            for pos in range(n):
                out[pos] = (not LATE_ONLY_MASK[pos]) and counts[pos] < league.caps[pos]
    return out


def _flex_short(counts, league):
    """Is this roster still short of flex bodies?"""
    surplus = sum(max(0, counts[pos] - league.starters[pos])
                  for pos in FLEX_POSITIONS)
    return surplus < league.flex


def startable(pos, league):
    """How many at this position could plausibly be in a weekly lineup.

    Mandatory slots, plus every flex slot for a flex-eligible position. Deliberately
    optimistic: the two flex spots are counted once for running backs, once for
    receivers and once for tight ends, so the three numbers sum to more slots than
    the lineup has.

    That is the right sort of wrong for the question this answers. "Could my fourth
    running back ever be in the lineup" is yes -- backs get hurt, backs have byes,
    and one of the flex spots is often his. Charging him for the receivers who might
    have taken that flex spot instead would say no, and would rank him behind a
    backup quarterback who genuinely cannot play.
    """
    return league.starters[pos] + (league.flex if FLEX_ELIGIBLE[pos] else 0)


def bench_depth(counts, pos, league):
    """How deep past startable would the next player at ``pos`` land? 0 means startable.

    The distinction value-over-replacement cannot make on its own. VOR asks how much
    better than a startable player someone is, which is the right question for a pick
    who will play and the wrong one for a pick who will not: the twentieth
    quarterback in the league still clears quarterback replacement, and the
    arithmetic has no idea only one of him can start.

    Measured per position against ``startable``, not against the pooled flex
    requirement that ``starter_need`` uses. Those are two different questions and
    conflating them was a real bug: pooling the flex surplus made a fifth receiver
    count as five deep -- because his team's spare backs and tight ends counted
    against him too -- which drove his discount to almost nothing and let a backup
    quarterback win the comparison on the strength of raw projected points. Legality
    is about filling one lineup, so it pools. Depth is about how often this
    particular player takes the field, so it does not.
    """
    have = int(counts[pos])
    limit = startable(pos, league)
    return have - limit + 1 if have >= limit else 0


def lineup_score_one(roster_pos, available, points, league, free_agents=None,
                     free_agent_mean=None):
    """Points scored by one team in one week. The readable implementation.

    ``roster_pos``, ``available`` and ``points`` are parallel and must already be
    ordered by *projected* PPG, descending. That ordering is the manager's
    decision and it is the whole reason this is a single greedy pass: walking the
    roster best-projection first, giving each available player his mandatory slot
    if it is open and a flex slot otherwise, is optimal for projected points, and
    projected points is what a lineup is set on.

    Setting the lineup from projections and scoring it on realised points is the
    load-bearing choice in the whole simulator. Choosing starters by realised
    points would be clairvoyance, and it would reward a roster for being lucky as
    if it had been good -- which is the exact confusion the simulator exists to
    avoid.

    The bench is drawn from here, and always was: every one of the sixteen rostered
    players is walked, and "bench" simply means whoever did not win a slot. What
    ``free_agents`` adds is the week when *nothing on the roster* can fill a slot --
    a singleton starter's bye. Then the slot is streamed off the waiver wire rather
    than forfeited.

    That distinction was a real bug. Scoring an unfillable slot as zero meant the
    only way a draft could cover its quarterback's bye week was to spend a pick on a
    backup, and the search duly recommended a bench quarterback in round 8 ahead of
    a second starting receiver. No manager does that; they stream one arm for one
    week. Zero is the pessimistic bound, and the pessimistic bound was distorting
    every plan that carried a singleton.

    ``free_agents`` maps position to the streamer's *realised* points this week;
    ``free_agent_mean`` maps position to his projected PPG, which is what decides
    which flex-eligible streamer a manager would pick. Absent, an unfillable slot
    scores zero as before.
    """
    slots = list(league.starters)
    flex_left = league.flex
    total = 0.0
    for pos, ok, pts in zip(roster_pos, available, points):
        if not ok:
            continue
        if slots[pos] > 0:
            slots[pos] -= 1
            total += pts
        elif FLEX_ELIGIBLE[pos] and flex_left > 0:
            flex_left -= 1
            total += pts

    if not free_agents:
        return total

    for pos, left in enumerate(slots):
        if left > 0:
            total += left * free_agents.get(pos, 0.0)
    if flex_left > 0:
        # Whichever flex-eligible streamer projects best -- the manager's choice is
        # made on projection, so the pick is made on the mean and paid in realised
        # points, exactly as for a rostered player.
        means = free_agent_mean or free_agents
        best = max(FLEX_POSITIONS, key=lambda p: means.get(p, 0.0))
        total += flex_left * free_agents.get(best, 0.0)
    return total


def lineup_scores(roster_pos, available, points, league, wire=None, wire_mean=None):
    """Every team's score in every week. The same rule, vectorised.

    Shapes: ``roster_pos`` is (teams, roster), ``available`` and ``points`` are
    (teams, roster, weeks). ``wire`` is (teams, positions, weeks) of streamer
    realised points and ``wire_mean`` is (teams, positions) of their projections.
    Returns (teams, weeks).

    Rosters must be ordered by projected PPG descending along the roster axis, as
    in ``lineup_score_one``. The loop is over roster spots -- sixteen of them --
    not over teams or weeks, so the whole league's season costs a few dozen numpy
    calls rather than a few thousand Python iterations. That ratio is what makes a
    million simulated seasons finish in minutes.
    """
    teams, roster = roster_pos.shape
    weeks = available.shape[2]

    slots = np.repeat(np.asarray(league.starters, dtype=np.int16)[None, None, :],
                      teams * weeks, axis=0).reshape(teams, weeks, len(POSITIONS)).copy()
    flex_left = np.full((teams, weeks), league.flex, dtype=np.int16)
    total = np.zeros((teams, weeks), dtype=np.float64)

    for r in range(roster):
        pos = roster_pos[:, r]                                   # (teams,)
        ok = available[:, r, :]                                  # (teams, weeks)
        pts = points[:, r, :]                                    # (teams, weeks)
        idx = np.broadcast_to(pos[:, None, None], (teams, weeks, 1))
        have = np.take_along_axis(slots, idx, axis=2)[:, :, 0]    # (teams, weeks)

        use_mand = ok & (have > 0)
        total += np.where(use_mand, pts, 0.0)
        np.put_along_axis(slots, idx,
                          (have - use_mand.astype(np.int16))[:, :, None], axis=2)

        eligible = FLEX_ELIGIBLE[pos][:, None]                   # (teams, 1)
        use_flex = ok & ~use_mand & eligible & (flex_left > 0)
        total += np.where(use_flex, pts, 0.0)
        flex_left -= use_flex.astype(np.int16)

    if wire is None:
        return total

    # Whatever the roster could not fill is streamed. `slots` now holds the leftover
    # mandatory openings per (team, week, position) and `flex_left` the leftover flex
    # openings, so both are a single weighted sum against the wire.
    total += (slots * np.transpose(wire, (0, 2, 1))).sum(axis=2)

    if league.flex:
        means = wire_mean if wire_mean is not None else wire.mean(axis=2)
        flex_cols = list(FLEX_POSITIONS)
        # Per team, the flex-eligible streamer with the best projection -- chosen on
        # the mean, paid in realised points, exactly as a rostered player would be.
        best = np.asarray(flex_cols)[np.argmax(means[:, flex_cols], axis=1)]
        picked = wire[np.arange(teams), best, :]                  # (teams, weeks)
        total += flex_left * picked

    return total
