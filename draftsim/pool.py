"""The player pool: CSV in, numpy arrays out.

One job, and a boundary worth defending: this is the only module in the package
that knows what a CSV looks like or that ``draft_tiers`` exists. Everything
downstream sees a ``Pool`` of parallel arrays and nothing else.

Pricing is deliberately not reimplemented here. ``draft_tiers.read_players``
already converts a season projection to PPG, re-scores tight ends for the 1.5 PPR,
and moves rank and ADP by what that gain is worth -- and it is the function that
built the board the user already trusts. A second implementation of the premium
would be a second thing to keep in sync, and the first time they disagreed the
simulator would be quietly answering a question about a different league.

What this module adds on top of it:

* bye weeks, which ``read_players`` drops (its Player tuple has no room for one)
  and which the season simulator cannot do without;
* kickers and defenses, which the board leaves out because they are not a draft
  decision but which are startable and so have to be on a roster;
* the ADP sort key, including the tie-break that keeps ESPN's undrafted plateau
  from being shuffled at random;
* replacement level per position, so value-over-replacement means the same thing
  here as it does on the board.
"""

import csv
import os
from dataclasses import dataclass

import numpy as np

import draft_tiers

from .config import (ADP_TIEBREAK_SPAN, DEFAULT_LEAGUE, LATE_ONLY, POS_INDEX,
                     POSITIONS, SKILL_POSITIONS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DEFAULT_RANKINGS = os.path.join(SCRIPT_DIR, "Draft-rankings-export-2026 (8-21).csv")
DEFAULT_PROJECTIONS = os.path.join(SCRIPT_DIR, "projections (8-21).csv")


@dataclass(frozen=True)
class Pool:
    """Every draftable player, ordered by the market.

    Index order IS ADP order: player 0 is the first name off the board on
    average. That is not decoration. The draft engine scores candidates by ADP
    plus noise and the bots' choice is an argmin over this array, so having the
    array already in market order means a bot's "reach" and "fall" are visible as
    index distance, and it means a truncated view of the top of the pool is
    exactly the set of players anyone might plausibly take next.

    Arrays are parallel and never reordered after construction. Frozen, because
    one pool is shared read-only across every worker process and every simulation.
    """

    name: np.ndarray          # object, player names as printed
    team: np.ndarray          # object, NFL team abbreviation
    pos: np.ndarray           # int8, index into config.POSITIONS
    ppg: np.ndarray           # float64, projected points per game under our scoring
    adp: np.ndarray           # float64, average draft position (see adp_model)
    rank: np.ndarray          # int32, overall rank from the export
    bye: np.ndarray           # int8, bye week, 1-based; 0 means none known
    adp_key: np.ndarray       # float64, adp with the rank tie-break folded in
    by_pos: tuple             # per position, indices sorted by ppg descending
    replacement: np.ndarray   # float64, replacement-level ppg per position
    adp_model: str = "premium"
    te_premium: bool = True

    @property
    def size(self):
        return len(self.name)

    def index_of(self, player_name):
        """Pool index of a player by name, or None. For tests and reports only."""
        hits = np.flatnonzero(self.name == player_name)
        return int(hits[0]) if len(hits) else None

    def describe(self, index):
        """One player as a readable string. Reports use this; nothing hot does."""
        return (f"{self.name[index]} ({POSITIONS[self.pos[index]]}"
                f"{'-' + str(self.team[index]) if self.team[index] else ''}, "
                f"{self.ppg[index]:.1f} PPG, ADP {self.adp[index]:.1f})")


def _read_aux(path):
    """{name: (bye_week, raw_espn_adp)} straight from the rankings export.

    Two fields ``read_players`` cannot hand back: the bye week has nowhere to live
    in its Player tuple, and its ADP has already been shifted by the premium by the
    time it returns. Both are read here from the same file, so there is no second
    source of truth -- only a second pass over the one source.

    The export opens with a title line and a blank line before the real header, and
    which line that is has changed before, so the header is found rather than
    assumed.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lines = fh.read().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("Overall Rank,")), 2)
    out = {}
    for row in csv.DictReader(lines[start:]):
        name = (row.get("Full Name") or "").strip()
        if not name:
            continue
        try:
            bye = int(float(row.get("Bye Week")))
        except (TypeError, ValueError):
            bye = 0
        try:
            espn_adp = float(row.get("ADP"))
        except (TypeError, ValueError):
            espn_adp = None
        out[name] = (bye, espn_adp)
    return out


def _replacement_levels(by_position, league):
    """Replacement-level PPG per position, as an array in POSITIONS order.

    Replacement is the last player at a position who starts *somewhere* in the
    league, which is what makes value comparable across positions: a position you
    start more of runs out sooner. The depths come from
    ``draft_tiers.starter_depths`` so that this agrees with the board, including
    the part that matters most in this format -- tight ends winning flex slots on
    merit rather than by an assumed split, which pushes TE replacement well past
    the usual TE12.

    A pool too thin to reach replacement at some position (only possible with a
    hand-built test pool) falls back to the worst player there rather than
    raising: a shallow pool is a smaller league, not an error.
    """
    lineup = {POSITIONS[i]: n for i, n in enumerate(league.starters) if n}
    depths = draft_tiers.starter_depths(by_position, league.teams, lineup, league.flex)
    levels = np.zeros(len(POSITIONS), dtype=np.float64)
    for i, position in enumerate(POSITIONS):
        players = sorted(by_position.get(position, []), key=lambda p: -p.ppg)
        if not players:
            continue
        depth = depths.get(position, 0) or 1
        levels[i] = players[min(depth, len(players)) - 1].ppg
    return levels


def load_pool(rankings=None, projections=None, league=DEFAULT_LEAGUE,
              adp_model="premium", te_premium=True, pool_skill=0):
    """Build the pool. See ``Pool`` for what the arrays mean.

    ``pool_skill`` caps the skill-position pool by overall rank; 0 means no cap,
    which is the default. It used to default to 300, as a speed optimisation on the
    reasoning that a 12-team draft takes at most 192 players and the tail is pure cost
    in every per-pick scan.

    That became a correctness bug the moment the waiver wire existed. The wire IS the
    undrafted tail: capping at rank 300 left only 68 skill players unclaimed across
    four positions after a draft, so the tight end wire came out at 0.0 PPG -- the
    model was saying a manager whose only tight end is on bye can find literally
    nobody, when the real export lists 64 tight ends and reality lists hundreds. The
    full pool costs about 16% more per draft and is what makes the streaming levels
    mean anything.

    ``adp_model``:
      ``premium`` uses the ADP ``read_players`` produces, in which the premium has
      already moved the market's clock. It is the room this user is drafting in.
      ``espn`` uses the export's untouched ESPN ADP -- a public standard-scoring
      room. Note that PPG is premium-priced either way, because the league scores
      that way regardless of what the room believes; only the *timing* changes.
    """
    rankings = rankings or DEFAULT_RANKINGS
    projections = projections or DEFAULT_PROJECTIONS

    receptions = None
    if te_premium and projections and os.path.exists(projections):
        receptions = draft_tiers.read_receptions(projections)

    lineup = {POSITIONS[i]: n for i, n in enumerate(league.starters) if n}
    # limit=0 is falsy inside read_players, which is how it is told "no cap": the
    # kickers and defenses start at overall rank 193 and the board's own default of
    # 250 would keep an arbitrary 57 of them.
    by_position = draft_tiers.read_players(
        rankings, limit=0, receptions=receptions,
        te_premium=draft_tiers.TE_PREMIUM if te_premium else 0.0,
        teams=league.teams, lineup=lineup, flex=league.flex)

    aux = _read_aux(rankings)
    replacement = _replacement_levels(by_position, league)

    rows = []
    for position, players in by_position.items():
        if position not in POS_INDEX:
            continue
        code = POS_INDEX[position]
        skill = code in SKILL_POSITIONS
        for p in players:
            if skill and pool_skill and p.rank > pool_skill:
                continue
            bye, espn_adp = aux.get(p.name, (0, None))
            # Missing ADP falls back to overall rank, which is read_players' own
            # convention and the right one: ranks run past 500 while picks stop at
            # 192, so rank-as-ADP reads as "nobody drafts him".
            adp = p.adp if adp_model == "premium" else (
                espn_adp if espn_adp is not None else float(p.rank))
            rows.append((p.name, p.team, code, float(p.ppg), float(adp),
                         int(p.rank), int(bye)))

    if not rows:
        raise ValueError(f"no players parsed from {rankings}")

    # The tie-break, folded in before the sort so index order is final. ESPN stops
    # reporting ADP near 171 and dozens of players share that value; ordering them by
    # overall rank is the only defensible answer. The nudge is spread across the whole
    # pool so its total span is ADP_TIEBREAK_SPAN however deep the pool goes -- well
    # inside the 0.1 that separates two genuinely different reported ADPs, so it can
    # only ever break a tie, never reorder a real difference.
    worst_rank = max(row[5] for row in rows) or 1
    keys = [row[4] + ADP_TIEBREAK_SPAN * row[5] / worst_rank for row in rows]
    order = sorted(range(len(rows)), key=keys.__getitem__)
    rows = [rows[i] for i in order]
    keys = [keys[i] for i in order]

    name = np.array([r[0] for r in rows], dtype=object)
    team = np.array([r[1] for r in rows], dtype=object)
    pos = np.array([r[2] for r in rows], dtype=np.int8)
    ppg = np.array([r[3] for r in rows], dtype=np.float64)
    adp = np.array([r[4] for r in rows], dtype=np.float64)
    rank = np.array([r[5] for r in rows], dtype=np.int32)
    bye = np.array([r[6] for r in rows], dtype=np.int8)

    # Per position, indices in descending PPG. This is the hero's shopping list:
    # "best available running back" is a scan down by_pos[RB] for the first index
    # still on the board, which is O(depth) instead of O(pool).
    by_pos = tuple(
        np.array(sorted(np.flatnonzero(pos == code).tolist(),
                        key=lambda i: -ppg[i]), dtype=np.int32)
        for code in range(len(POSITIONS)))

    for code in LATE_ONLY:
        if len(by_pos[code]) < league.teams:
            raise ValueError(
                f"only {len(by_pos[code])} {POSITIONS[code]} in the pool but "
                f"{league.teams} teams each need one")

    return Pool(name=name, team=team, pos=pos, ppg=ppg, adp=adp, rank=rank,
                bye=bye, adp_key=np.array(keys, dtype=np.float64), by_pos=by_pos,
                replacement=replacement, adp_model=adp_model,
                te_premium=bool(te_premium))
