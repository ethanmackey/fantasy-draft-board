#!/usr/bin/env python3
"""Build a tiered fantasy draft board in Google Sheets from a rankings CSV.

Usage:
    python draft_tiers.py --dry-run          # print the tiers, touch nothing
    python draft_tiers.py                    # write / rewrite the sheet
    python draft_tiers.py --setup-auth       # one time, sign in as the Sheets account

Reads a Draft-rankings export (2 preamble lines, then a header row), converts each
player's season projection to points per game over a 17-game season, cuts each
position into tiers where the drop to the next player is unusually large, and
writes one worksheet laid out as side-by-side position blocks.

Built for a tight end premium league: tight ends are re-scored for the extra half
point per catch, and that gain over the replacement tight end moves their PPG,
their ADP and their place in the overall rank together. A checkbox in the control
row switches the whole board back to standard scoring.

The sheet doubles as a live draft board: type a name into the "Drafted" column and
that player drops out of their position's column only, with the rest of the column
sliding up. That is why the visible blocks are FILTER formulas over hidden source
columns rather than static values -- hiding spreadsheet *rows* would hide a row
across all four position blocks at once.

Sheets auth uses the same OAuth-refresh-token approach as the dk_line project,
reading GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REFRESH_TOKEN from a .env.
"""

import argparse
import csv
import datetime
import json
import math
import os
import statistics
import sys
from typing import NamedTuple

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")
# The dk_line project already holds a working refresh token for SHEETS_ACCOUNT;
# fall back to it rather than duplicating the secret into a second .env.
ENV_CANDIDATES = [
    ENV_PATH,
    os.path.join(os.path.dirname(SCRIPT_DIR), "dk_line", ".env"),
]
CLIENT_SECRET_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "client_secret.json"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "dk_line", "client_secret.json"),
]
DEFAULT_CLIENT_SECRET = next(
    (p for p in CLIENT_SECRET_CANDIDATES if os.path.exists(p)), CLIENT_SECRET_CANDIDATES[0]
)

DEFAULT_CSV = os.path.join(SCRIPT_DIR, "Draft-rankings-export-2026 (8-21).csv")
# Component projections sitting next to the rankings export. Used when present,
# purely for its receptions column; absent, the TE premium falls back to an
# estimate. Not required for anything else on the board.
DEFAULT_RECEPTIONS_CSV = os.path.join(SCRIPT_DIR, "projections (8-21).csv")
DEFAULT_SPREADSHEET_NAME = "2026 Draft Tiers"
SHEET_TITLE = "Draft Board"
GAMES = 17
DEFAULT_LIMIT = 250      # only the top N by overall rank make the board

# --------------------------------------------------------------------------
# Tight end premium.
#
# The league pays 1.5 per reception to tight ends and 1.0 to everyone else, so
# every TE is worth an extra 0.5 x receptions over the season and nobody else
# changes at all.
#
# The rankings export gives a season points total but no receptions, so unless a
# receptions column is supplied the count is backed out of the points. For tight
# ends that inversion is unusually well behaved: they have no rushing line and a
# narrow band of yards per catch, so PPR points land near 2.5 x receptions
# (~1 for the catch, ~1.15 for the yards, ~0.35 for the touchdowns). Backing out
# through a constant makes the premium a flat multiplier on TE points -- honest
# about carrying no per-player information, and right on average.
#
# A real receptions column beats it, because the whole point of TE premium is
# the *spread*: a 100-catch tight end gains ~3.0 PPG and a touchdown-dependent
# 50-catch one gains ~1.5, and only real counts can tell them apart.
# PPG comes from the ranking export's own Projected Points column, not from
# re-scoring the raw stat lines in the projections file.
#
# That was tried and reverted. The projections file carries full stat lines, so
# scoring them under an explicit table is possible and looks like the more
# principled option -- but measured against this board it moved 13 players out of
# 250, by a mean of 0.10 PPG, and every one of them moved DOWN because PFF scores
# return yardage and a hand-written table forgets to. The upside was a scoring
# table you could edit; the cost was a second source of truth, 38 ranked players
# with no row in it needing a fallback, and kickers and defences needing a table
# of their own. Not worth it to reproduce numbers that already agreed.
#
# The one house rule this league has that PFF's column does not is the tight end
# premium, and that is applied as a per-player bonus below -- which is also what
# lets the sheet's toggle show both boards at once.
TE_POSITION = "TE"
TE_PREMIUM = 0.5             # points per reception above standard PPR
TE_POINTS_PER_RECEPTION = 2.5
# How a gain in points becomes a move up the board.
#
# Not a constant. The market's exchange rate between points and board position is
# strongly convex -- fitted across this board it runs ~3.4 slots per PPG at the
# top and ~12 by the end -- so any single number is right in one place and wrong
# everywhere else. The old flat 6.0 happened to be close for the top tight ends
# and understated everyone from TE3 down by two to four times.
#
# So the rate is not assumed at all. Fit a monotone value curve to the field --
# what PPG does a player ranked at slot r actually project for -- and invert it: a
# tight end worth `gain` more than he was moves to wherever the field prices that
# value. No free parameter, the convexity falls out, and a gain of exactly zero
# maps to a move of exactly zero by construction.
#
# Monotone (pool-adjacent-violators) rather than a fitted function because none of
# the obvious forms fit: log R2 0.77, sqrt 0.86, linear 0.87, and the local slopes
# are non-monotonic noise. Differentiating that is hopeless; a monotone fit and
# one inversion is not.
FIELD_POSITIONS = ("RB", "WR")   # the comparison class the curve is built from
# Guard for the flat middle of the board, where the field is near-indifferent
# across dozens of slots and inverting the curve can send a 0.2 PPG deficit 30
# slots on nothing but the accident of which two players sit either side.
#
# Bounded on the RATE, not on the distance. A flat slot cap is the wrong shape,
# and the wrongness only became visible once the premium was being applied at
# full strength: a 25-slot cap that never fired on anyone while the gains were a
# third of their true size fired on eight startable tight ends the moment they
# were corrected, and at that point a constant, not the model, was setting the
# position of every tight end that matters.
#
# The actual pathology is regions where value barely changes across many slots,
# so inverting divides by something near zero. That is a statement about the
# implied exchange rate -- slots bought per point of gain -- and it is what to
# bound. The field's own rate runs ~3 slots/PPG at the top of the board and ~12
# across the whole of it; the degenerate stretches run into the hundreds. So a
# large move earned by a large gain passes untouched, which is right, and a large
# move squeezed out of a rounding error does not.
MAX_SLOTS_PER_PPG = 30.0
# Better than bounding the output is not letting the degenerate region into the
# inversion at all. The curve is fitted over the whole board but inverted only
# inside this many slots; past roughly here the field is flat -- its last knots
# run rank 166 to 247 for 1.2 PPG, about 113 slots per point -- so an inversion
# out there is reading rounding error, and it was setting the board position of
# five tight ends via the rate cap. Beyond the boundary the curve is clamped,
# which is what _interpolate already does at both ends. With this in place the
# rate cap should catch nobody; if it starts catching people again, that is a
# signal to look at the curve, not to widen the bound.
CURVE_DOMAIN = 180.0

# ADP IS REPRICED TOO, on its own axis, so the toggle moves the market's timing
# and not just your queue.
#
# The argument for leaving it alone was that ADP answers "will he still be there
# at pick 47", and that is settled by eleven other people reading standard ADP off
# the same sites you are. True in a league that only *scores* tight ends
# differently. It is not true of a league that has been drafting under these rules
# for years: a room in a 1.5-PPR-TE format takes tight ends earlier than a
# standard board says, and a board that insists otherwise is wrong in the
# expensive direction -- it tells you to wait on a player who will be gone.
#
# So the same model runs on the ADP axis. What a premium is worth to a tight end
# in points is measured once (te_value_gains) and then read against two separate
# curves: the rank curve says where your own queue should put him, the ADP curve
# says where the market prices that value in picks. Same inversion, different
# exchange rate, because 250 players span 250 rank slots and only ~170 picks.
#
# What this is NOT is a measurement of your actual room. It is what the market's
# own pricing implies if the market shared your scoring, which is an approximation
# on top of an average. The uncertainty band on the ADP colouring
# (ADP_SAFE/ADP_GONE below) is what carries that, and it is deliberately wide.
#
# The ADP axis needs its own domain, and a much tighter one than rank.
#
# The reason is in the data, not in theory: an export's ADP column tops out. Every
# player the market does not really draft is parked at the same sentinel -- in the
# 2026 export, ~60 players sit between 170.1 and 171.5 -- so past there the axis
# carries no ordering at all. Inside that, the field's own knots go degenerate
# earlier than the rank axis does: 103.7 -> 110.3 buys 0.05 PPG (128 picks/PPG)
# and 116.4 -> 132.9 buys 0.03 (705). Inverting out there reads the sentinel and
# the rounding error, so the curve is fitted and inverted inside this many picks
# only, and a tight end whose ADP is past it does not move. That is round 10 of a
# 12-team draft, and the market has no real opinion about a tight end after it.
ADP_CURVE_DOMAIN = 120.0

# Which tight end counts as replacement: the last one who starts somewhere in the
# league, averaged over his neighbours so a single projection cannot set it.
TE_BASELINE_SPAN = 3
# Converts a gain in points into board slots when the curve is switched off with
# --te-shift-model linear. The curve is the default and this is the fallback.
TE_SLOTS_PER_PPG = 6.0
# Column names the figures might arrive under, in any supplied CSV.
RECEPTION_HEADERS = ("recvReceptions", "Receptions", "Rec", "REC",
                     "Projected Receptions", "Catches")
NAME_HEADERS = ("Full Name", "playerName", "Name", "Player", "Player Name")

# Google Sheets target. The refresh token must belong to this account.
SHEETS_ACCOUNT = "ethanmackey36@gmail.com"
# /auth/drive.file is non-sensitive and enough to create + reopen our own file;
# full /auth/drive is a restricted scope an unverified app cannot use.
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    # Non-sensitive; only so --setup-auth can confirm which account consented.
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]
# Asked for during --setup-auth only. Requesting script.projects on a *refresh* of
# a token that was never granted it fails the whole request with invalid_scope, so
# the consent list and the refresh list have to be kept apart: refreshes ask for
# the base set, and the access token comes back carrying whatever was granted.
SCRIPT_SCOPE = "https://www.googleapis.com/auth/script.projects"
SETUP_SCOPES = GOOGLE_SCOPES + [SCRIPT_SCOPE]

# Display order of the position blocks, left to right.
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
# Which of them actually earn a block. Kickers and defences are parsed like
# everything else so they keep holding their rank slots, and the premium reprice
# shuffles them along with everyone else -- but they do not go on the board. Every
# kicker in the top 250 sits within 0.85 PPG of every other kicker and every
# defence within 1.06, so two full blocks of screen were being spent on a decision
# that does not exist. Dropping them is what gets the board onto one screen.
#
# They used to be described as feeding the tiering's gap signal. They did not,
# usefully: they occupy overall ranks 193-216 and 229-236, one contiguous dead
# zone, so all they ever contributed was a manufactured wall around rank 190. It
# cost nothing, because that is round 16, and it is moot now that tiers are cut on
# points rather than rank gaps.
DEFAULT_BOARD_POSITIONS = ["QB", "RB", "WR", "TE"]

# Tier fills: six hue families, and the palette is four passes of lightness through
# them. What has to be true of a band is that ADJACENT tiers never look alike, and a
# six-family rotation guarantees a repeat is six bands away -- far enough that the
# collapse of a position as it is drafted can never bring two of them together.
#
# Only the first pass is painted now (see TIER_BANDS). The rest is kept because it
# is the palette this was designed around, and because a wider cycle is one constant
# away if six ever proves too few.
TIER_COLORS = [
    "#c6dbef",  # 1  blue
    "#c7e9c0",  # 2  green
    "#fdd0a2",  # 3  orange
    "#dadaeb",  # 4  purple
    "#fcbba1",  # 5  salmon
    "#ccece6",  # 6  teal
    "#9ecae1",  # 7  mid blue
    "#a1d99b",  # 8  mid green
    "#fdae6b",  # 9  deep orange
    "#bcbddc",  # 10 lavender
    "#f4a6a6",  # 11 rose
    "#99d8c9",  # 12 mid teal
    "#d0e3f2",  # 13 pale blue
    "#d9f0d3",  # 14 pale green
    "#ffeda0",  # 15 yellow
    "#e0dced",  # 16 pale purple
    "#fde0ef",  # 17 pink
    "#dff0ec",  # 18 pale teal
    "#7fb3d5",  # 19 deep blue
    "#7fc97f",  # 20 deep green
    "#e7cb94",  # 21 tan
    "#a5a3cf",  # 22 deep lavender
    "#e88f8f",  # 23 deep rose
    "#d9d9d9",  # 24 grey
]
# What the bands are actually painted from: one fill per hue family, cycled, rather
# than one per tier number.
#
# A colour per tier meant 59 conditional format rules across the four blocks -- 21
# at running back alone -- and a rule is evaluated per cell of every range it is
# attached to, on every repaint, over a five-column band. It bought nothing, because
# the tier NUMBER is hidden: a fill never identified a tier, it only separated one
# from the next. Six fills cycled separate them exactly as well, in six rules per
# block instead of up to 21.
#
# Six and not fewer for the reason the palette has six families: a cycle of n makes
# a repeat n bands away, and the block collapses as players are drafted, so two
# bands that share a fill can end up touching. Five would put tier 8 next to tier 13
# whenever the tiers between them empty out first, which is an ordinary thing for
# the middle of a position to do.
#
# What is lost is the lightness cue -- the old palette got heavier the further down
# a position you read -- and the property that a tier number had a fixed colour.
# Nothing read either one: the number is not on the board.
TIER_BANDS = TIER_COLORS[:6]
HEADER_GREY = {"red": 0.827, "green": 0.827, "blue": 0.827}

# Sheet geometry. The position blocks start at column A -- nothing else earns
# screen space. Each block is BLOCK_WIDTH columns plus one spacer column. The
# drafted log lives out in the hidden region on the right: it is state, not
# something to read during a draft.
FIRST_BLOCK_COL = 0      # A
# checkbox | Rank | Player | Tm | PPG | ADP | Tier | Fall | VOR | VONA.
#
# Offsets 1..5 are exactly the hidden source block, in order, because the visible
# body is one FILTER that spills the source across them -- change one and the
# other has to move with it. Tier is the only one still hidden: it reads as a
# colour band, and the number itself would just be another figure competing for
# the eye. VOR and VONA are computed on the board from what the FILTER left, so
# they sit past the spill.
#
# Every number answers a different question, which is why they all earn a column:
# Rank is who PFF says is better, ADP is when he actually goes, PPG is what he
# scores under this league's rules, Fall is how far the market is off the ranking,
# VOR is how far above a startable player he is all season, VONA is what waiting
# one turn costs right now.
#
# Fall is ADP minus Rank, and it is the one column that is pure arbitrage: both of
# its inputs were already on the board and the difference between them never was,
# so the read that actually wins drafts -- who is going later than he is ranked --
# had to be done by eye across two columns forty rows apart. Positive means the
# market is letting him slide; negative means the room reaches. It carries no
# opinion of its own, which is the point: PFF's rank and the market's clock
# disagree, and the size of the disagreement is the opportunity.
#
# Blanked past ADP_SENTINEL. Past there an export's ADP column is a pile of
# identical placeholder values, so the subtraction stops measuring disagreement
# and starts measuring the fact that 250 ranks do not fit in 171 picks.
#
# VOR and VONA disagree, and both are right. VONA prices Josh Allen at +0.5
# because if you wait you get Hurts; VOR prices him at +2.6 because the QB12 is
# genuinely much worse. The first is the question at this pick, the second is the
# question about the season.
# Team earns its column because this board's whole reason for preferring PFF's
# rank to PFF's own points is contingent value -- backup running backs ranked 80
# slots above their projection because they are one injury from a starter's role.
# Without the team beside the name that judgement is illegible: you can see that
# Jadarian Price is ranked at 89 on an 11.5 PPG projection and you cannot see
# *whose* backup he is. The board was showing the conclusion and hiding the
# premise. It is also the only way to read a handcuff or a stack off this thing.
BLOCK_WIDTH = 10
# Blocks butt up against each other -- the border down each block's edge separates
# them, and a spacer column per position costs a position's worth of screen.
BLOCK_STRIDE = BLOCK_WIDTH
SOURCE_WIDTH = 6         # hidden source: Rank | Player | Tm | PPG | ADP | Tier
# Fixed widths beat auto-fit here: four positions have to share one screen, and
# auto-fit sizes Player to the longest name in the column ("Los Angeles Rams DST").
# Only the visible columns are listed; the hidden one keeps whatever width.
# Trimmed again to absorb Fall: Player gives up 6px, which is still wider than
# every name on the board.
COLUMN_WIDTHS = [26, 34, 98, 30, 40, 42, 36, 38, 40]
VISIBLE_OFFSETS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
TEAM_COL = 3             # block offset of the team abbreviation
PPG_COL = 4              # block offset of the points-per-game column
ADP_COL = 5              # block offset of the average draft position
TIER_COL = 6             # block offset of the hidden tier column
FALL_COL = 7             # block offset of ADP-minus-Rank
VOR_COL = 8              # block offset of season-long value over replacement
VAL_COL = 9              # block offset of the value-over-next-available column
# Past this ADP an export has stopped ordering players and started parking them:
# in the 2026 file ~60 players sit between 170.1 and 171.5. Fall is blanked there
# rather than reporting the placeholder as a 60-pick slide.
ADP_SENTINEL = 165.0
CONTROL_ROW = 1          # 0-indexed: row 2 in the UI, the draft-size inputs
# The controls live in the first block's columns, so a control's width is that
# column's width and there is no per-cell override in Sheets: widening the cell
# that holds "Teams" would widen Player, or Rank, all the way down the board.
#
# So the labels are placed in columns that are already wide enough and
# right-aligned to hug their input, and the inputs go in the narrow columns beside
# them -- a number needs 30px, a word does not. "Teams" was in the 26px checkbox
# column and rendered as a clipped "Te", because the input to its right is
# non-empty and Sheets will not overflow text into an occupied cell.
#
# The alternative, if a label ever outgrows every column, is to merge cells across
# row 2 only: a merge is a property of the range, not of the columns, so it cannot
# disturb anything below. Wider inputs are the one thing that needs it.
TEAMS_LABEL_COL = 2      # C2: the Player column, the widest one on the board
CONTROL_INPUT_COL = 3    # D2 holds teams -- the Tm column, wide enough for a number
# Your pick in the draft order -- "Pick" on the board, 1 to Teams: label in E2,
# input in the next VISIBLE column after it. It once sat squarely in the first
# block's hidden Tier column, so the input was invisible while its own label sat
# beside it in plain sight, and the cell you would naturally type into instead was
# wired to nothing.
#
# It is seeded from --pick (default 1) rather than left blank, because a blank cell
# does not error; it falls back to a derivation that is wrong by up to a full round
# in the direction that says players will still be there when they are about to go.
# The board looked configured and was quietly guessing.
#
# Note the two senses of "pick" the board has to keep apart: this one is the same
# every round (which of the twelve you are), and the one in the status line is the
# running pick number of the draft. MY TEAM's own "Slot" column is a third thing --
# a lineup slot, QB or FLEX or BN.
#
# Any control column added here must be checked against the hidden ones: the
# blocks own columns 0..BLOCK_WIDTH-1 and hide TIER_COL out of the middle of them.
PICK_LABEL_COL = CONTROL_INPUT_COL + 1
# Derived, not counted out by hand: the next column along unless that is the
# hidden one. Counting by hand is what put it in a hidden column the first time,
# and the block width has already changed once since.
PICK_INPUT_COL = (PICK_LABEL_COL + 2 if PICK_LABEL_COL + 1 == TIER_COL
                  else PICK_LABEL_COL + 1)
# The TE premium toggle goes in the first block's VONA column: the visible columns
# left of it are all spoken for by the controls, and a control nobody can see is
# not a control. The status string then starts one column later and points back at
# it, the same way the banner points at the reset box in A1.
TOGGLE_COL = BLOCK_WIDTH - 1     # I2 at the current block width
STATUS_COL = BLOCK_WIDTH         # J2 onward, merged
# The on-the-clock flag: A2:B2, the two cells in the control row nothing else wants.
#
# It is a cell of its own because a highlight in Sheets is a property of the cell,
# not of a run of text inside it -- there is no way to colour two words of a
# formula's output. Painting the whole status strip amber to light up "YOUR PICK"
# lit up the pick number, the round, the horizon and the TE premium pointer with it.
#
# Blank when it is not your turn, so the 11 picks out of 12 where this cell says
# nothing cost nothing: no gap in a sentence, no colour, no furniture. Sized to the
# words rather than to a block, which is the other half of "just the text".
FLAG_COL = 0
FLAG_END = 2                     # merged A2:B2 -- 60px at the current widths
HEADER_ROW = 3           # 0-indexed: row 4 in the UI
FIRST_DATA_ROW = 4       # 0-indexed: row 5 in the UI
# Rows of the drafted log the FILTER formulas watch. Every block re-runs a COUNTIF
# of its source against this whole range on every tick, so the constant is pure
# recalc cost past the point a draft can reach: 12 teams x 18 rounds is 216.
DRAFTED_LIMIT = 220

# Draft timing. Value is measured against one full round from the pick on the
# clock -- the realistic next chance at a position for whoever is picking, with
# nobody's own pick involved. A player is treated as gone by then if their ADP falls
# before that pick; ADP is a mean, so that is the 50/50 line.
DEFAULT_TEAMS = 12
# Your pick in the order, seeded into the Pick cell so the board is never shipped
# unconfigured.
DEFAULT_PICK = 1
# Replacement is the mean of this many best survivors, not the single best. One
# player is a thin thing to price a whole position against -- if he goes early,
# every value behind him was wrong.
REPLACEMENT_POOL = 3
# ADP is a mean, and a much noisier one than the old symmetric +/-8 band allowed.
# Real dispersion through rounds 4-10 runs 15-25 picks either side, so "ADP is 9
# past your turn" is not safety, it is a coin flip being coloured as a certainty.
#
# The band is deliberately asymmetric now, because the two errors are not equally
# expensive. Calling a player safe when he is gone costs you the player; calling
# him a coin flip when he would have lasted costs you nothing but a moment's
# thought. So green has to clear the horizon by a lot, and amber reaches back
# behind it -- a player whose ADP is slightly *before* your pick is still a live
# possibility, which a hard line at the horizon denied.
# Scaled by the number of picks that actually elapse before your turn, with a
# floor so a one-pick gap is never called a certainty. The rates are per
# intervening pick: over ~20 picks that reproduces the old +20/-10 band, and over
# two picks it collapses to the floors instead of painting the whole board amber.
ADP_SAFE_RATE = 1.2
ADP_SAFE_FLOOR = 6.0
ADP_GONE_RATE = 0.6
ADP_GONE_FLOOR = 4.0

# Starting lineup: QB, RB, RB, WR, WR, TE, FLEX, FLEX -- flex takes anything but
# a quarterback. Replacement at a position is the last player there who starts
# somewhere in the league, and that is what makes value comparable across
# positions: a position you start more of runs out sooner.
LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_SLOTS = 2


def set_lineup(lineup, flex, bench=None):
    """Install the league's lineup as the module's, once, at startup.

    --starters and --flex used to reach only starter_depths. Everything that
    draws the roster panel -- its slot labels, the caps that decide who is a
    starter, the need gate behind the best-value highlight -- read the module
    constants instead, so a command line that asked for three receivers still got
    a two-receiver panel and a need gate that thought it was full. One league per
    run, so one place to say what the league is.
    """
    global LINEUP, FLEX_SLOTS, ROSTER_BENCH_ROWS
    # Whole slots. The panel draws one labelled row per mandatory slot and the
    # need gate counts them, neither of which means anything at RB:2.5.
    LINEUP = {position: int(round(count)) for position, count in lineup.items()}
    FLEX_SLOTS = int(round(flex))
    if bench is not None:
        ROSTER_BENCH_ROWS = max(0, int(round(bench)))
FLEX_POSITIONS = ("RB", "WR", "TE")
# The flex is deliberately NOT split by a fixed share. Who fills it is decided on
# merit from the projections, which matters enormously here: tight ends are flex
# eligible, and a premium league is exactly the format where they win those slots.
# A fixed share would have pinned tight end replacement at TE12 when the real
# answer on this data is far deeper -- and the deeper it goes, the larger every
# premium gain, because replacement gained less of the premium than the top did.
# Val colouring saturates at +/- this many points per game.
VALUE_SCALE = 5

# --------------------------------------------------------------------------
# My team.
#
# The roster panel is derived entirely from the drafted log and the Pick cell;
# nothing about it is baked in at build time. The log is already an ordered
# record of the draft -- row n of it is pick n -- so which of those picks are
# YOURS is pure snake arithmetic against that cell, and the whole panel recomputes
# the moment it changes. That matters because your place in the order changes from
# draft to draft: a panel that knew it at build time would need a rebuild every
# league, and a rebuild is the one thing you cannot do mid-draft.
#
# The cost of deriving it is the assumption the rest of the board already makes:
# every pick gets logged, in order. Skip one and the parity of every later round
# is wrong -- and since the log is also the clock, the fix is to tick the pick that
# was missed rather than to tell the board a different number.
ROSTER_WIDTH = 4          # Slot | Player | Pos | PPG
# Starters + this many, so the panel is as deep as the league's roster and no
# deeper: a pick past the last row is logged, counted in the banner's "N picks",
# and then shown nowhere. Set with --bench.
ROSTER_BENCH_ROWS = 6
# The panel is the rightmost thing on the board, so its widths answer to nothing
# but themselves: the merged MY TEAM banner across all four is the binding
# constraint, not the names under them.
ROSTER_COLUMN_WIDTHS = [46, 128, 34, 48]
ROSTER_STAGE_ROWS = 60    # rows of hidden staging -- far past any real roster
# Targets: the players you want, typed in before the draft starts. Every paper
# cheat sheet ever made has had circled names on it and this board had nowhere to
# put them -- the checkbox column can only say "gone", never "mine if he falls".
# A name here lights up wherever that player sits on the board.
TARGET_ROWS = 10
# Positional runs. The urgency banner measures a static cliff; a run is the other
# thing that strips a position out from under you, and the drafted log already
# holds the ordered data to see one coming. Called out only when it is actually
# hot, because a badge that is always on is furniture.
RUN_WINDOW = 10
RUN_ALERT = 4
# Hidden bookkeeping past the position sources: the name lookup the panel and the
# drafted dropdown share, then the roster staging columns.
LOOKUP_WIDTH = 3          # All Players | Pos | PPG
# Staging, in order: my picks, their position, PPG, a tie-broken sort key, rank
# within position, flex eligibility, flex rank, bench flag, bench rank.
ROSTER_STAGE_WIDTH = 9


class Player(NamedTuple):
    """One player as parsed, before tiering.

    Named rather than a bare tuple because the premium reprices some fields and
    not others, and every consumer indexed by position: adding the bye week in
    the middle of a 4-tuple would have silently shifted ADP into PPG's place in
    six call sites. `bonus` is the tight end premium's per-player gain, kept only
    long enough to price the rank shift off it.
    """
    rank: int
    name: str
    ppg: float
    adp: float
    bonus: float = 0.0
    team: str = ""


class Row(NamedTuple):
    """A player with their tier, as written to the sheet."""
    tier: int
    name: str
    rank: int
    ppg: float
    adp: float
    team: str = ""


# --------------------------------------------------------------------------
# CSV -> per-position player lists
# --------------------------------------------------------------------------

def column_letter(index):
    """0 -> 'A', 26 -> 'AA'."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _first_key(row, candidates):
    """The first of `candidates` present in `row`, or None."""
    return next((key for key in candidates if key in row), None)


def read_receptions(path):
    """Load {player name: projected receptions} from any CSV that carries both.

    Deliberately forgiving about shape: it hunts for the header row rather than
    assuming one, and accepts whichever of the usual name/receptions column
    labels it finds. Any projections export with a receptions column will do.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lines = fh.read().splitlines()

    start = next(
        (i for i, line in enumerate(lines)
         if any(h in line for h in NAME_HEADERS) and any(h in line for h in RECEPTION_HEADERS)),
        None)
    if start is None:
        raise SystemExit(f"{path}: found no header row with both a name and a receptions column")

    reader = csv.DictReader(lines[start:])
    name_key = _first_key(reader.fieldnames or [], NAME_HEADERS)
    rec_key = _first_key(reader.fieldnames or [], RECEPTION_HEADERS)

    receptions = {}
    for row in reader:
        name = (row.get(name_key) or "").strip()
        try:
            receptions[name] = float(row.get(rec_key))
        except (TypeError, ValueError):
            continue
    print(f"receptions: {len(receptions)} player(s) from {os.path.basename(path)}",
          file=sys.stderr)
    return receptions


def starter_depths(by_position, teams=DEFAULT_TEAMS, lineup=None, flex=FLEX_SLOTS):
    """{position: how many of them start somewhere in the league}.

    Mandatory slots are filled first, then the flex spots go to whoever is best
    among what is left of the flex-eligible positions -- resolved on projected
    points rather than by assuming a split. The last starter at each position is
    replacement level for that position.

    Run against premium-adjusted points, so in this format tight ends win flex
    slots they would never win in a standard league, which is the correct reason
    for tight end replacement to sit deeper here than the usual TE12.
    """
    lineup = LINEUP if lineup is None else lineup
    depths, bench = {}, []
    for position, per_team in lineup.items():
        players = sorted(by_position.get(position, []), key=lambda p: -p.ppg)
        mandatory = min(int(round(teams * per_team)), len(players))
        depths[position] = mandatory
        if position in FLEX_POSITIONS:
            bench.extend((p.ppg, position) for p in players[mandatory:])

    bench.sort(key=lambda item: -item[0])
    for _, position in bench[:max(0, int(round(teams * flex)))]:
        depths[position] += 1
    return depths


def _isotonic_decreasing(values):
    """Pool adjacent violators: the closest decreasing series to `values`.

    No functional form, no smoothing parameter. Where the data really is flat the
    fit is flat; where it drops it drops.
    """
    levels = [[v, 1] for v in values]
    i = 0
    while i < len(levels) - 1:
        if levels[i][0] / levels[i][1] < levels[i + 1][0] / levels[i + 1][1]:
            levels[i][0] += levels[i + 1][0]
            levels[i][1] += levels[i + 1][1]
            levels.pop(i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    return [total / n for total, n in levels for _ in range(n)]


def value_curve(points):
    """[(position on the board, ppg)] -> knots of a strictly decreasing curve.

    `position` is whatever axis the move is measured on -- ADP for the ADP
    reprice, overall rank for the rank reprice. Plateaus in the monotone fit are
    collapsed to their mean position so the curve can be inverted.
    """
    points = sorted(points)
    if len(points) < 2:
        return [(p[0], p[1]) for p in points]
    xs = [x for x, _ in points]
    fitted = _isotonic_decreasing([v for _, v in points])

    knots, i = [], 0
    while i < len(fitted):
        j = i
        while j + 1 < len(fitted) and fitted[j + 1] == fitted[i]:
            j += 1
        knots.append((statistics.fmean(xs[i:j + 1]), fitted[i]))
        i = j + 1
    return knots


def _interpolate(knots, key, want, other):
    """Piecewise-linear lookup along `key`, returning `other`. Clamped at both ends."""
    if not knots:
        return None
    lo, hi = knots[0], knots[-1]
    if (want - lo[key]) * (1 if key == 0 else -1) <= 0:
        return lo[other]
    if (want - hi[key]) * (1 if key == 0 else -1) >= 0:
        return hi[other]
    for a, b in zip(knots, knots[1:]):
        if min(a[key], b[key]) <= want <= max(a[key], b[key]):
            span = b[key] - a[key]
            if span == 0:
                return a[other]
            return a[other] + (b[other] - a[other]) * (want - a[key]) / span
    return hi[other]


def curve_move(knots, position, gain):
    """Where the field prices a player currently at `position` once worth `gain` more.

    The whole model in two lookups: read his implied value off the curve, add the
    gain, read back the position that value commands. A gain of zero returns the
    position it was given, exactly.
    """
    if not knots or not gain:
        return position
    value = _interpolate(knots, 0, position, 1)
    return _interpolate(knots, 1, value + gain, 0)


def _replacement_level(players, value, depth):
    """(mean value at `depth`, the window it was averaged over).

    Averaged over TE_BASELINE_SPAN neighbours so one projection cannot set a
    replacement level the whole board is priced against.
    """
    ranked = sorted(players, key=lambda p: -value(p))
    index = min(max(depth - 1, 0), len(ranked) - 1)
    low = max(0, index - TE_BASELINE_SPAN // 2)
    window = ranked[low:low + TE_BASELINE_SPAN] or ranked[index:index + 1]
    # The player AT the depth, not the last one in the smoothing window. The log
    # line used to name window[-1] and so reported the TE19 as the replacement in
    # a league whose replacement is the TE18.
    return statistics.fmean(value(p) for p in window), window, ranked[index]


def te_value_gains(tight_ends, depth, standard_depth):
    """Return ({name: PPG a tight end gains on the field}, replacement move, window).

    What the premium is actually worth to a tight end is the change in his value
    over replacement between the two formats -- because value over replacement is
    what makes a tight end comparable to a running back at all, and it is the only
    thing that moves him past one on a cross-positional board:

        gain = (ppg + bonus - replacement_premium) - (ppg - replacement_standard)
             =  bonus - (replacement_premium - replacement_standard)

    So the quantity that comes off the bonus is how much the REPLACEMENT LEVEL
    moved, not how much bonus the replacement tight end happened to collect.

    Those are wildly different numbers and the difference is not academic. This
    code used to subtract the replacement's own bonus -- 1.87 PPG on this data --
    when the replacement level moved 0.28. Standard scoring starts about one tight
    end per team, premium scoring starts eighteen of them, and the TE18 is 1.59
    PPG worse than the TE12 before either of them collects a single premium point.
    Subtracting the bonus silently charged every tight end for that, and the board
    ended up applying the premium at roughly a third strength: McBride got 44% of
    what he was owed, Kraft 26%, Kelce 19%.

    The sanity checks both still hold. If every tight end gained the same bonus and
    replacement stayed put, gain is zero for everyone and nobody moves -- correct,
    because a premium every tight end collects equally changes no decision. And a
    touchdown-dependent tight end whose bonus is below the replacement move gets a
    negative gain and slides down, which is also right: he is relatively worse in
    this format.

    Deeper replacement now *raises* every gain, which is the direction the format
    actually works in -- the further down the tight end pool your league forces
    you, the more the ones at the top are worth.

    This is points, not slots. Turning it into a move is the market curve's job,
    and it converts at a different rate at every depth.
    """
    if not tight_ends:
        return {}, 0.0, []
    premium, _, at = _replacement_level(tight_ends, lambda p: p.ppg, depth)
    standard, _, _ = _replacement_level(
        tight_ends, lambda p: p.ppg - p.bonus, standard_depth)
    moved = premium - standard
    gains = {(TE_POSITION, p.name): p.bonus - moved for p in tight_ends}
    return gains, moved, at


def reprice(by_position, gains, axis="rank", knots=None,
            slots_per_ppg=TE_SLOTS_PER_PPG, domain=CURVE_DOMAIN):
    """Re-order the whole board by the tight ends' gains, on one axis. In place.

    One function, two axes, because it is one model: `axis` is the Player field
    being moved ("rank" or "adp") and `knots` is a curve fitted to that same axis.
    Rank and ADP are not the same scale -- 250 players span 250 rank slots but only
    ~170 picks of ADP, and they compress differently with depth -- so a shift
    measured in picks cannot be reused as a shift in slots. Each axis gets its own
    curve, its own inversion and its own domain; what is shared is the gain in
    points, which is the thing the premium actually changes.

    Rank is your queue: whether to take this tight end before that running back.
    ADP is the market's clock: whether he survives to your next pick. The premium
    moves both, and it does not move them by the same amount.

    Every position is in the pool on both axes, not just the tight ends. On the
    rank axis that is obvious -- a queue cannot make room for one player without
    the rest shuffling back. On the ADP axis it is what keeps the board's arithmetic
    honest: shifting twelve tight ends 30 picks earlier and leaving everyone else
    alone would have 66 players expected gone in the first 60 picks, and the
    survivor pool behind VONA and the green/amber colouring both count exactly that.
    Someone has to be pushed later for a tight end to be taken earlier.

    The new numbers are the *same set* of numbers, dealt out in the new order,
    rather than a fresh 1..N. Rows drop out upstream -- past the limit, or with
    nothing parseable in them -- so a fresh 1..N would quietly renumber the entire
    board around a single skipped row, and the premium-off build would stop
    matching the export it came from. On the ADP axis the re-deal does a second job:
    it preserves the *distribution* of pick numbers, sentinel pile-up included, so
    nothing downstream can start counting more players than there are picks.
    """
    if not gains:
        return
    # Only for the log lines, but they are the only view of this step there is, and
    # a shift on the ADP axis reported in "slots" reads as a rank move.
    label, verb, unit = (("ranks", "re-slotted", "slot") if axis == "rank"
                         else ("ADP", "re-timed", "pick"))
    board = []
    capped = []
    for position, players in by_position.items():
        for i, player in enumerate(players):
            # Keyed on (position, name), not the bare name. `standard` is keyed
            # this way for exactly the same reason: two players can share a name,
            # and applying a tight end's premium gain to a running back would be
            # invisible in the output and wrong in the draft.
            gain = gains.get((position, player.name), 0.0)
            here = float(getattr(player, axis))
            if not gain:
                target = here
            else:
                if knots and here > domain:
                    # Off the end of the curve, so the curve has nothing to say
                    # about him. Clamping the INPUT would read his value off the
                    # boundary knot and then invert it back to somewhere inside
                    # the domain -- a 238th-ranked tight end landing at 150 on a
                    # gain the model never actually measured for him. Leaving him
                    # where he is says the honest thing: past here the board is
                    # flat, and nothing at the bottom of it is a decision.
                    shift = 0.0
                elif knots:
                    shift = curve_move(knots, here, gain) - here
                else:
                    shift = -gain * slots_per_ppg
                limit = abs(gain) * MAX_SLOTS_PER_PPG
                if abs(shift) > limit:
                    capped.append(f"{player.name} "
                                  f"({abs(shift) / abs(gain):.0f} {unit}s/PPG)")
                    shift = math.copysign(limit, shift)
                target = here + shift
            # Ties break on the original position, so a tight end landing exactly on
            # a slot someone already holds goes in behind him.
            board.append((target, here, position, i))
    board.sort()

    slots = sorted(item[1] for item in board)
    moved = 0
    biggest = None
    for slot, (_, was, position, i) in zip(slots, board):
        # The dealt value has to go back on the axis it came off, and in the type
        # it came off it: rank is an int on the export and formats as "#47" on the
        # board, ADP is a float. float(getattr(...)) above is only so the sort and
        # the arithmetic agree.
        if axis == "rank":
            slot = int(round(slot))
        by_position[position][i] = by_position[position][i]._replace(**{axis: slot})
        if slot != was:
            moved += 1
            if biggest is None or abs(slot - was) > abs(biggest[1]):
                biggest = (by_position[position][i].name, slot - was)

    if moved:
        # was - slot, not slot - was: a smaller number is a move UP the board.
        # Printed the other way round this line reported Cade Otton, who dropped 22
        # slots, as having moved 22 slots earlier.
        way = "earlier" if biggest[1] < 0 else "later"
        size = (f"{abs(biggest[1]):.0f}" if axis == "rank"
                else f"{abs(biggest[1]):.1f}")
        print(f"{label}: {moved} player(s) {verb} around the premium; biggest move "
              f"{biggest[0]} {size} {unit}(s) {way}", file=sys.stderr)
    if capped:
        # Named rather than counted, with the rate that tripped it, so it is
        # obvious whether the guard caught a genuine artefact or is quietly
        # setting somebody's position for him.
        print(f"  rate-capped at {MAX_SLOTS_PER_PPG:g} {unit}s/PPG: "
              f"{', '.join(capped[:6])}{' ...' if len(capped) > 6 else ''}",
              file=sys.stderr)


def read_players(path, games=GAMES, limit=DEFAULT_LIMIT, receptions=None,
                 te_premium=TE_PREMIUM, te_points_per_reception=TE_POINTS_PER_RECEPTION,
                 te_slots_per_ppg=TE_SLOTS_PER_PPG, teams=None,
                 te_shift_model="curve", te_starters=None, lineup=None,
                 flex=FLEX_SLOTS):
    """Return {position: [(overall_rank, name, ppg, adp), ...]} sorted by rank.

    Rank drives the board -- order and tiers both. PPG rides along as a second
    opinion to read next to a name, not as an input to either. ADP is a third
    thing again: it is the market's timing, and it is what says whether a player
    survives to your next pick. Missing ADP falls back to the overall rank.

    Tight ends are re-scored for the league's 1.5 PPR: their season total gains
    `te_premium` x receptions, taken from `receptions` where that name is known
    and estimated from their points otherwise. What that gains them against the
    field then moves their overall RANK and their ADP, on separate curves, with the
    board re-numbered around each move so the players a tight end stepped over give
    up exactly the places he took. No other position's points change -- but other
    positions' ranks and ADPs do, by a slot or two, because neither a queue nor a
    draft can make room for one player without the rest shuffling back.

    ADP moves because the room has been drafting this format for years: a premium
    league takes tight ends earlier than a standard board says, and the timing is
    what the board is for. It is an approximation of the market under your scoring,
    not a measurement of your room -- see ADP_CURVE_DOMAIN.

    With the premium off, every shift is zero and the board is the export's,
    untouched. That is what the sheet's toggle switches back to.

    Only the top `limit` players by overall rank. Everything past that is draft
    filler, and it actively distorts the tiering: 20 WRs projected at exactly
    0.00 PPG form a cluster nothing can split, which drags tier boundaries away
    from the part of the board that gets drafted.

    The export starts with a title line and a blank line before the real header,
    so the first two lines are dropped before handing the rest to DictReader.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        lines = fh.read().splitlines()

    # Find the header row rather than assuming it is always line 3.
    start = next((i for i, line in enumerate(lines) if line.startswith("Overall Rank,")), 2)
    reader = csv.DictReader(lines[start:])
    # A receptions column in the rankings export itself beats anything estimated,
    # so prefer it over --receptions and over the fallback.
    own_rec_key = _first_key(reader.fieldnames or [], RECEPTION_HEADERS)
    receptions = receptions or {}

    by_position = {}
    skipped = 0
    estimated = []
    for row in reader:
        position = (row.get("Position") or "").strip()
        name = (row.get("Full Name") or "").strip()
        try:
            projected = float(row.get("Projected Points"))
            overall = int(row.get("Overall Rank"))
        except (TypeError, ValueError):
            skipped += 1
            continue
        try:
            adp = float(row.get("ADP"))
        except (TypeError, ValueError):
            adp = float(overall)
        if not position or not name:
            skipped += 1
            continue
        if limit and overall > limit:
            continue

        bonus = 0.0
        if position == TE_POSITION and te_premium:
            caught = None
            if own_rec_key:
                try:
                    caught = float(row.get(own_rec_key))
                except (TypeError, ValueError):
                    caught = None
            if caught is None:
                caught = receptions.get(name)
            if caught is None:
                caught = projected / te_points_per_reception
                estimated.append(name)
            bonus = te_premium * caught / games

        by_position.setdefault(position, []).append(Player(
            rank=overall, name=name, ppg=round(projected / games + bonus, 2),
            adp=adp, bonus=bonus,
            team=(row.get("Team Abbreviation") or row.get("Team") or "").strip()))

    if estimated and te_premium:
        print(f"TE premium: {len(estimated)} tight end(s) with no receptions figure, "
              f"estimated at points/{te_points_per_reception}: "
              f"{', '.join(sorted(estimated)[:6])}"
              f"{' ...' if len(estimated) > 6 else ''}", file=sys.stderr)

    if te_premium:
        tight_ends = by_position.get(TE_POSITION)
        # Two replacement levels, because the gain is the difference between them.
        #
        # Premium depths come from premium-adjusted points: the bonus is already on
        # every tight end by now, so any flex slot a tight end wins here he wins on
        # this league's actual scoring. Standard depths come from the same lineup
        # run against the same players with the bonus taken back off -- which is
        # the format the market's ADP and PFF's rank were built for, and the
        # baseline the premium is being measured against.
        depths = starter_depths(by_position, teams or DEFAULT_TEAMS, lineup, flex)
        plain = {position: [p._replace(ppg=round(p.ppg - p.bonus, 2)) for p in players]
                 for position, players in by_position.items()}
        standard_depths = starter_depths(plain, teams or DEFAULT_TEAMS, lineup, flex)
        te_depth = te_starters or depths.get(TE_POSITION)
        std_depth = standard_depths.get(TE_POSITION)
        gains, moved, replacement = te_value_gains(tight_ends, te_depth, std_depth)
        print("starters: " + ", ".join(
            f"{pos}{depths[pos]}" for pos in ("QB", "RB", "WR", "TE") if pos in depths)
            + f"  (replacement TE = TE{te_depth}, TE{std_depth} at standard scoring)",
            file=sys.stderr)
        best = max(tight_ends, key=lambda p: gains[(TE_POSITION, p.name)])
        print(f"TE premium: replacement level moved {moved:+.2f} PPG "
              f"(TE{std_depth} -> TE{te_depth}, {replacement.name}); "
              f"biggest gain {best.name} "
              f"{gains[(TE_POSITION, best.name)]:+.2f} PPG on the field",
              file=sys.stderr)
        # Both curves are fitted to the field at *standard* scoring, which is what
        # these players' own PPG still is: the premium only ever touched the tight
        # ends, and they are deliberately not in the field. Fitting the board the
        # tight ends are moving through, not the board after they have moved.
        #
        # Two fits, one per axis, each over its own domain -- the same points priced
        # by where they sit in PFF's queue and by when the market actually takes
        # them. Both are built before either reprice runs, because a reprice mutates
        # the axis the next fit would read.
        rank_knots = adp_knots = None
        if te_shift_model == "curve":
            field = [p for pos in FIELD_POSITIONS for p in by_position.get(pos, [])
                     if p.adp >= 1]
            ranked = [(float(p.rank), p.ppg) for p in field if p.rank <= CURVE_DOMAIN]
            timed = [(p.adp, p.ppg) for p in field if p.adp <= ADP_CURVE_DOMAIN]
            if len(ranked) >= 20:
                rank_knots = value_curve(ranked)
            if len(timed) >= 20:
                adp_knots = value_curve(timed)
        reprice(by_position, gains, "rank", rank_knots, te_slots_per_ppg,
                CURVE_DOMAIN)
        # And the market's clock, so the toggle moves "will he be there at my next
        # pick" and not only "should I want him". Everything timing-related on the
        # sheet reads this: the green/amber ADP band, the VONA survivor pool, the
        # cost-of-waiting strip.
        reprice(by_position, gains, "adp", adp_knots, te_slots_per_ppg,
                ADP_CURVE_DOMAIN)

    if skipped:
        print(f"skipped {skipped} row(s) with no name/position/projection", file=sys.stderr)

    for players in by_position.values():
        players.sort(key=lambda p: p.rank)
    return by_position


# --------------------------------------------------------------------------
# Tiers, from the gaps in PFF's overall rank -- after the premium has moved the
# tight ends through it.
#
# Rank is the backbone of this board, and deliberately not a re-sort of the
# projections. PFF's ranking disagrees with PFF's own projected points by 35
# slots on average across the top 250, and the disagreement is nearly all
# cross-positional: quarterbacks scoring 14+ PPG ranked in the 130s because the
# position is deep, backup running backs ranked 80 slots above their projection
# because they are one injury from a starter's role. That is replacement level and
# contingent value already priced in by people who do this full time, and no
# amount of sorting a points column recovers it.
#
# Which is exactly why the tiers are cut from rank gaps. A position's overall
# ranks are a point process laid over the whole draft board: the WRs run 3, 4, 5,
# 8, 9, 10, 17, and every gap is other positions ranked in between. The gap is
# therefore the signal -- it is PFF saying this is where the board turns to
# something else -- and a tier ends where the position goes quiet for longer than
# the local density says it should.
#
# Sorting a block by points instead was tried and reverted. It cost little inside
# a block (PFF's order and its own points order differ by 1.7 slots at tight end,
# 0.7 at quarterback) and cost the whole cross-positional judgement at the
# boundaries, which is the part worth having. --tier-on ppg is still there to
# compare against; rank is the default and the one the board is built for.
#
# A position's overall ranks are a point process laid over the whole draft
# board: the WRs run 3, 4, 5, 8, 9, 10, 17, and every gap is other positions
# ranked in between. That makes the gap itself the signal -- a tier ends where
# the position goes quiet for longer than the local density says it should.
#
# So a break is a *statistical* judgement, not a clustering one. Model the gaps
# as exponential with a locally estimated rate, and break where the gap lands in
# the tail: P(gap > g) = exp(-g / rate), so g >= rate * ln(1/p) fires at
# significance p.
#
# This replaced exact 1-D k-means on the same ranks, which minimises within-tier
# variance and therefore cannot see a gap as large *for its neighbourhood*. Both
# ways that failed came from the one blind spot:
#
#   * RB ranks run 1, 2, 6, 7, 11..16, 19..24 -- one tight blob by variance, so
#     tier 1 swallowed 15 players. But the local gaps there are 1, which makes
#     the 4-rank jumps real walls, and this test splits them.
#   * TE ranks 27, 29, 47: an 18-rank gap sitting next to a 2-rank gap. Splitting
#     three players barely moves variance, so k-means kept Loveland with McBride
#     and Bowers. Against a local rate of 2, an 18 is off the scale.
#
# The local rate deliberately leaves the candidate gap out of its own baseline.
# Including it is what let that 18 inflate the average it was being judged
# against, which is the specific reason the TE tier came out wrong.
#
# Note this reads the *overall* rank only. Position Rank is 1, 2, 3, ... by
# construction -- perfectly uniform, no gaps, nothing to find.
# --------------------------------------------------------------------------

BREAK_P = 0.08        # a gap this improbable for its neighbourhood ends a tier
GAP_WINDOW = 6        # gaps either side used to estimate the local rate
MIN_GAP = 2           # consecutive ranks never split, whatever the local rate
MAX_GAP = 10          # ... and this many ranks always split, whatever the rate

# The same test, run on PPG drops instead of rank gaps. Rank gaps measure when the
# market takes a position; PPG drops measure what you actually lose by waiting.
# They are not the same board, and in a tight end premium league the rank gaps are
# the wrong one: they come from a standard-scoring consensus, so a tier can end up
# holding a 1.6 PPG spread -- as wide as the drop that opened it.
#
# The floors and ceilings are in points per game rather than rank slots. A tenth
# of a point is inside any projection's error bar and never ends a tier; three
# quarters of a point is a cliff whatever the neighbourhood looks like.
MIN_DROP = 0.10
MAX_DROP = 0.75
# ... and a tighter window than the rank basis uses. Rank gaps are roughly one
# process down the whole column, so a wide neighbourhood estimates it well. PPG
# drops are not: the top of a position is a genuinely different regime, where
# elite players sit 0.4-0.7 apart because they really are that far apart. A window
# of six reaches from the middle of a position back into that spread and sets the
# bar by it -- which is how a 0.74 drop between the RB8 and RB9 came to be judged
# against a bar of 0.78 and swallowed, leaving nine backs and 2.1 PPG in one tier.
PPG_GAP_WINDOW = 3
TIER_BASES = ("rank", "ppg")

# A ceiling on how far apart two players in one tier may be.
#
# The gap test above asks whether a DROP is large for its neighbourhood, which is
# the right question and answers a different one from "is this tier honest". A
# position can decline smoothly, never opening a drop unusual enough to break, and
# accumulate a tier that spans more than the drops the same test calls cliffs:
# before this cap, RB tier 1 held a 1.50 PPG spread (Gibbs 20.18 to Taylor 18.68,
# 26 points of season production) and receiver tiers 6 and 7 held 12 and 13
# players across ~0.95. A tier is a claim that its members are interchangeable,
# and at that width the claim is simply false.
#
# The number is MAX_DROP, deliberately and not by coincidence: this file already
# calls three quarters of a point "a cliff whatever the neighbourhood looks like",
# so a spread that large cannot also be the inside of one tier without the board
# contradicting itself.
#
# Measured across cap values, scoring every same-tier pair further apart than a
# cliff as a lie in one direction and every adjacent-tier pair closer than a coin
# flip (0.35 PPG) as a lie in the other:
#
#     cap    tiers QB/RB/WR/TE   too-far pairs   falsely-split pairs
#     none          9/13/15/11              73                    13
#     0.90          9/18/19/11              14                    25
#     0.75          9/19/20/12               0                    29
#     0.60         10/23/22/12               0                    44
#     0.50         10/23/26/13               0                    68
#
# 0.75 is where the first column reaches zero. Tightening past it buys nothing
# there and pays for it in the second, which is the board calling players
# different who are inside each other's error bars.
TIER_MAX_SPREAD = 0.75
# ... and a too-wide tier is only ever split at a drop worth at least this much.
#
# Without this the rule splits at the largest internal drop whatever its size, and
# in a smooth ramp that manufactures a boundary where the data has none. At a 0.50
# cap it was shaving single players off the ends of groups on 0.24 PPG -- Adams
# alone in a tier, one quarter of a point above the eleven receivers below him.
# A wide tier with no real gap in it is left wide on purpose: the honest report is
# that the position declines steadily there, and inventing a line through it to
# satisfy the cap would be a worse lie than the width.
TIER_SPLIT_DROP = 0.20


def local_rate(gaps, index, window=GAP_WINDOW, floor=1.0):
    """Median of the gaps around `index`, excluding the gap at `index` itself.

    `floor` keeps a neighbourhood of near-identical values from setting a bar of
    zero, which everything would then clear. It is in the units of the series: one
    whole rank slot when reading ranks, a fraction of a point when reading PPG.

    Median rather than mean because one outlier nearby otherwise sets the bar for
    its neighbours. Among the top WRs the gaps run 1, 1, 3, 1, 1, 7: the mean of
    that neighbourhood is 2.4, which hides the 3 between Smith-Njigba and
    Jefferson -- a real break, since every other gap up there is 1. The median
    reads the neighbourhood as 1 and the 3 stands out, which is what puts Nacua,
    Chase and Smith-Njigba in a tier of their own.
    """
    low = max(0, index - window)
    high = min(len(gaps), index + window + 1)
    neighbours = [gap for offset, gap in enumerate(gaps[low:high])
                  if low + offset != index]
    if not neighbours:
        return float(gaps[index])
    return max(statistics.median(neighbours), floor)


def tier_breaks(ranks, p=BREAK_P, window=GAP_WINDOW, min_gap=MIN_GAP,
                max_gap=MAX_GAP, floor=1.0):
    """Indices in `ranks` that start a new tier.

    A break at i means ranks[i] opens a tier and ranks[i-1] closed the last one.

    max_gap is a backstop for the one case the local test reads wrong: where a
    neighbourhood is itself mostly large gaps, the bar rises high enough to swallow
    a break that is obvious in absolute terms. Without it a 14-rank hole sat inside
    a QB tier while 6-rank gaps were splitting tiers elsewhere in the same column.
    """
    if len(ranks) < 2:
        return []
    gaps = [ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)]
    tail = math.log(1.0 / p)

    breaks = []
    for i, gap in enumerate(gaps):
        if gap < min_gap:
            continue
        if gap >= max_gap or gap >= local_rate(gaps, i, window, floor) * tail:
            breaks.append(i + 1)
    return breaks


def _split_wide_tiers(values, breaks, max_spread, min_drop):
    """Add breaks until no tier spans more than `max_spread`.

    `values` is the descending series the tiers were cut from (points per game),
    `breaks` the indices that already open a tier. Each pass finds the widest
    offending tier's largest internal drop and splits there, but only if that drop
    is worth `min_drop` -- a tier that is wide because the position declines
    smoothly has no boundary to find, and is left alone.

    Iterative rather than one pass: splitting a 2.0-wide tier once can leave both
    halves still over the cap.
    """
    if max_spread is None or max_spread <= 0 or len(values) < 2:
        return breaks
    cuts = set(breaks)
    while True:
        bounds = [0] + sorted(cuts) + [len(values)]
        added = False
        for lo, hi in zip(bounds, bounds[1:]):
            if hi - lo < 2 or values[lo] - values[hi - 1] <= max_spread:
                continue
            drop, at = max((values[i] - values[i + 1], i)
                           for i in range(lo, hi - 1))
            if drop < min_drop or (at + 1) in cuts:
                continue
            cuts.add(at + 1)
            added = True
        if not added:
            return sorted(cuts)


def assign_tiers(players, p=BREAK_P, window=GAP_WINDOW, min_gap=MIN_GAP,
                 max_gap=MAX_GAP, basis="ppg", max_spread=TIER_MAX_SPREAD,
                 split_drop=TIER_SPLIT_DROP):
    """[Player] -> ([Row], tier_count).

    basis="ppg" -- the default -- re-sorts the position by premium-adjusted points
    per game and cuts where the drop between neighbours is large for its
    neighbourhood.

    basis="rank" cuts tiers from gaps in the overall rank instead, and leaves the
    players in rank order. That was the default and it should not have been. A
    rank gap counts how many OTHER positions PFF slotted in between, so the test
    is scale-free in the wrong units: through the dense middle of the board a
    position can collapse 5.65 PPG without ever opening a gap wide enough to
    break, and the board shipped a 23-man running back tier because of it. Worse,
    four tier bands ran backwards against the PPG column two cells away -- Travis
    Etienne at 13.76 sitting a tier BELOW D'Andre Swift at 11.90 -- and at draft
    speed the colour band is the loudest thing on the row.

    Rank stays on the board as the timing column either way, so ordering by value
    costs no information. The same exponential-tail test runs on both; only the
    series it reads changes.

    On the ppg basis the gap test's breaks are then topped up by a spread cap, so
    no tier claims that players TIER_MAX_SPREAD apart are interchangeable. The cap
    does not apply on the rank basis: its units are points, and a block ordered by
    rank is not monotone in points, so there is no "spread" down a tier to bound
    and no drop within one that a split would land on meaningfully.
    """
    if not players:
        return [], 0

    if basis == "ppg":
        players = sorted(players, key=lambda pl: (-pl.ppg, pl.rank))
        # Drops, so the series is increasing the same way rank gaps are and the
        # one test reads both. Ties give a zero drop, which never breaks.
        series = [round(-pl.ppg, 4) for pl in players]
        breaks = tier_breaks(series, p, PPG_GAP_WINDOW, MIN_DROP, MAX_DROP,
                             floor=MIN_DROP)
        breaks = _split_wide_tiers([pl.ppg for pl in players], breaks,
                                   max_spread, split_drop)
    else:
        players = sorted(players, key=lambda pl: pl.rank)
        breaks = tier_breaks([pl.rank for pl in players], p, window, min_gap,
                             max_gap)
    bounds = [0] + breaks + [len(players)]

    rows = []
    for tier in range(len(bounds) - 1):
        for pl in players[bounds[tier]:bounds[tier + 1]]:
            rows.append(Row(tier=tier + 1, name=pl.name, rank=pl.rank,
                            ppg=pl.ppg, adp=pl.adp, team=pl.team))
    return rows, len(bounds) - 1


# --------------------------------------------------------------------------
# Google auth / .env handling (same approach as dk_line/dk_player_props.py)
# --------------------------------------------------------------------------

def load_env_file(path):
    """Minimal .env reader so this stays free of a python-dotenv dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def save_env_value(path, key, value):
    """Set key=value in the .env file, replacing any existing line for that key."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh
                     if not ln.strip().startswith(f"{key}=")]
    lines.append(f'{key}="{value}"')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def google_credentials():
    """OAuth user credentials from GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN.

    No scopes are passed. A refresh that names scopes gets an access token limited
    to exactly those, so listing the base set here quietly stripped script.projects
    back off a token that had been granted it. Omitting them returns a token
    carrying the full grant, whatever that grant happens to be.
    """
    from google.oauth2.credentials import Credentials

    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        return None
    return Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
    )


def setup_google_auth(client_secret_path, env_path):
    """Run the browser consent flow once and save the refresh token to .env.

    Sign in as SHEETS_ACCOUNT -- the token identifies whichever account consents,
    so signing in as the wrong one is the easiest mistake to make here.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not os.path.exists(client_secret_path):
        print(f"OAuth client file not found: {client_secret_path}\n"
              "Pass --client-secret with the path to a Desktop-app client_secret.json "
              "(console.cloud.google.com/apis/credentials).", file=sys.stderr)
        return 1

    print(f"A browser window will open -- sign in as {SHEETS_ACCOUNT}", file=sys.stderr)
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SETUP_SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")

    with open(client_secret_path, encoding="utf-8") as fh:
        config = json.load(fh)
    installed = config.get("installed", config.get("web", {}))

    email = ""
    try:  # confirm which account actually consented
        from googleapiclient.discovery import build
        email = build("oauth2", "v2", credentials=creds).userinfo().get().execute().get("email", "")
    except Exception:
        pass

    if email and email.lower() != SHEETS_ACCOUNT.lower():
        print(f"\nAuthorised as {email}, but this script expects {SHEETS_ACCOUNT}.\n"
              "Re-run --setup-auth and pick the right account.", file=sys.stderr)
        return 1

    # Update the three credential keys in place. Rewriting the file wholesale
    # would take the spreadsheet URL and script id down with it, which is exactly
    # what happened the first time this ran after those keys existed.
    save_env_value(env_path, "GOOGLE_CLIENT_ID", installed["client_id"])
    save_env_value(env_path, "GOOGLE_CLIENT_SECRET", installed["client_secret"])
    save_env_value(env_path, "GOOGLE_REFRESH_TOKEN", creds.refresh_token)
    print(f"\nSaved credentials for {email or SHEETS_ACCOUNT} to {env_path}", file=sys.stderr)
    print("Keep that file out of version control.", file=sys.stderr)
    return 0


def open_spreadsheet(client, url, name):
    """Open the target spreadsheet, creating it by name only on a first run."""
    import gspread

    if url:
        return client.open_by_url(url), False
    try:
        return client.open(name), False
    except gspread.SpreadsheetNotFound:
        return client.create(name), True


def hex_to_rgb(value, default=None):
    """'#c6dbef' -> {'red': .., 'green': .., 'blue': ..} for the Sheets API."""
    value = (value or "").lstrip("#")
    if len(value) != 6:
        return default
    try:
        r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return default
    return {"red": r, "green": g, "blue": b}


def clear_conditional_formats(spreadsheet, sheet_id):
    """Drop every conditional format rule on a tab.

    worksheet.clear() only clears values, so without this the tier colour rules
    would stack up one full set per run.
    """
    metadata = spreadsheet.fetch_sheet_metadata()
    for sheet in metadata.get("sheets", []):
        if sheet.get("properties", {}).get("sheetId") != sheet_id:
            continue
        count = len(sheet.get("conditionalFormats", []))
        if count:
            spreadsheet.batch_update({"requests": [
                {"deleteConditionalFormatRule": {"sheetId": sheet_id, "index": i}}
                for i in range(count - 1, -1, -1)
            ]})
        return


# --------------------------------------------------------------------------
# Sheet writing
# --------------------------------------------------------------------------

def block_columns(order):
    """0-indexed first column of the nth visible position block."""
    return FIRST_BLOCK_COL + order * BLOCK_STRIDE


def roster_columns(blocks):
    """0-indexed first column of the visible My Team panel.

    One spacer column past the last position block, so the panel reads as its own
    thing rather than as a fifth position.
    """
    return block_columns(len(blocks) - 1) + BLOCK_WIDTH + 1


def source_columns(order, blocks):
    """0-indexed first column of the nth hidden source block.

    The hidden source sits two spacer columns past the My Team panel so a stray
    edit at the right edge of the board cannot land in it. A source block is
    Rank | Player | Tm | PPG | ADP | Tier -- no checkbox column, that only exists on
    the board, and no VONA, which the board computes for itself.
    """
    start = roster_columns(blocks) + ROSTER_WIDTH + 2
    return start + order * SOURCE_WIDTH


def hidden_columns(blocks):
    """0-indexed first column of every hidden bookkeeping area, in one place.

    build_sheet_payload and format_requests both need these and must agree, so
    neither computes them for itself.

    The timing helpers and the roster staging go to the LEFT of the drafted log on
    purpose: the writer rewrites everything up to the log and leaves the log and
    its backup alone, so a rebuild cannot clobber picks in progress -- and anything
    living out past the log would never be rewritten at all.
    """
    names = blocks[-1]["src"] + SOURCE_WIDTH
    stage = names + LOOKUP_WIDTH
    helper = stage + ROSTER_STAGE_WIDTH
    return {"names": names, "stage": stage, "helper": helper,
            "drafted": helper + 1, "backup": helper + 2}


def build_sheet_payload(tiered, positions, generated_at, teams=DEFAULT_TEAMS,
                        pick=DEFAULT_PICK, te_premium=TE_PREMIUM, standard=None,
                        depths=None, standard_depths=None, off_board=None,
                        standard_tiers=None, tier_on="ppg"):
    """Return (values, blocks) for the whole sheet, as a dense 2-D value grid.

    values is written in a single update; blocks carries the geometry every
    formatting request needs afterwards.
    """
    blocks = [
        {"position": position, "rows": tiered[position],
         "col": block_columns(order), "order": order}
        for order, position in enumerate(positions) if tiered.get(position)
    ]
    depths = depths or {}
    standard_depths = standard_depths or {}
    for block in blocks:
        block["src"] = source_columns(block["order"], blocks)
        # The high-water mark across BOTH boards. The band rules are emitted once
        # and have to cover whichever board the toggle is showing, and while they
        # cycle now -- so this only ever decides whether a position is short enough
        # to need fewer than the whole cycle -- a count taken off one board would
        # still be the wrong count for the other. Premium runs deeper at tight end
        # (12 tiers against 10); on --tier-on rank either side can be the deeper one.
        block["tiers"] = max(
            [row.tier for row in block["rows"]]
            + [tier for (position, _), tier in (standard_tiers or {}).items()
               if position == block["position"]],
            default=0)
        block["depth"] = depths.get(block["position"])
        block["standard_depth"] = standard_depths.get(block["position"])

    longest = max(len(block["rows"]) for block in blocks)
    hidden = hidden_columns(blocks)
    names_col = hidden["names"]
    stage_col = hidden["stage"]
    helper_col = hidden["helper"]
    drafted_col = hidden["drafted"]
    backup_col = hidden["backup"]
    roster_col = roster_columns(blocks)
    width = backup_col + 1
    all_names = [row.name for block in blocks for row in block["rows"]]
    off_board = list(off_board or [])
    total_rows = max(FIRST_DATA_ROW + longest,
                     FIRST_DATA_ROW + len(all_names) + len(off_board))

    # Three rows past the data, in each block's hidden Tier column, hold the
    # cross-block bookkeeping the banner and the best-value highlight compare
    # against: what waiting costs at this position, whether your roster still
    # needs it, and the best value on offer there if it does. They sit below every
    # range the board's formulas read, so they cannot be swept into their own
    # inputs, and in the hidden column so they are never on screen.
    grid = [["" for _ in range(width)] for _ in range(total_rows + 3)]
    urgency_row = total_rows
    need_row = total_rows + 1
    best_row = total_rows + 2
    lookup_range = (f"${column_letter(names_col)}${FIRST_DATA_ROW + 1}:"
                    f"${column_letter(names_col + LOOKUP_WIDTH - 1)}"
                    f"${FIRST_DATA_ROW + max(len(all_names), 1)}")
    # {name: (standard ppg, standard adp)} for the tight ends, so the toggle has an
    # off position to switch to. Empty when the premium is disabled at build time,
    # which also leaves the checkbox unticked and every TE cell a plain number.
    standard = standard or {}
    toggle_cell = f"${column_letter(TOGGLE_COL)}${CONTROL_ROW + 1}"

    # A1 is the reset checkbox and the banner starts at B1, so the one control
    # that wipes the draft sits in the one cell that is always on screen. Ticking
    # it stashes the log in the backup column first, so a misclick is undoable.
    drafted_letter = column_letter(drafted_col)
    drafted_range = (f"${drafted_letter}${FIRST_DATA_ROW + 1}"
                     f":${drafted_letter}${DRAFTED_LIMIT}")
    # The scoring label reads off the toggle rather than off the build, so the
    # banner can never claim a premium the numbers below it are not using.
    scoring = (f'IF({toggle_cell}, "PPR, TE {1 + te_premium:g}", "PPR")'
               if standard else '"PPR"')
    grid[0][1] = (f'="◀ reset   ·   2026 Draft Tiers   ·   PPG over {GAMES} games'
                  f'   ·   "&{scoring}&'
                  f'"   ·   "&COUNTA({drafted_range})&" drafted   ·   {generated_at}"')
    grid[HEADER_ROW][drafted_col] = "Drafted"
    grid[HEADER_ROW][backup_col] = "Drafted (backup)"
    grid[HEADER_ROW][helper_col] = "Draft timing"

    # Draft-timing helpers, stacked in one hidden column so every block's formulas
    # can point at fixed cells. Rows, 1-indexed on the sheet: teams, current pick,
    # target pick, your pick in the order, waiting horizon, that pick valid, on the
    # clock. "my_pick" throughout is your place in the order, never a pick number.
    helper = column_letter(helper_col)
    teams_cell = f"${helper}${FIRST_DATA_ROW + 1}"
    current_cell = f"${helper}${FIRST_DATA_ROW + 2}"
    target_cell = f"${helper}${FIRST_DATA_ROW + 3}"
    my_pick_cell = f"${helper}${FIRST_DATA_ROW + 4}"
    horizon_cell = f"${helper}${FIRST_DATA_ROW + 5}"
    valid_cell = f"${helper}${FIRST_DATA_ROW + 6}"

    # The controls read/write these through row 2, so the inputs live here and the
    # visible cells are plain mirrors -- one source of truth per value.
    control = f"${column_letter(CONTROL_INPUT_COL)}${CONTROL_ROW + 1}"
    grid[FIRST_DATA_ROW][helper_col] = f"={control}"
    # Current pick: one past however many players are off the board. There is no
    # manual override any more -- the log IS the clock, so the only way the two can
    # disagree is a pick nobody ticked, and the fix for that is ticking it.
    grid[FIRST_DATA_ROW + 1][helper_col] = f"=COUNTA({drafted_range})+1"
    # Target pick: when YOU next pick. Everything timing-related on the board hangs
    # off this one cell -- the ADP colouring, VONA, the cost-of-waiting banner.
    #
    # It used to be derived from the pick on the clock plus the snake turn, on the
    # reasoning that the current pick already fixes the position in the round. That
    # is true, but it fixes *the current picker's* position, not yours -- so the
    # board was right at the instant of your own turn and wrong for the other
    # eleven. Picking 6th, at pick 1, it claimed your next pick was 24 when it was 6:
    # an 18-pick error, in the direction that tells you players will still be there
    # who are about to go.
    #
    # Knowing your pick in the order makes it exact. Your picks in a snake are, for
    # round r,
    #   (r-1)*teams + my_pick            on odd rounds
    #   (r-1)*teams + teams-my_pick+1    on even rounds
    # and the target is the first of those strictly past the current pick -- which
    # gives your next turn while you wait, and your turn *after this one* when you
    # are the one on the clock. Checking the current round and the next one is
    # enough, since the next round's pick always lands beyond the current one.
    #
    # The Pick cell ships populated, so this is the live path. Cleared, or typed
    # outside the league, it still falls back to the old derivation rather than
    # breaking, and the status line says so while it does.
    def mine(round_expr):
        return (f"(({round_expr})-1)*{teams_cell} + "
                f"IF(ISODD({round_expr}), {my_pick_cell}, {teams_cell}-{my_pick_cell}+1)")
    this_round = f"CEILING({current_cell}/{teams_cell})"
    legacy = (f"{current_cell} + 2*({teams_cell}-1-MOD({current_cell}-1, {teams_cell})) + 1")
    grid[FIRST_DATA_ROW + 2][helper_col] = (
        f"=IF(NOT({valid_cell}), {legacy}, "
        f"LET(r_, {this_round}, "
        f"p0_, {mine('r_')}, p1_, {mine('r_+1')}, "
        f"IF(p0_>{current_cell}, p0_, p1_)))")
    grid[FIRST_DATA_ROW + 3][helper_col] = \
        f"={column_letter(PICK_INPUT_COL)}${CONTROL_ROW + 1}"

    # The waiting horizon, which is NOT the target pick at the turn.
    #
    # At the wheel your next pick is the very next one, so "what survives to my
    # next pick" is a question about a zero-pick gap and every cost-of-waiting
    # number on the board collapses to nothing -- picking 12th, after 11 picks, the
    # target is 13, no tier can be scored as urgent, and VONA reports about a
    # quarter of the movement it should. That is precisely backwards: the wheel is
    # where waiting is most expensive, because after the pair you wait the longest
    # of anyone.
    #
    # So the horizon is the pick after the pair whenever the pair is back to back,
    # and the target pick otherwise. The ADP colouring deliberately keeps using the
    # target -- "will he be there when I pick" really is about the next pick -- but
    # VONA and the urgency strip, which both ask "what does waiting cost", use this.
    grid[FIRST_DATA_ROW + 4][helper_col] = (
        f"=IF(AND({valid_cell}, {target_cell}-{current_cell}<=1), "
        f"LET(r_, CEILING(({target_cell}+1)/{teams_cell}), "
        f"p0_, {mine('r_')}, p1_, {mine('r_+1')}, "
        f"IF(p0_>{target_cell}, p0_, p1_)), {target_cell})")
    # One place that says whether the Pick cell is usable, so the status line, the
    # My Team banner and the horizon cannot disagree about it. A 13 in a 12-team
    # league is nobody's pick; it used to print "you pick at 216" in perfect
    # confidence next to an empty roster panel.
    grid[FIRST_DATA_ROW + 5][helper_col] = (
        f"=AND({my_pick_cell}<>\"\", ISNUMBER({my_pick_cell}), "
        f"{my_pick_cell}>=1, {my_pick_cell}<={teams_cell})")
    # Whether the pick on the clock is YOURS. The target pick is deliberately the
    # first of your picks strictly *past* the current one, so on your own turn it
    # already reads as your next turn -- correct for every waiting calculation on the
    # board, and silent about the thing you most need to know at that moment.
    # Picking 1st, at pick 1, the board said "you pick at 24" and never said "and you
    # are picking right now". Same snake arithmetic, evaluated at the current pick.
    grid[FIRST_DATA_ROW + 6][helper_col] = (
        f"=AND({valid_cell}, {current_cell}=LET(r_, {this_round}, {mine('r_')}))")
    on_clock_cell = f"${helper}${FIRST_DATA_ROW + 7}"

    # The positions of the last RUN_WINDOW picks, spilled down the helper column.
    #
    # Every position banner needs this to light its RUN badge, and each one used to
    # work it out for itself: four copies of the same FILTER over the log and the
    # same VLOOKUP of every recent name against the 250-row All Players column, all
    # four re-running on every tick of a checkbox to answer a question with one
    # answer. Once here, and the banners become a COUNTIF over ten cells.
    #
    # Spilled rather than joined into a string because a spill is what COUNTIF wants,
    # and it cannot overflow: the FILTER admits at most RUN_WINDOW rows by
    # construction, and the grid leaves that many blank below this cell.
    run_first = FIRST_DATA_ROW + 8            # 1-indexed: the row this spills from
    run_range = (f"${helper}${run_first}:${helper}${run_first + RUN_WINDOW - 1}")
    grid[FIRST_DATA_ROW + 7][helper_col] = (
        f"=IFERROR(LET(log_, {drafted_range}, c_, COUNTIF(log_, \"?*\"), "
        f"i_, SEQUENCE(ROWS(log_)), "
        f"recent_, FILTER(log_, i_>MAX(0, c_-{RUN_WINDOW}), i_<=c_), "
        f"IFERROR(VLOOKUP(recent_, {lookup_range}, 2, FALSE), \"\")), \"\")")

    # Controls: two inputs -- how many teams, and which pick is yours -- then a live
    # status string. Nothing else, and no instructions: the two things the board
    # cannot work out for itself are the two things in the row.
    #
    # Each label sits in a wide column and is right-aligned against its input, which
    # sits in a narrow one. Column widths belong to the whole board, so this is the
    # only way to fit a word next to a two-digit box without stretching a column of
    # 200 players below it.
    grid[CONTROL_ROW][TEAMS_LABEL_COL] = "Teams"
    grid[CONTROL_ROW][CONTROL_INPUT_COL] = teams
    grid[CONTROL_ROW][PICK_LABEL_COL] = "Pick"
    grid[CONTROL_ROW][PICK_INPUT_COL] = pick
    grid[CONTROL_ROW][TOGGLE_COL] = bool(standard)     # TE premium on by default
    # The flag, in its own cell so the highlight can be the words and nothing else.
    grid[CONTROL_ROW][FLAG_COL] = f'=IF({on_clock_cell}, "YOUR PICK", "")'
    # The status line: state, not advice. Where the draft is, whether the pick on the
    # clock is yours, where your next one is, and -- the one thing it must never do
    # quietly -- that the Pick cell is not a pick in this league, because then every
    # timing number below it is a guess.
    #
    # "n picks away" is the distance to that pick, target - current: from pick 1, pick
    # 24 is 23 picks away. It used to count the picks in between instead, which reads
    # as one short of the answer to the question actually being asked -- and at the
    # wheel that count was zero, so the number had to be suppressed. This way it is
    # never zero and never needs a special case.
    away = f"{target_cell}-{current_cell}"
    grid[CONTROL_ROW][STATUS_COL] = (
        f'="◀ TE premium   ·   pick "&{current_cell}&"   ·   round "'
        f'&CEILING({current_cell}/{teams_cell})'
        f'&IF(NOT({valid_cell}), "   ·   there is no pick "&{my_pick_cell}&" in a "'
        f'&{teams_cell}&"-team draft",'
        f' IF({on_clock_cell}, "   ·   next at ",'
        f' "   ·   you pick at ")&{target_cell}'
        f'&" ("&({away})&IF({away}=1, " pick away)", " picks away)")'
        f'&IF({horizon_cell}>{target_cell}, " — back to back, then "&{horizon_cell}, ""))')

    for block in blocks:
        col, src = block["col"], block["src"]
        # Every range in this block is bounded by THIS position's length, not by the
        # height of the tallest thing on the sheet.
        #
        # It used to be total_rows -- 250-odd rows, the length of the All Players
        # column -- for all four blocks, so the quarterback block's VONA, VOR and
        # Fall columns each evaluated over 250 rows to fill 27, and every conditional
        # format rule keyed to them was tested against the same 250. The bound is
        # exact rather than padded: the FILTER below can only ever spill as many rows
        # as the position has players, and a rebuild rewrites every one of these
        # formulas anyway, so there is nothing for the padding to protect.
        rows_here = len(block["rows"])
        src_last = FIRST_DATA_ROW + rows_here
        # "VONA" not "VOR": the column is value over the *next available* player at
        # the position, which moves with every pick, not value over a season-long
        # replacement baseline.
        grid[HEADER_ROW][col:col + BLOCK_WIDTH] = [
            "", "Rank", "Player", "Tm", "PPG", "ADP", "Tier", "Fall", "VOR", "VONA"]

        # The block body is one spilling FILTER over the hidden source, excluding
        # any name in the drafted log. It starts one column right of the block so
        # the checkbox column stays writable -- a spilled array cannot be typed in.
        src_first = column_letter(src)                        # Rank
        src_name = column_letter(src + 1)                     # Player, matched
        src_last_col = column_letter(src + SOURCE_WIDTH - 1)  # Tier
        drafted = drafted_range
        # Sorted, not just filtered.
        #
        # The hidden source is written once, in the order assign_tiers cut it: by
        # PPG descending with rank breaking ties. The toggle can only ever swap a
        # cell's *value*, so a FILTER on its own renumbered the block and left it
        # standing in premium order -- switch the premium off and the tight ends
        # showed standard PPG running 12.4, 12.9, 12.1 down the column, out of
        # order against the very number the board is sorted by.
        #
        # Sorting on the same key the source was built with makes the premium
        # position a no-op by construction -- the array is already in that order --
        # and re-orders every block the toggle actually moves. The tie-break column
        # matches assign_tiers' so the two orders agree exactly rather than nearly.
        #
        # Indices are 1-based into the filtered range, which starts at Rank, so a
        # block offset doubles as a source index: Rank 1, PPG PPG_COL.
        sort_keys = ("1, TRUE" if tier_on == "rank"
                     else f"{PPG_COL}, FALSE, 1, TRUE")
        grid[FIRST_DATA_ROW][col + 1] = (
            f"=IFERROR(SORT(FILTER("
            f"{src_first}${FIRST_DATA_ROW + 1}:{src_last_col}${src_last}, "
            f"{src_name}${FIRST_DATA_ROW + 1}:{src_name}${src_last}<>\"\", "
            f"COUNTIF({drafted}, "
            f"{src_name}${FIRST_DATA_ROW + 1}:{src_name}${src_last})=0), "
            f"{sort_keys}), \"\")"
        )

        # Val and % read the *displayed* block, which the FILTER has already cut
        # down to undrafted players, so neither needs its own availability logic.
        name_col = column_letter(col + 2)
        ppg_col = column_letter(col + PPG_COL)
        adp_col = column_letter(col + ADP_COL)
        first, last = FIRST_DATA_ROW + 1, FIRST_DATA_ROW + rows_here
        names_rng = f"${name_col}${first}:${name_col}${last}"
        ppg_rng = f"${ppg_col}${first}:${ppg_col}${last}"
        adp_rng = f"${adp_col}${first}:${adp_col}${last}"

        # Value over replacement: what this player is worth above what the position
        # will still offer a round from now. Replacement is the average of the top
        # REPLACEMENT_POOL survivors rather than the single best, so one player with
        # a late ADP cannot set the bar for a whole position and flatten every value
        # behind him -- if that one gets sniped early, the average barely moves.
        # If the survivor pool comes back empty, treat every available player as a
        # survivor rather than pricing against the weakest one on the board. An
        # empty pool almost never means "the position is about to be stripped": it
        # means the horizon has run past the deepest ADP in the data (which tops
        # out near 171, so any pick past ~160 in a 12-team league), and the old
        # fallback then measured everyone against the worst player left and
        # reported a fake +6 with half the board over +1.
        # Done by dropping the horizon to zero rather than by switching between two
        # arrays: this whole expression sits inside an ARRAYFORMULA, which
        # evaluates *both* arms of an IF before selecting, so an arm containing
        # SEQUENCE(0) poisons the column with #NUM! even when it is not chosen.
        # One scalar horizon feeding one FILTER has no unselected arm to blow up.
        survivors = (f"COUNTIFS({adp_rng}, \">\"&{horizon_cell}, {names_rng}, \"<>\")")
        horizon = f"IF({survivors}=0, 0, {horizon_cell})"
        pool = (f"IFERROR(FILTER({ppg_rng}, {adp_rng}>tgt_, "
                f"{names_rng}<>\"\"), \"\")")
        # A player must not be in his own survivor pool. If he is -- and any player
        # whose own ADP is past the horizon is -- then he is partly being priced
        # against himself, which flatters nobody but understates exactly the
        # players the column exists to judge: the ones you are deciding whether to
        # wait on. Josh Allen came out at +0.30 where the honest number is +0.51.
        #
        # It cannot be fixed by filtering per row, because the pool is one scalar
        # for the whole column and ARRAYFORMULA has no per-row FILTER. So it is
        # fixed arithmetically instead: take the top REPLACEMENT_POOL+1 survivors,
        # and for a player who is himself inside the top REPLACEMENT_POOL, subtract
        # his own points from that sum and divide by REPLACEMENT_POOL. That is
        # exactly the mean of the pool with him removed. Everyone else keeps the
        # plain top-REPLACEMENT_POOL mean, and when the pool is too small to have a
        # spare member the correction switches itself off.
        pool_size = REPLACEMENT_POOL
        next_best = (
            f"LET(tgt_, {horizon}, pool_, {pool}, n_, COUNT(pool_), "
            f"k_, MIN({pool_size}, MAX(1, n_)), "
            f"k1_, MIN({pool_size + 1}, MAX(1, n_)), "
            f"top_, SUM(LARGE(pool_, SEQUENCE(k_))), "
            f"top1_, SUM(LARGE(pool_, SEQUENCE(k1_))), "
            f"cut_, LARGE(pool_, k_), "
            f"IF(({adp_rng}>tgt_) * ({ppg_rng}>=cut_) * (n_>{pool_size}), "
            f"(top1_ - {ppg_rng})/{pool_size}, top_/k_))")
        # The outer IFERROR is the backstop for a position drafted out entirely,
        # where there is no pool left to average at all.
        grid[FIRST_DATA_ROW][col + VAL_COL] = (
            f"=ARRAYFORMULA(IF({names_rng}=\"\", \"\", "
            f"IFERROR({ppg_rng} - {next_best}, \"\")))")

        # Season-long VOR, next to VONA because they answer different questions and
        # a drafter needs both. VONA is what one turn of waiting costs right now;
        # VOR is how far above a startable player he is all year. They can disagree
        # sharply and both be right -- a quarterback whose backup is nearly as good
        # has a small VONA and a large VOR, and that is exactly the truth about him.
        #
        # Replacement is the last starter at the position, counted off the *source*
        # column rather than the visible board, so the league-wide half of this
        # stays a preseason judgement about the player and does not drift as the
        # draft empties the position.
        #
        # The DEPTH follows the toggle too, not just the points. Replacement is a
        # property of the format: premium scoring wins tight ends six flex slots
        # this league would otherwise spend on running backs and receivers, so the
        # TE replacement is the TE18 with the premium on and the TE12 with it off.
        # Baking in the premium depth as a literal -- which is what this did --
        # switched the numerator and left the denominator behind, and the toggle's
        # off position priced tight ends against a replacement level that only
        # exists because of the premium.
        # And the DEPTH follows the Teams cell, so the one control that changes the
        # size of the league changes the replacement level with it. It used to be a
        # literal: set Teams to 10 and the pick timing followed while every VOR on
        # the board went on pricing against a 12-team replacement, silently, about
        # a point high on every running back.
        #
        # Scaled proportionally rather than recomputed. The merit-resolved flex
        # cannot be redone in a formula, but replacement depth is very nearly
        # linear in league size -- twice the teams, twice as far down each position
        # you are forced -- and a proportional estimate that moves is worth far
        # more than an exact number that does not.
        depth = min(block.get("depth") or rows_here, rows_here)
        plain_depth = min(block.get("standard_depth") or depth, rows_here)
        src_ppg = (f"${column_letter(src + 3)}${FIRST_DATA_ROW + 1}"
                   f":${column_letter(src + 3)}${src_last}")
        built = (str(depth) if plain_depth == depth
                 else f"IF({toggle_cell}, {depth}, {plain_depth})")
        # Clamped to leave room for the smoothing window BELOW it, not just above.
        # The band reads LARGE(ppg, {d-1, d, d+1}), so a depth sitting on the last
        # player asks for one past the end of the position -- LARGE returns #NUM!,
        # AVERAGE propagates it, and the IFERROR wrapped round the whole column
        # turned that into a silently EMPTY VOR column. It could not happen at the
        # shipped settings (RB29 of 70, TE18 of 31) and happened at every position
        # at --limit 60, where replacement is deeper than the pool is long.
        if rows_here >= 3:
            at = (f"MIN({rows_here - 1}, "
                  f"MAX(2, ROUND(({built}) * {teams_cell} / {teams})))")
        else:
            at = "1"
        # Replacement is the mean of the three around that depth, not the single
        # player standing on it. The Python side has always smoothed it -- one
        # projection is a thin thing to price a whole position against -- and the
        # sheet was the one place still taking a single LARGE(). Through the flat
        # stretch at RB29 and WR37 that one number moved the entire column.
        band = (f"AVERAGE(LARGE({src_ppg}, "
                f"IF(d_=1, SEQUENCE(1), SEQUENCE(3, 1, d_-1))))")
        league_bar = f"LET(d_, {at}, {band})"

        # ... and then the same question asked of YOUR roster, which is the half
        # that decides anything once you own players.
        #
        # A league-wide replacement level answers "is he a startable player". By
        # the fourth round that is not the question any more: the question is what
        # he ADDS TO THIS TEAM, and for a position you have already filled the
        # answer is not measured against the WR37 -- it is measured against the
        # receiver of yours he would displace. An elite WR3 priced against the
        # WR37 looks enormous and priced against your own WR2 looks like what it
        # is, and both boards agree he is worth taking; a fourth running back
        # priced against the RB29 also looks large, and against your own RB2 looks
        # like nothing, which is the truth the old column could not tell.
        #
        # This replaces the need gate that used to sit in front of the highlight.
        # That gate was binary and it locked positions out: two receivers on the
        # roster set WR's need to 0, and no receiver could be recommended until
        # QB, both backs and the tight end were filled -- so at a realistic round
        # five it pointed at Etienne (+3.3) while hiding Smith-Njigba (+8.0),
        # 4.7 PPG hidden behind a checklist. Worse, a position not in
        # FLEX_POSITIONS never came back at all: once you owned a quarterback, no
        # quarterback could ever be the best value on the board again.
        #
        # Marginal value has no such cliff. It needs no gate, it is still in points
        # per game so it stays comparable across positions -- which is the whole
        # reason this column and not VONA drives the cross-position cue -- and it
        # handles the cases the gate got wrong by construction rather than by
        # special case.
        #
        # The bar only moves once there is genuinely nowhere left to put him: while
        # a mandatory slot at his position is open, or while the flex is open and he
        # can fill it, he is additive and the league replacement is the honest
        # comparison. Empty roster, or no pick set, means every slot is open, so
        # this degrades to exactly the old column.
        position = block["position"]

        def staged(offset):
            letter = column_letter(stage_col + offset)
            return (f"${letter}${FIRST_DATA_ROW + 1}:"
                    f"${letter}${FIRST_DATA_ROW + ROSTER_STAGE_ROWS}")

        st_pos, st_ppg, st_frank, st_bench = (staged(1), staged(2),
                                              staged(6), staged(7))
        open_here = f"COUNTIF({st_pos}, \"{position}\")<{LINEUP.get(position, 0)}"
        flex_full = (f"COUNTIFS({st_frank}, \">=1\", {st_frank}, "
                     f"\"<={FLEX_SLOTS}\")>={FLEX_SLOTS}")
        flex_open = "FALSE" if position not in FLEX_POSITIONS else f"NOT({flex_full})"
        # Who he would displace: the weakest of your starters at his own position,
        # and -- if he is flex eligible -- the weakest of whoever is holding your
        # flex slots, since he can take either. Bench players are not in it: a
        # player who only beats your bench has not improved your lineup.
        #
        # 1E9 when a candidate set is empty so it loses the MIN rather than winning
        # it with a zero. Both sets are non-empty on the branch that reads this, but
        # a formula that returns garbage when its guard is bypassed is a trap.
        mine_at = (f"IF(COUNTIFS({st_pos}, \"{position}\", {st_bench}, 0)=0, 1E9, "
                   f"MINIFS({st_ppg}, {st_pos}, \"{position}\", {st_bench}, 0))")
        parts = [mine_at]
        if position in FLEX_POSITIONS:
            parts.append(
                f"IF(COUNTIFS({st_frank}, \">=1\", {st_frank}, \"<={FLEX_SLOTS}\")=0, "
                f"1E9, MINIFS({st_ppg}, {st_frank}, \">=1\", "
                f"{st_frank}, \"<={FLEX_SLOTS}\"))")
        mine_bar = f"MIN({', '.join(parts)})"
        # MAX against the league bar, so a roster whose own worst starter is below
        # replacement level cannot make a player look better than startable.
        bar = (f"LET(lg_, {league_bar}, "
               f"IF(OR({open_here}, {flex_open}), lg_, "
               f"LET(my_, {mine_bar}, IF(my_>1E8, lg_, MAX(lg_, my_)))))")
        grid[FIRST_DATA_ROW][col + VOR_COL] = (
            f"=ARRAYFORMULA(IF({names_rng}=\"\", \"\", "
            f"IFERROR({ppg_rng} - {bar}, \"\")))")

        # Fall: how far the market is off the ranking. Both inputs are already in
        # the block, so this is a subtraction and nothing more -- see the column
        # comments at the top of the file for why it earns the space.
        rank_rng = (f"${column_letter(col + 1)}${first}:"
                    f"${column_letter(col + 1)}${last}")
        # Rounded in the formula, not just in the number format. ADP is a decimal
        # and rank is an integer, so a gap of -0.4 is a real value that a "+0;-0;0"
        # format renders as "-0" -- which reads as a bug rather than as "he goes
        # about where he is ranked". Rounding first sends it to the format's zero
        # section and it prints a plain 0.
        grid[FIRST_DATA_ROW][col + FALL_COL] = (
            f"=ARRAYFORMULA(IF({names_rng}=\"\", \"\", "
            f"IF({adp_rng}>={ADP_SENTINEL:g}, \"\", "
            f"IFERROR(ROUND({adp_rng} - {rank_rng}), \"\"))))")

        # Tier scarcity, live, in the position banner. Once the draft is a few
        # rounds old VOR flattens towards zero for everyone -- correctly, because
        # the cost of waiting really has collapsed -- and the question that still
        # has an answer is what disappears if you wait. So: which tier is on the
        # clock at this position, how many of it are left, and how far the drop is
        # to the next tier that still has anyone in it. Two players left above a
        # 2.9 drop is a cliff; three left above 0.4 is not, and VOR cannot tell
        # those apart.
        tier_rng = f"${column_letter(col + TIER_COL)}${first}:" \
                   f"${column_letter(col + TIER_COL)}${last}"
        # How many of the last RUN_WINDOW picks came from this position. A tier
        # cliff says what you lose by waiting; a run says the room has decided to
        # take it now, which is the other way a position disappears and the one
        # the static numbers cannot see.
        #
        # A count over the shared window in the helper column, not a FILTER and a
        # VLOOKUP of its own: the recent picks are already resolved to positions
        # there, once, for all four banners.
        run = f"COUNTIF({run_range}, \"{block['position']}\")"
        # Fired against this position's OWN share of the board, not a flat count.
        # A flat 4-in-10 is below what receivers do at random -- 90 of the 218
        # players on this board are receivers, so 4.1 per 10 picks is their
        # baseline -- and the badge was lit for 69% of a normal draft at WR and
        # 35% at RB. A badge that is nearly always on is furniture. Expected plus
        # two means it fires on a genuine cluster and stays dark otherwise.
        # Rounded UP before the margin: receivers expect 4.1 in ten picks, and
        # rounding that to 4 then adding 2 fires at 6, which is barely above
        # chance. Ceiling gives WR 7, RB 6, TE 4, QB 4 -- a real cluster at each.
        share = len(block["rows"]) / max(sum(len(b["rows"]) for b in blocks), 1)
        alert = max(RUN_ALERT, math.ceil(share * RUN_WINDOW) + 2)
        badge = (f"IF(run_>={alert}, \"  ·  RUN \"&run_&\"/{RUN_WINDOW}\", \"\")")
        # IFERROR(..., "") not IFERROR(..., ). The bare form returns 0, which is a
        # number, so COUNT() saw one value and the "position is drafted out" arm
        # never ran: draft every quarterback and the banner read "QB · T0 ×0"
        # rather than "QB". Plausible-looking output from a dead guard.
        grid[HEADER_ROW - 1][col] = (
            f"=LET(t_, {tier_rng}, p_, {ppg_rng}, n_, {names_rng}, run_, {run}, "
            f"live_, IFERROR(FILTER(t_, n_<>\"\"), \"\"), "
            f"IF(COUNT(live_)=0, \"{block['position']}\"&{badge}, "
            f"LET(top_, MIN(live_), "
            f"left_, COUNTIFS(t_, top_, n_, \"<>\"), "
            f"next_, IFERROR(MIN(FILTER(t_, n_<>\"\", t_>top_)), 0), "
            f"cur_, AVERAGEIFS(p_, t_, top_, n_, \"<>\"), "
            f"nxt_, IFERROR(AVERAGEIFS(p_, t_, next_, n_, \"<>\"), 0), "
            # Spelled out with units. It read "QB · T3 ×5 ↓1.8", which is three
            # figures in three private notations: a tier number, a count of players,
            # and a points-per-game drop, all of them bare. The words cost width the
            # banner has -- it spans a whole block -- and they say which number is
            # which without anyone having to remember what × meant.
            f"\"{block['position']}  ·  tier \"&top_&\"  ·  \"&left_&\" left\"&"
            f"IF(next_=0, \"\", \"  ·  ↓\"&TEXT(cur_-nxt_, \"0.0\")&\" PPG\")"
            f"&{badge})))")

        # What waiting a round actually costs at this position, for the banner
        # highlight. A big cliff is only a problem if the tier will be gone before
        # you pick again, so the drop counts for nothing while anyone in the tier
        # is still expected to survive the horizon -- same ADP test the VOR column
        # uses, applied to the tier instead of the player. Seven left above a 0.2
        # drop scores zero; two left above a 2.3 drop with neither expected to
        # last scores the full 2.3.
        grid[urgency_row][col + TIER_COL] = (
            f"=LET(t_, {tier_rng}, p_, {ppg_rng}, n_, {names_rng}, "
            f"live_, IFERROR(FILTER(t_, n_<>\"\"), \"\"), "
            f"IF(COUNT(live_)=0, 0, "
            f"LET(top_, MIN(live_), "
            f"surv_, COUNTIFS(t_, top_, {adp_rng}, \">\"&{horizon_cell}, n_, \"<>\"), "
            f"next_, IFERROR(MIN(FILTER(t_, n_<>\"\", t_>top_)), 0), "
            f"IF(OR(surv_>0, next_=0), 0, "
            f"AVERAGEIFS(p_, t_, top_, n_, \"<>\") "
            f"- AVERAGEIFS(p_, t_, next_, n_, \"<>\")))))")

        # The bar this position's VOR is currently measured against, published for
        # its own sake: it is the one number in the roster-aware column you cannot
        # read off the board, and when a cue looks wrong it is the first thing to
        # check. Hidden, like everything else in this column.
        grid[need_row][col + TIER_COL] = f"={bar}"
        # Ranked on VOR, not VONA, because this is the one cue that compares across
        # positions and VONA cannot do that. VONA measures the drop to the next
        # three players AT THE SAME POSITION, so a deep position can never win it
        # however good its best player is -- which in a premium league points the
        # cue systematically away from tight end, the one thing the whole board
        # exists to price properly. Live: LaPorta VOR +4.3 against Irving +3.2, and
        # the cue chose Irving because his VONA was +1.0 to LaPorta's +0.3.
        #
        # VOR is built to be comparable -- now every position measured against what
        # it would actually add to YOUR lineup -- so it is what a cross-position
        # "take this" should read. VONA keeps its column and its meaning: can I wait
        # on this one.
        #
        # No eligibility test any more. The old one lived here because VOR was a
        # league-wide number that could not tell a first running back from a fourth,
        # so a gate had to do it from outside; the marginal bar above tells them
        # apart itself, and a fourth back now loses this comparison on his own
        # merits instead of being struck off a list. That is what lets an elite WR3
        # win the cue in a league where receivers win flex slots, which the gate
        # made structurally impossible.
        vor_rng = (f"${column_letter(col + VOR_COL)}${first}:"
                   f"${column_letter(col + VOR_COL)}${last}")
        grid[best_row][col + TIER_COL] = f"=IFERROR(MAX({vor_rng}), -999)"

        grid[HEADER_ROW][src:src + SOURCE_WIDTH] = [
            f"{block['position']} Rank", f"{block['position']} Player",
            f"{block['position']} Tm", f"{block['position']} PPG",
            f"{block['position']} ADP", f"{block['position']} Tier",
        ]
        # Every number the premium moves -- rank, PPG and ADP -- becomes a two-way
        # switch on the toggle cell; anything it left alone stays a plain number, so
        # the formulas land only where the two boards actually disagree. Doing it
        # here, in the hidden source, means everything downstream -- the FILTER, VOR,
        # the scarcity strip, the amber rule -- recomputes off the switched values
        # without knowing the toggle exists.
        #
        # Tier follows the toggle too, which it used to be unable to do.
        #
        # The objection was real: tiers are cut from the value drops, so switching
        # scoring recuts them, and these source rows are written once, in premium
        # order. A standard-scoring tier column laid over premium-ordered rows can
        # run backwards down the block and stripe the colour bands.
        #
        # What was wrong about it was the premise that row order "is the grid" and
        # so cannot follow the toggle. The visible block is not the grid: it is a
        # spilling FILTER over these rows, and an array formula can sort. It does --
        # see the SORT above -- on the same key assign_tiers cut on, so the block
        # re-orders itself when the toggle moves and the tier bands come out
        # contiguous by construction rather than by luck.
        #
        # They did come out contiguous by luck, which is why this stood for so long.
        # Cutting tiers on PPG means only the tight ends' own numbers move, so QB,
        # RB and WR tiers are byte-identical between the two boards and only the
        # tight ends recut; and down the tight end block in premium order the
        # standard tiers still ran monotone, because the premium is very nearly a
        # monotone transform of the position -- it reorders neighbours inside a
        # tier, never across one. Contiguous bands over a column that was visibly
        # out of order by its own sort key.
        #
        # tier_striping() still warns at build time. With the sort in place it
        # should never fire, so if it ever does the sort key and the tier basis have
        # drifted apart and both are worth looking at.
        #
        # On --tier-on rank the switch matters much more: ranks move at every
        # position, so every block recuts, and the sort follows the basis.
        for offset, row in enumerate(block["rows"]):
            cells = [row.rank, row.name, row.team, row.ppg, row.adp, row.tier]
            plain = standard.get((block["position"], row.name))
            if plain:
                plain_ppg, plain_adp, plain_rank = plain
                if plain_rank != row.rank:
                    cells[0] = f"=IF({toggle_cell}, {row.rank}, {plain_rank})"
                if plain_ppg != row.ppg:
                    cells[3] = f"=IF({toggle_cell}, {row.ppg}, {plain_ppg})"
                if plain_adp != row.adp:
                    cells[4] = f"=IF({toggle_cell}, {row.adp}, {plain_adp})"
            plain_tier = (standard_tiers or {}).get((block["position"], row.name))
            if plain_tier and plain_tier != row.tier:
                cells[5] = f"=IF({toggle_cell}, {row.tier}, {plain_tier})"
            grid[FIRST_DATA_ROW + offset][src:src + SOURCE_WIDTH] = cells

    # Every name on the board, in one column, with its position and its PPG beside
    # it. The names alone are the dropdown source for the drafted log (see the
    # ONE_OF_RANGE validation in format_requests); all three together are the lookup
    # the My Team panel reads, which is the only way to get a player's numbers back
    # once he has been drafted -- by then the FILTER has cut him from the board.
    #
    # PPG is a reference into that position's hidden source cell rather than a copy
    # of the number, so it carries the premium toggle: switch the premium off and
    # the roster panel re-values with everything else.
    grid[HEADER_ROW][names_col:names_col + LOOKUP_WIDTH] = [
        "All Players", "Pos", "Player PPG"]
    offset = 0
    for block in blocks:
        ppg_letter = column_letter(block["src"] + 3)
        for index, row in enumerate(block["rows"]):
            grid[FIRST_DATA_ROW + offset][names_col] = row.name
            grid[FIRST_DATA_ROW + offset][names_col + 1] = block["position"]
            grid[FIRST_DATA_ROW + offset][names_col + 2] = (
                f"={ppg_letter}{FIRST_DATA_ROW + index + 1}")
            offset += 1
    # Kickers and defences get no block, but they still get drafted, and every one
    # of them has to reach the log or the round parity behind MY TEAM breaks. They
    # were not in the dropdown, so the only way to record one was to type it by
    # hand and hope the spelling matched. Names and positions only -- there is no
    # source cell to point a PPG at, and their points are not a decision anyway.
    for position, name in (off_board or []):
        grid[FIRST_DATA_ROW + offset][names_col] = name
        grid[FIRST_DATA_ROW + offset][names_col + 1] = position
        offset += 1

    build_roster_panel(grid, roster_col, stage_col, lookup_range,
                       drafted_range, teams_cell, my_pick_cell, valid_cell)

    return grid, blocks, total_rows, width, drafted_col, backup_col


def tier_striping(tiered, standard_tiers):
    """[(position, rows out of order)] where the toggle's tiers would stripe.

    Row order is fixed at build time off the premium board, so the standard tier
    numbers ride on premium-ordered rows. If they are not monotone down a block,
    switching the premium off paints colour bands that run backwards and then
    forwards again -- purely cosmetic, entirely silent, and unfixable in a formula,
    which is exactly the kind of thing worth a line on stderr.
    """
    out = []
    for position, rows in tiered.items():
        previous, backwards = 0, 0
        for row in rows:
            tier = standard_tiers.get((position, row.name))
            if tier is None:
                continue
            if tier < previous:
                backwards += 1
            previous = tier
        if backwards:
            out.append((position, backwards))
    return out


def roster_rows():
    """(starter slot labels, total panel rows before the targets block)."""
    labels = []
    for position, count in LINEUP.items():
        for nth in range(1, count + 1):
            labels.append(position if count == 1 else f"{position}{nth}")
    labels.extend(["FLEX"] * FLEX_SLOTS)
    return labels, len(labels) + ROSTER_BENCH_ROWS


def target_range(roster_col):
    """(A1 range of the target inputs, 0-indexed grid row of the TARGETS header).

    The header sits one blank row below the bench so the two lists do not read as
    one; the inputs are the TARGET_ROWS rows under it.
    """
    _, body = roster_rows()
    header = FIRST_DATA_ROW + body + 1
    letter = column_letter(roster_col + 1)
    return (f"${letter}${header + 2}:${letter}${header + 1 + TARGET_ROWS}", header)


def build_roster_panel(grid, roster_col, stage_col, lookup_range,
                       drafted_range, teams_cell, my_pick_cell, valid_cell):
    """The My Team panel and the hidden staging that fills it.

    Your picks are read straight out of the drafted log. Row n of that log is pick
    n of the draft, and in a snake your picks are, for round r,

        (r-1)*teams + my_pick            on odd rounds
        (r-1)*teams + teams-my_pick+1    on even rounds

    so the panel is a FILTER of the log down to the rows whose index is one of
    those. Nothing here is fixed at build time -- change the Pick cell and the
    whole roster reshuffles, which is what makes one board usable across leagues
    where you draw a different place in the order.

    Lineup slots are then filled on merit rather than in pick order: the best two
    running backs you own start, whoever you took first. Everyone surplus to a
    mandatory slot competes for the flex, and the rest fall to the bench. Same
    rule starter_depths uses to price replacement, so the panel and the VOR column
    are answering with one definition of "starter" between them.
    """
    stage_first, stage_last = FIRST_DATA_ROW + 1, FIRST_DATA_ROW + ROSTER_STAGE_ROWS

    def stage(index):
        letter = column_letter(stage_col + index)
        return f"${letter}${stage_first}:${letter}${stage_last}"

    mine, pos, ppg, key, prank, elig, frank, bench, brank = (stage(i) for i in range(9))
    lookup = lookup_range

    # How many of each position start before the flex is reached, per staged row.
    caps = "0"
    for position, count in reversed(list(LINEUP.items())):
        caps = f'IF({pos}="{position}", {count}, {caps})'
    flexable = "+".join(f'({pos}="{p}")' for p in FLEX_POSITIONS)

    # Which picks are mine. The round is read off the log row index, so this is the
    # same snake arithmetic the target-pick helper uses, run backwards over picks
    # already made instead of forwards over picks to come.
    #
    # The ARRAYFORMULA wrappers are load-bearing, not decoration. CEILING and IF do
    # not broadcast over an array on their own -- inside a bare LET they quietly
    # collapse to their first element, and the whole expression still evaluates and
    # still returns a plausible-looking answer: your round-one pick, every time,
    # with every later round silently missing. Wrapping each computed array is what
    # makes the round vary down the column.
    #
    # A blank Pick cell makes the arithmetic fail, which the IFERROR turns into an empty
    # panel rather than a column of #VALUE! -- the banner is what says why.
    grid[FIRST_DATA_ROW][stage_col] = (
        f"=IFERROR(LET(log_, {drafted_range}, t_, {teams_cell}, s_, {my_pick_cell}, "
        f"i_, SEQUENCE(ROWS(log_)), "
        f"r_, ARRAYFORMULA(CEILING(i_/t_)), "
        f"p_, ARRAYFORMULA((r_-1)*t_+IF(ISODD(r_), s_, t_-s_+1)), "
        f"FILTER(log_, log_<>\"\", i_=p_)), \"\")")
    # Position and PPG for each of them, off the lookup built above.
    for index, column in ((1, 2), (2, 3)):
        grid[FIRST_DATA_ROW][stage_col + index] = (
            f"=ARRAYFORMULA(IF({mine}=\"\", \"\", "
            f"IFERROR(VLOOKUP({mine}, {lookup}, {column}, FALSE), \"\")))")
    # A sort key, not the raw PPG: two players you own on the same projection would
    # otherwise both rank first at their position and one slot would show twice
    # while another showed blank. The row offset is far too small to reorder anyone.
    grid[FIRST_DATA_ROW][stage_col + 3] = (
        f"=ARRAYFORMULA(IF({mine}=\"\", 0, "
        f"IFERROR({ppg} - ROW({mine})/100000, 0)))")
    # Rank within position, best first. Every staged column past here is numeric
    # even on empty rows, so the flags below can do plain arithmetic on them
    # instead of guarding each one against a blank.
    grid[FIRST_DATA_ROW][stage_col + 4] = (
        f"=ARRAYFORMULA(IF({mine}=\"\", 0, "
        f"COUNTIFS({pos}, {pos}, {key}, \">\"&{key})+1))")
    # Flex eligible and surplus to the mandatory slots at his own position.
    grid[FIRST_DATA_ROW][stage_col + 5] = (
        f"=ARRAYFORMULA(IF({mine}=\"\", 0, "
        f"IF((({flexable})>0)*({prank}>{caps}), 1, 0)))")
    grid[FIRST_DATA_ROW][stage_col + 6] = (
        f"=ARRAYFORMULA(IF({elig}=1, "
        f"COUNTIFS({elig}, 1, {key}, \">\"&{key})+1, 0))")
    # Bench: everyone who won neither a mandatory slot nor a flex.
    grid[FIRST_DATA_ROW][stage_col + 7] = (
        f"=ARRAYFORMULA(IF({mine}=\"\", 0, "
        f"IF(({prank}<={caps}) + (({frank}>=1)*({frank}<={FLEX_SLOTS})) > 0, 0, 1)))")
    grid[FIRST_DATA_ROW][stage_col + 8] = (
        f"=ARRAYFORMULA(IF({bench}=1, "
        f"COUNTIFS({bench}, 1, {key}, \">\"&{key})+1, 0))")
    grid[HEADER_ROW][stage_col:stage_col + ROSTER_STAGE_WIDTH] = [
        "My picks", "My Pos", "My PPG", "My key", "My pos rank",
        "Flex eligible", "My flex rank", "Benched", "My bench rank"]

    # The visible panel. One row per roster slot, each pulling the one staged
    # player who won it.
    labels, _ = roster_rows()
    rows = []
    for position, count in LINEUP.items():
        for nth in range(1, count + 1):
            rows.append(f"FILTER({mine}, {pos}=\"{position}\", {prank}={nth})")
    for nth in range(1, FLEX_SLOTS + 1):
        rows.append(f"FILTER({mine}, {frank}={nth})")
    starters = len(rows)
    for nth in range(1, ROSTER_BENCH_ROWS + 1):
        labels.append(f"BN{nth}")
        rows.append(f"FILTER({mine}, {brank}={nth})")

    grid[HEADER_ROW][roster_col:roster_col + ROSTER_WIDTH] = [
        "Slot", "Player", "Pos", "PPG"]
    player_letter = column_letter(roster_col + 1)

    def describe(row):
        """Position and PPG beside whatever name is in this row's Player cell."""
        cell = f"${player_letter}{row + 1}"
        for index, column in ((2, 2), (3, 3)):
            grid[row][roster_col + index] = (
                f"=IF({cell}=\"\", \"\", "
                f"IFERROR(VLOOKUP({cell}, {lookup}, {column}, FALSE), \"\"))")

    for offset, (label, source) in enumerate(zip(labels, rows)):
        row = FIRST_DATA_ROW + offset
        grid[row][roster_col] = label
        grid[row][roster_col + 1] = f"=IFERROR(INDEX({source}, 1), \"\")"
        describe(row)

    # Targets. These rows are the only cells on the whole board you type a player
    # name into -- everything else is derived -- so they are left empty and given a
    # dropdown over every name. A name here lights that player up wherever he sits
    # in his position block, which is the "circled on the printout" the board had
    # no way to express.
    #
    # They live inside the panel rather than off in the hidden columns because the
    # list is worth reading on its own: it is your shortlist, and seeing it beside
    # your roster is how you notice you have taken none of it.
    targets, target_first = target_range(roster_col)
    grid[target_first][roster_col] = "TARGETS"
    grid[target_first][roster_col + 1] = (
        f'=IF(COUNTIF({targets}, "?*")=0, "type names below", '
        f'COUNTIF({targets}, "?*")&" listed, "&'
        f'SUMPRODUCT(--(COUNTIF({drafted_range}, {targets})=0), '
        f'--({targets}<>""))&" still there")')
    for nth in range(1, TARGET_ROWS + 1):
        row = target_first + nth
        grid[row][roster_col] = f"T{nth}"
        describe(row)

    # The panel banner. It carries the caveat as well as the count, because the
    # whole thing rests on the log being a complete record of the draft in order:
    # a pick nobody ticked shifts every later round by one and quietly hands you
    # somebody else's roster.
    #
    # Tight, and tighter than the position banners: those span nine columns, this one
    # spans four narrow ones, and at the wide separator the string ran well past the
    # merge -- centred and clipped at both ends, so "MY TEAM" itself was unreadable.
    # Single-space separators, "taken" rather than "N picks" (there are two senses of
    # pick in one line otherwise), and no "starters" before the PPG, which the rule
    # under the starting block already says.
    ppg_letter = column_letter(roster_col + 3)
    starter_ppg = (f"${ppg_letter}${FIRST_DATA_ROW + 1}:"
                   f"${ppg_letter}${FIRST_DATA_ROW + starters}")
    picks = f'COUNTIF({mine}, "?*")'
    grid[HEADER_ROW - 1][roster_col] = (
        f'=IF(NOT({valid_cell}), "MY TEAM · pick "&{my_pick_cell}&" invalid", '
        f'"MY TEAM · pick "&{my_pick_cell}&" · "&{picks}&" taken · "'
        f'&TEXT(SUM({starter_ppg}), "0.0")&" PPG")')
    return starters


def format_requests(sheet_id, blocks, total_rows, width, drafted_col=None,
                    names_col=None):
    """Every formatting / validation request for the board, in one list."""
    def rng(row1, row2, col1, col2):
        return {"sheetId": sheet_id, "startRowIndex": row1, "endRowIndex": row2,
                "startColumnIndex": col1, "endColumnIndex": col2}

    roster_col = roster_columns(blocks)
    starter_rows = sum(LINEUP.values()) + FLEX_SLOTS
    roster_body = starter_rows + ROSTER_BENCH_ROWS
    targets, target_header = target_range(roster_col)
    target_first = target_header + 1
    target_last = target_first + TARGET_ROWS
    # The banner and the control strip run across the My Team panel too -- it is
    # part of the board, not an appendix to it.
    last_visible = roster_col + ROSTER_WIDTH
    first_src = blocks[0]["src"]
    # Same cell build_sheet_payload writes the target pick into: the helper column
    # is wherever hidden_columns puts it, and the target is its third row. Both
    # sides ask that one function, so the two cannot drift.
    helper_letter = column_letter(hidden_columns(blocks)["helper"])
    current_cell = f"${helper_letter}${FIRST_DATA_ROW + 2}"
    target_cell = f"${helper_letter}${FIRST_DATA_ROW + 3}"

    requests = [
        # Wipe formatting first so a shorter run cannot leave last run's fills behind.
        {"updateCells": {"range": {"sheetId": sheet_id}, "fields": "userEnteredFormat"}},
        # Same for data validation: clearing values leaves rules behind, so an old
        # layout's checkboxes or dropdowns would survive into the new one.
        {"setDataValidation": {"range": {"sheetId": sheet_id}}},
        # And for merges -- a merge from a previous layout that only partly overlaps
        # the new position banner makes the mergeCells request below fail outright.
        {"unmergeCells": {"range": {"sheetId": sheet_id}}},
        # Unhide every column before re-hiding the ones this layout wants. Hidden
        # state is sticky and column meanings move between layouts: the previous
        # geometry parked its hidden source at AA, which is where DST's rank now
        # lives, so without this the whole DST block stays invisible.
        {"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": 0, "endIndex": width},
            "properties": {"hiddenByUser": False},
            "fields": "hiddenByUser",
        }},
        # Unfreeze before merging: a merge may not straddle the frozen boundary, and
        # an earlier layout froze column A.
        {"updateSheetProperties": {
            "properties": {"sheetId": sheet_id, "gridProperties": {
                "frozenRowCount": HEADER_ROW + 1, "frozenColumnCount": 0}},
            "fields": "gridProperties(frozenRowCount,frozenColumnCount)",
        }},
        # Banner merged across the board from B1, leaving A1 for the reset box.
        {"mergeCells": {"range": rng(0, 1, 1, last_visible), "mergeType": "MERGE_ALL"}},
        # The reset checkbox itself. One click, always on screen, and the script
        # backs the drafted log up before clearing it so it can be taken back.
        {"setDataValidation": {
            "range": rng(0, 1, 0, 1),
            "rule": {"condition": {"type": "BOOLEAN"}, "showCustomUi": True},
        }},
        {"repeatCell": {
            "range": rng(0, 1, 0, 1),
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.97, "green": 0.85, "blue": 0.85},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,horizontalAlignment)",
        }},
        {"repeatCell": {
            "range": rng(0, 1, 0, width),
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": True, "fontSize": 12},
                "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(textFormat,verticalAlignment)",
        }},
        {"repeatCell": {
            "range": rng(HEADER_ROW, HEADER_ROW + 1, 0, width),
            "cell": {"userEnteredFormat": {
                "backgroundColor": HEADER_GREY,
                "textFormat": {"bold": True},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }},
        # Hide the source columns: they exist only to feed the FILTER formulas.
        {"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": first_src, "endIndex": width},
            "properties": {"hiddenByUser": True},
            "fields": "hiddenByUser",
        }},
        # Control row: labels bold, the three inputs boxed so they read as fields.
        {"repeatCell": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, 0, last_visible),
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.94, "green": 0.94, "blue": 0.96},
                "textFormat": {"bold": True},
                "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
        }},
        {"repeatCell": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, STATUS_COL, last_visible),
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": False, "italic": True},
                "horizontalAlignment": "LEFT",
            }},
            "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
        }},
        {"mergeCells": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, STATUS_COL, last_visible),
            "mergeType": "MERGE_ALL",
        }},
    ]

    # The on-the-clock flag: merged, centred, and amber only while it has words in it.
    #
    # The first cut lit the whole merged status strip, which is what a cell-level
    # highlight does when the words live inside a longer string -- the pick number,
    # the round, the horizon and the TE premium pointer all went amber together, and
    # "just this bit" is not something a conditional format can express. So the words
    # get their own cell, and the rule paints that.
    on_clock_cell = f"${helper_letter}${FIRST_DATA_ROW + 7}"
    requests.append({"mergeCells": {
        "range": rng(CONTROL_ROW, CONTROL_ROW + 1, FLAG_COL, FLAG_END),
        "mergeType": "MERGE_ALL",
    }})
    # 8pt, not 9. A2:B2 is 60px and "YOUR PICK" in 9pt bold caps measures about 57
    # before padding, so Sheets clipped a letter off each end -- and it cannot overflow
    # into C2, which holds the Teams label. The two columns cannot be widened either:
    # they are the checkbox and Rank columns of every block below. The amber fill is
    # what carries this at a glance; the words only have to name it.
    requests.append({"repeatCell": {
        "range": rng(CONTROL_ROW, CONTROL_ROW + 1, FLAG_COL, FLAG_END),
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": True, "fontSize": 8},
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
    }})
    requests.append({"addConditionalFormatRule": {
        "rule": {
            "ranges": [rng(CONTROL_ROW, CONTROL_ROW + 1, FLAG_COL, FLAG_END)],
            "booleanRule": {
                "condition": {"type": "CUSTOM_FORMULA",
                              "values": [{"userEnteredValue": f"={on_clock_cell}"}]},
                "format": {
                    "backgroundColor": hex_to_rgb("#ffd75e"),
                    "textFormat": {"bold": True,
                                   "foregroundColor": hex_to_rgb("#3a2c00")},
                },
            },
        },
        "index": 0,
    }})

    # The TE premium toggle: a checkbox in the control row, styled like the two
    # typed inputs beside it so it reads as part of the same strip of controls.
    requests.append({"setDataValidation": {
        "range": rng(CONTROL_ROW, CONTROL_ROW + 1, TOGGLE_COL, TOGGLE_COL + 1),
        "rule": {"condition": {"type": "BOOLEAN"}, "showCustomUi": True},
    }})
    # A pick has to be a pick in this league. Not strict -- the status line already
    # says when it is out of range, and a hard reject on a cell someone is mid-edit in is
    # worse at draft speed than a warning triangle. No input message: the label
    # beside the cell is the whole explanation.
    teams_input = f"${column_letter(CONTROL_INPUT_COL)}${CONTROL_ROW + 1}"
    requests.append({"setDataValidation": {
        "range": rng(CONTROL_ROW, CONTROL_ROW + 1, PICK_INPUT_COL, PICK_INPUT_COL + 1),
        "rule": {
            "condition": {"type": "CUSTOM_FORMULA", "values": [{
                "userEnteredValue":
                    f"=AND(ISNUMBER({column_letter(PICK_INPUT_COL)}{CONTROL_ROW + 1}), "
                    f"{column_letter(PICK_INPUT_COL)}{CONTROL_ROW + 1}>=1, "
                    f"{column_letter(PICK_INPUT_COL)}{CONTROL_ROW + 1}<={teams_input})"}]},
            "strict": False,
        },
    }})

    # Labels right-aligned so each one hugs the input it names. They sit in wide
    # columns (Player, PPG) because a cell cannot be widened on its own -- see the
    # control-column comments -- and left-aligned they would float a column away
    # from the box they belong to.
    for column in (TEAMS_LABEL_COL, PICK_LABEL_COL):
        requests.append({"repeatCell": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, column, column + 1),
            "cell": {"userEnteredFormat": {"horizontalAlignment": "RIGHT"}},
            "fields": "userEnteredFormat(horizontalAlignment)",
        }})

    # The input cells: white, boxed, centred. Teams, Pick and the TE premium
    # checkbox -- the whole control row.
    for column in (CONTROL_INPUT_COL, PICK_INPUT_COL, TOGGLE_COL):
        requests.append({"repeatCell": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, column, column + 1),
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 1, "green": 1, "blue": 1},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,horizontalAlignment)",
        }})
        requests.append({"updateBorders": {
            "range": rng(CONTROL_ROW, CONTROL_ROW + 1, column, column + 1),
            "top": {"style": "SOLID", "color": {"red": .5, "green": .5, "blue": .5}},
            "bottom": {"style": "SOLID", "color": {"red": .5, "green": .5, "blue": .5}},
            "left": {"style": "SOLID", "color": {"red": .5, "green": .5, "blue": .5}},
            "right": {"style": "SOLID", "color": {"red": .5, "green": .5, "blue": .5}},
        }})

    for block in blocks:
        col = block["col"]
        # Conditional formats are bounded by this position's own length. A rule is
        # evaluated per cell of the ranges it is attached to, and running every one
        # of them to the full board height meant the 27-row quarterback block paid
        # for 250. Everything past here is blank in every state of the draft: the
        # block is a spilling FILTER and it cannot grow.
        block_last = FIRST_DATA_ROW + len(block["rows"])
        # Position banner across the block.
        requests.append({"mergeCells": {
            "range": rng(HEADER_ROW - 1, HEADER_ROW, col, col + BLOCK_WIDTH),
            "mergeType": "MERGE_ALL",
        }})
        requests.append({"repeatCell": {
            "range": rng(HEADER_ROW - 1, HEADER_ROW, col, col + BLOCK_WIDTH),
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.2, "green": 0.24, "blue": 0.29},
                "textFormat": {"bold": True, "fontSize": 11,
                               "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }})
        # One checkbox per row: ticking it is the whole draft interaction. The
        # onEdit script logs the player and clears the box again, so the same cell
        # is ready for whoever slides up into that row.
        #
        # Only as many boxes as the position has players. Running them to the full
        # board height put ~230 live checkboxes under the shortest blocks, all of
        # them doing nothing on a click but all of them drawing the eye.
        requests.append({"setDataValidation": {
            "range": rng(FIRST_DATA_ROW, FIRST_DATA_ROW + len(block["rows"]),
                         col, col + 1),
            "rule": {"condition": {"type": "BOOLEAN"}, "showCustomUi": True},
        }})
        for offset, pixels in zip(VISIBLE_OFFSETS, COLUMN_WIDTHS):
            requests.append({"updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                          "startIndex": col + offset, "endIndex": col + offset + 1},
                "properties": {"pixelSize": pixels},
                "fields": "pixelSize",
            }})
        # Hide the tier column only. It feeds the colour rules and the scarcity
        # strip; the number itself would be a fifth figure competing for the eye
        # when the band already says it. ADP is on the board now -- a tier only
        # means "interchangeable" if its players leave at similar times, and
        # inside one tier the spread can run fifty picks.
        requests.append({"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": col + TIER_COL, "endIndex": col + TIER_COL + 1},
            "properties": {"hiddenByUser": True},
            "fields": "hiddenByUser",
        }})
        # Rank centred and bold, PPG right-aligned with two decimals.
        requests.append({"repeatCell": {
            "range": rng(FIRST_DATA_ROW, total_rows, col + 1, col + 2),
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "CENTER", "textFormat": {"bold": True}}},
            "fields": "userEnteredFormat(horizontalAlignment,textFormat)",
        }})
        requests.append({"repeatCell": {
            "range": rng(FIRST_DATA_ROW, total_rows, col + PPG_COL, col + PPG_COL + 1),
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "RIGHT",
                "numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
            "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
        }})
        # ADP: one decimal, right-aligned so it columns up against PPG.
        requests.append({"repeatCell": {
            "range": rng(FIRST_DATA_ROW, total_rows, col + ADP_COL, col + ADP_COL + 1),
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "RIGHT",
                "numberFormat": {"type": "NUMBER", "pattern": "0.0"}}},
            "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
        }})
        # Team: centred, small and grey. It is context for the name beside it --
        # whose backup, whose stack -- not a figure to read down the column, and
        # at full weight it would compete with the numbers that are.
        requests.append({"repeatCell": {
            "range": rng(FIRST_DATA_ROW, total_rows, col + TEAM_COL, col + TEAM_COL + 1),
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "CENTER",
                "textFormat": {"fontSize": 9,
                               "foregroundColor": hex_to_rgb("#6b6b6b")}}},
            "fields": "userEnteredFormat(horizontalAlignment,textFormat)",
        }})
        # Left edge border so the blocks read as separate columns of players.
        requests.append({"updateBorders": {
            "range": rng(HEADER_ROW - 1, total_rows, col, col + BLOCK_WIDTH),
            "left": {"style": "SOLID", "width": 1,
                     "color": {"red": 0.6, "green": 0.6, "blue": 0.6}},
            "right": {"style": "SOLID", "width": 1,
                      "color": {"red": 0.6, "green": 0.6, "blue": 0.6}},
        }})

        # Tier colours. Keyed on the hidden tier column rather than painted onto
        # fixed rows, so the bands follow the players as picks collapse the column.
        # With the tier number no longer displayed, this is what shows the tiers.
        # The band stops after PPG. ADP and VONA carry their own colour -- green
        # when a player lasts to your next pick, red-to-green on value -- and a
        # tier fill underneath them would fight both.
        # See TIER_BANDS: the fills cycle, so this is five rules and not one per
        # tier. The blank test is load-bearing -- a band's own MOD of an empty cell
        # comes out at 4 and would paint every empty row past the end of the block
        # in the fifth colour.
        tier_col = column_letter(col + TIER_COL)
        tier_cell = f"${tier_col}{FIRST_DATA_ROW + 1}"
        for band in range(min(block["tiers"], len(TIER_BANDS))):
            requests.append({"addConditionalFormatRule": {
                "index": 0,
                "rule": {
                    "ranges": [rng(FIRST_DATA_ROW, block_last,
                                   col, col + PPG_COL + 1)],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue":
                                        f"=AND({tier_cell}<>\"\", "
                                        f"MOD({tier_cell}-1, {len(TIER_BANDS)})"
                                        f"={band})"}],
                        },
                        "format": {
                            "backgroundColor": hex_to_rgb(TIER_BANDS[band])},
                    },
                },
            }})

        # ADP green when the player is expected to still be there at your next
        # turn. ADP is a mean, so ADP past the target pick is the 50/50 line --
        # the same test the VONA column applies to the replacement pool, pointed
        # at the player himself. This is what stops a tier being read as "take
        # any of them": two tight ends on the same projection are not the same
        # pick when one of them lasts four more rounds.
        # Three states, not two. ADP is a mean with real spread around it, so a
        # player one pick either side of the horizon is not the near-certainty a
        # hard green/white line implies -- inside the scaled band around your turn he is
        # a coin flip, and the board now says so in amber instead of pretending.
        if target_cell:
            adp_letter = column_letter(col + ADP_COL)
            cell = f"${adp_letter}{FIRST_DATA_ROW + 1}"
            adp_range = rng(FIRST_DATA_ROW, block_last, col + ADP_COL, col + ADP_COL + 1)
            # The band scales with how many picks actually elapse before your turn,
            # because that is what the risk depends on. Fixed offsets were right
            # once a round and wrong the rest of it: picking 6th, at pick 41, only
            # TWO picks elapse, and a flat +20 painted nine players amber who were
            # near-certainties to survive both of them -- Etienne at 49.6, Higgins
            # at 57.6, McLaurin at 61.7. The same +20 is about right at the top of
            # round 1, where 22 picks elapse. One constant cannot be both.
            #
            # Floors keep it honest when the gap is tiny: even one intervening pick
            # can take a player, so nothing is ever declared safe by a hair.
            span = f"MAX(0, {target_cell}-{current_cell}-1)"
            safe = f"{target_cell}+MAX({ADP_SAFE_FLOOR:g}, {ADP_SAFE_RATE:g}*{span})"
            gone = f"{target_cell}-MAX({ADP_GONE_FLOOR:g}, {ADP_GONE_RATE:g}*{span})"
            for condition, fill, ink in (
                (f"=AND({cell}<>\"\", {cell}>={safe})",
                 "#d9ead3", "#1e5631"),                       # should still be there
                (f"=AND({cell}<>\"\", {cell}>{gone}, {cell}<{safe})",
                 "#fdebc8", "#7a5312"),                       # live, not safe
            ):
                requests.append({"addConditionalFormatRule": {
                    "index": 0,
                    "rule": {
                        "ranges": [adp_range],
                        "booleanRule": {
                            "condition": {"type": "CUSTOM_FORMULA",
                                          "values": [{"userEnteredValue": condition}]},
                            "format": {
                                "backgroundColor": hex_to_rgb(fill),
                                "textFormat": {"bold": True,
                                               "foregroundColor": hex_to_rgb(ink)},
                            },
                        },
                    },
                }})

        # Fall: signed, no decimals -- it is a count of picks, and a tenth of a pick
        # is not a thing. Coloured as INK rather than fill, and boolean rather than
        # a gradient: VONA already owns the one gradient on the board, and a second
        # one two columns away would read as a single smear across both.
        #
        # The threshold is a full round, off the Teams cell rather than a constant,
        # so "he is falling" means the same thing in a 10-team league as in a 14.
        # Below that the number is noise -- ADP is a mean and the ranking is one
        # opinion, and they are entitled to disagree by a few picks.
        requests.append({"repeatCell": {
            "range": rng(FIRST_DATA_ROW, total_rows, col + FALL_COL, col + FALL_COL + 1),
            "cell": {"userEnteredFormat": {
                "horizontalAlignment": "RIGHT",
                "numberFormat": {"type": "NUMBER", "pattern": "+0;-0;0"}}},
            "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
        }})
        fall_cell = f"${column_letter(col + FALL_COL)}{FIRST_DATA_ROW + 1}"
        fall_range = rng(FIRST_DATA_ROW, block_last,
                         col + FALL_COL, col + FALL_COL + 1)
        for condition, ink in (
            (f"=AND({fall_cell}<>\"\", {fall_cell}>={teams_input})", "#1e5631"),
            (f"=AND({fall_cell}<>\"\", {fall_cell}<=-{teams_input})", "#a32828"),
        ):
            requests.append({"addConditionalFormatRule": {
                "index": 0,
                "rule": {
                    "ranges": [fall_range],
                    "booleanRule": {
                        "condition": {"type": "CUSTOM_FORMULA",
                                      "values": [{"userEnteredValue": condition}]},
                        "format": {"textFormat": {
                            "bold": True, "foregroundColor": hex_to_rgb(ink)}},
                    },
                },
            }})

        # Value columns: signed, one decimal. VONA's colouring is set up after this
        # loop, across every block at once, so the scales are comparable. VOR is
        # left uncoloured on purpose -- two gradients side by side read as one
        # smear, and VONA is the one that changes as the draft moves.
        for offset in (VOR_COL, VAL_COL):
            requests.append({"repeatCell": {
                "range": rng(FIRST_DATA_ROW, total_rows, col + offset, col + offset + 1),
                "cell": {"userEnteredFormat": {
                    "horizontalAlignment": "RIGHT",
                    "numberFormat": {"type": "NUMBER", "pattern": "+0.0;-0.0;0.0"}}},
                "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
            }})
        val_range = rng(FIRST_DATA_ROW, total_rows, col + VAL_COL, col + VAL_COL + 1)

    # ------------------------------------------------------------------
    # My Team panel.
    # ------------------------------------------------------------------
    roster_first, roster_last = FIRST_DATA_ROW, FIRST_DATA_ROW + roster_body
    # A narrow gutter between the last position block and the panel, so the two
    # read as separate things without a whole column of white between them.
    requests.append({"updateDimensionProperties": {
        "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                  "startIndex": roster_col - 1, "endIndex": roster_col},
        "properties": {"pixelSize": 12},
        "fields": "pixelSize",
    }})
    for offset, pixels in enumerate(ROSTER_COLUMN_WIDTHS):
        requests.append({"updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS",
                      "startIndex": roster_col + offset,
                      "endIndex": roster_col + offset + 1},
            "properties": {"pixelSize": pixels},
            "fields": "pixelSize",
        }})
    # Banner, in the same navy as the position banners: it sits on the same line
    # and anything else there would read as a different kind of object.
    requests.append({"mergeCells": {
        "range": rng(HEADER_ROW - 1, HEADER_ROW, roster_col, roster_col + ROSTER_WIDTH),
        "mergeType": "MERGE_ALL",
    }})
    # A point smaller than the position banners: same navy, same weight, but this one
    # has four narrow columns to say four things in and they were being clipped.
    requests.append({"repeatCell": {
        "range": rng(HEADER_ROW - 1, HEADER_ROW, roster_col, roster_col + ROSTER_WIDTH),
        "cell": {"userEnteredFormat": {
            "backgroundColor": {"red": 0.2, "green": 0.24, "blue": 0.29},
            "textFormat": {"bold": True, "fontSize": 10,
                           "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
    }})
    # The starting eleven-ish gets a tint and the bench does not, so the line
    # between "this is my lineup" and "this is depth" is visible without reading
    # the slot labels.
    requests.append({"repeatCell": {
        "range": rng(roster_first, roster_first + starter_rows,
                     roster_col, roster_col + ROSTER_WIDTH),
        "cell": {"userEnteredFormat": {
            "backgroundColor": hex_to_rgb("#eef2f7")}},
        "fields": "userEnteredFormat(backgroundColor)",
    }})
    requests.append({"repeatCell": {
        "range": rng(roster_first, target_last, roster_col, roster_col + 1),
        "cell": {"userEnteredFormat": {
            "horizontalAlignment": "CENTER",
            "textFormat": {"bold": True, "fontSize": 9},
        }},
        "fields": "userEnteredFormat(horizontalAlignment,textFormat)",
    }})
    requests.append({"repeatCell": {
        "range": rng(roster_first, target_last, roster_col + 2, roster_col + 3),
        "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER"}},
        "fields": "userEnteredFormat(horizontalAlignment)",
    }})
    requests.append({"repeatCell": {
        "range": rng(roster_first, target_last, roster_col + 3, roster_col + 4),
        "cell": {"userEnteredFormat": {
            "horizontalAlignment": "RIGHT",
            "numberFormat": {"type": "NUMBER", "pattern": "0.00"}}},
        "fields": "userEnteredFormat(horizontalAlignment,numberFormat)",
    }})
    requests.append({"updateBorders": {
        "range": rng(HEADER_ROW - 1, roster_last, roster_col, roster_col + ROSTER_WIDTH),
        "left": {"style": "SOLID", "width": 1,
                 "color": {"red": 0.6, "green": 0.6, "blue": 0.6}},
        "right": {"style": "SOLID", "width": 1,
                  "color": {"red": 0.6, "green": 0.6, "blue": 0.6}},
        "bottom": {"style": "SOLID", "width": 1,
                   "color": {"red": 0.6, "green": 0.6, "blue": 0.6}},
    }})
    # A rule under the last starter: the bench begins here and slot labels alone
    # are easy to miss at draft speed.
    requests.append({"updateBorders": {
        "range": rng(roster_first, roster_first + starter_rows,
                     roster_col, roster_col + ROSTER_WIDTH),
        "bottom": {"style": "SOLID", "width": 2,
                   "color": {"red": 0.35, "green": 0.39, "blue": 0.45}},
    }})
    # An unfilled starting slot in red. This is the panel's real job during a
    # draft: not admiring what you have, but seeing at a glance what you still
    # have to fill and how few rounds are left to fill it in.
    requests.append({"addConditionalFormatRule": {
        "index": 0,
        "rule": {
            "ranges": [rng(roster_first, roster_first + starter_rows,
                           roster_col, roster_col + ROSTER_WIDTH)],
            "booleanRule": {
                "condition": {"type": "CUSTOM_FORMULA", "values": [{
                    "userEnteredValue":
                        f"=${column_letter(roster_col + 1)}{roster_first + 1}=\"\""}]},
                "format": {
                    "backgroundColor": hex_to_rgb("#fce4e4"),
                    "textFormat": {"foregroundColor": hex_to_rgb("#9c3a3a")},
                },
            },
        },
    }})

    # The targets block, below the bench. Its header row is styled like a section
    # divider rather than a second banner -- it belongs to the panel, it is not a
    # panel of its own.
    requests.append({"repeatCell": {
        "range": rng(target_header, target_header + 1,
                     roster_col, roster_col + ROSTER_WIDTH),
        "cell": {"userEnteredFormat": {
            "backgroundColor": hex_to_rgb("#e7e0f2"),
            "textFormat": {"bold": True, "fontSize": 9,
                           "foregroundColor": hex_to_rgb("#5b21a8")},
            "horizontalAlignment": "LEFT",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
    }})
    # White and boxed, because unlike everything else in the panel these are cells
    # you type into. Same treatment as the control-row inputs, for the same reason.
    requests.append({"repeatCell": {
        "range": rng(target_first, target_last, roster_col + 1, roster_col + 2),
        "cell": {"userEnteredFormat": {
            "backgroundColor": {"red": 1, "green": 1, "blue": 1}}},
        "fields": "userEnteredFormat(backgroundColor)",
    }})
    requests.append({"updateBorders": {
        "range": rng(target_first, target_last, roster_col, roster_col + ROSTER_WIDTH),
        "left": {"style": "SOLID", "color": {"red": .6, "green": .6, "blue": .6}},
        "right": {"style": "SOLID", "color": {"red": .6, "green": .6, "blue": .6}},
        "bottom": {"style": "SOLID", "color": {"red": .6, "green": .6, "blue": .6}},
        "innerHorizontal": {"style": "DOTTED",
                            "color": {"red": .8, "green": .8, "blue": .8}},
    }})
    if names_col is not None:
        # Same dropdown the drafted log gets, and non-strict for the same reason:
        # typing four letters of a name beats scrolling for it, and a warning
        # triangle over a spelling the board disagrees with helps nobody.
        names_letter = column_letter(names_col)
        requests.append({"setDataValidation": {
            "range": rng(target_first, target_last, roster_col + 1, roster_col + 2),
            "rule": {
                "condition": {"type": "ONE_OF_RANGE", "values": [{
                    "userEnteredValue":
                        f"=${names_letter}${FIRST_DATA_ROW + 1}:${names_letter}"}]},
                "showCustomUi": True,
                "strict": False,
            },
        }})
    # A target already off the board, struck through. Otherwise the list quietly
    # becomes a record of players you did not get.
    if drafted_col is not None:
        gone = (f"${column_letter(drafted_col)}${FIRST_DATA_ROW + 1}:"
                f"${column_letter(drafted_col)}${DRAFTED_LIMIT}")
        typed = f"${column_letter(roster_col + 1)}{target_first + 1}"
        requests.append({"addConditionalFormatRule": {
            "index": 0,
            "rule": {
                "ranges": [rng(target_first, target_last,
                               roster_col, roster_col + ROSTER_WIDTH)],
                "booleanRule": {
                    "condition": {"type": "CUSTOM_FORMULA", "values": [{
                        "userEnteredValue":
                            f"=AND({typed}<>\"\", COUNTIF({gone}, {typed})>0)"}]},
                    "format": {"textFormat": {
                        "strikethrough": True,
                        "foregroundColor": hex_to_rgb("#9a9a9a")}},
                },
            },
        }})

    # Value colouring, applied to every block's Val column in one rule apiece so
    # the scale is shared: a +4.6 RB has to look greener than a +1.0 TE, which it
    # would not if each column were scaled to its own min and max.
    val_ranges = [rng(FIRST_DATA_ROW, FIRST_DATA_ROW + len(b["rows"]),
                      b["col"] + VAL_COL, b["col"] + VAL_COL + 1) for b in blocks]
    # Red through white to green, white pinned to zero -- the point where a player
    # is worth exactly what the position will still offer later. The ends are fixed
    # rather than scaled to the data: replacement-level players run to -14 while the
    # best on the board is nearer +5, so MIN/MAX handed almost the whole scale to
    # the reds and left every positive looking the same pale green. Anything past
    # the clamp just saturates, which is fine -- by then the sign is the message.
    requests.append({"addConditionalFormatRule": {
        "index": 0,
        "rule": {
            "ranges": val_ranges,
            "gradientRule": {
                "minpoint": {"color": hex_to_rgb("#e88b8b"),
                             "type": "NUMBER", "value": f"-{VALUE_SCALE}"},
                "midpoint": {"color": {"red": 1, "green": 1, "blue": 1},
                             "type": "NUMBER", "value": "0"},
                "maxpoint": {"color": hex_to_rgb("#57bb63"),
                             "type": "NUMBER", "value": f"{VALUE_SCALE}"},
            },
        },
    }})
    # The best value on the board *for your roster*, in purple. Added after the
    # gradient so it lands at index 0 and wins: rules are tried in order and the
    # first match paints.
    #
    # It reads VOR, not VONA. A bare max over four VONA columns compares numbers
    # that were never built to be compared -- each one is measured against the next
    # player at its own position, so a deep position can never win however good its
    # best player is. VOR is measured in points against a single bar per position,
    # which is what makes a cross-position "take this" mean anything.
    #
    # The roster awareness is now inside VOR itself rather than a gate in front of
    # it: once a position is full its bar becomes the starter of yours the player
    # would displace, so a fourth running back loses this comparison by arithmetic.
    # Each block publishes its best VOR into a hidden cell -- -999 only if the
    # position has been drafted out entirely -- and the winner is the max of those.
    # An empty roster leaves every bar at league replacement, which is exactly the
    # original behaviour.
    #
    # Written relative in column and absolute in row, so the same rule lands on
    # each block's own cells as it is applied to each range.
    first_vor = column_letter(blocks[0]["col"] + VOR_COL)
    best_row = total_rows + 3            # 1-indexed, third row past the data
    my_best = f"{column_letter(blocks[0]['col'] + TIER_COL)}${best_row}"
    all_best = ", ".join(
        f"${column_letter(b['col'] + TIER_COL)}${best_row}" for b in blocks)
    vor_ranges = [rng(FIRST_DATA_ROW, FIRST_DATA_ROW + len(b["rows"]),
                      b["col"] + VOR_COL, b["col"] + VOR_COL + 1) for b in blocks]
    requests.append({"addConditionalFormatRule": {
        "index": 0,
        "rule": {
            "ranges": vor_ranges,
            "booleanRule": {
                "condition": {"type": "CUSTOM_FORMULA", "values": [{
                    "userEnteredValue":
                        f"=AND({first_vor}{FIRST_DATA_ROW + 1}<>\"\", "
                        f"{my_best}>-900, "
                        f"{first_vor}{FIRST_DATA_ROW + 1}={my_best}, "
                        f"{my_best}=MAX({all_best}))"}]},
                "format": {
                    "backgroundColor": hex_to_rgb("#b18cd9"),
                    "textFormat": {"bold": True},
                },
            },
        },
    }})

    # Targets, wherever they sit on the board. Deliberately ink and weight rather
    # than a fill: the tier band underneath is information too, and a solid block
    # of colour over the name would trade one signal for another instead of
    # adding one.
    for block in blocks:
        name_cell = f"${column_letter(block['col'] + 2)}{FIRST_DATA_ROW + 1}"
        requests.append({"addConditionalFormatRule": {
            "index": 0,
            "rule": {
                "ranges": [rng(FIRST_DATA_ROW,
                               FIRST_DATA_ROW + len(block["rows"]),
                               block["col"] + 1, block["col"] + TEAM_COL + 1)],
                "booleanRule": {
                    "condition": {"type": "CUSTOM_FORMULA", "values": [{
                        "userEnteredValue":
                            f"=AND({name_cell}<>\"\", "
                            f"COUNTIF({targets}, {name_cell})>0)"}]},
                    "format": {"textFormat": {
                        "bold": True, "italic": True,
                        "foregroundColor": hex_to_rgb("#5b21a8")}},
                },
            },
        }})
    # The banner highlight answers a different question from the purple cell, so it
    # gets its own colour. Purple means "best value on the board" and lives on the
    # Val column; painting a banner purple would only repeat where that cell
    # already is. Amber means "this position is about to lose its tier" -- the one
    # thing VOR cannot say, which is the reason the strip exists at all.
    #
    # Applied to each banner's anchor cell only: the banner is a MERGE_ALL, so the
    # anchor paints the whole span, and a rule spanning every column would
    # have its relative reference shift once per column inside the merge. The
    # reference is relative in column and absolute in row, so it lands on each
    # block's own hidden urgency cell.
    banner_ranges = [rng(HEADER_ROW - 1, HEADER_ROW, b["col"], b["col"] + 1)
                     for b in blocks]
    urgency_row = total_rows + 1                      # 1-indexed, one past the data
    mine = f"{column_letter(blocks[0]['col'] + TIER_COL)}${urgency_row}"
    all_urgency = ", ".join(
        f"${column_letter(b['col'] + TIER_COL)}${urgency_row}" for b in blocks)
    requests.append({"addConditionalFormatRule": {
        "index": 0,
        "rule": {
            "ranges": banner_ranges,
            "booleanRule": {
                "condition": {"type": "CUSTOM_FORMULA", "values": [{
                    "userEnteredValue":
                        f"=AND({mine}>0, {mine}=MAX({all_urgency}))"}]},
                "format": {
                    "backgroundColor": hex_to_rgb("#e8a33d"),
                    # The banner is white-on-navy by default, which would vanish
                    # on this fill, so the rule repaints the text as well.
                    "textFormat": {"bold": True,
                                   "foregroundColor": hex_to_rgb("#20262e")},
                },
            },
        },
    }})

    # A ticked row, struck through and greyed, from the moment it is ticked.
    #
    # This is the only part of a pick that can be instant. Everything else waits on
    # Google: the checkbox writes locally and the browser paints this rule itself,
    # but the script that logs the player and collapses the column runs on a trigger
    # Google dispatches when it gets round to it -- measured at 400-800ms from the
    # click, with the recalculation behind it taking no measurable time at all. So
    # the row cannot leave any sooner, and there is no version of this board where
    # it can. What it can do is stop looking like nothing happened.
    #
    # Text and not fill: the tier band underneath is information, and this rule sets
    # no background so the band still shows through it. Appended last so it lands
    # ahead of every other rule -- a row on its way off the board should not also be
    # reading as a target or as the best value on it.
    for block in blocks:
        box = f"${column_letter(block['col'])}{FIRST_DATA_ROW + 1}"
        requests.append({"addConditionalFormatRule": {
            "index": 0,
            "rule": {
                "ranges": [rng(FIRST_DATA_ROW, FIRST_DATA_ROW + len(block["rows"]),
                               block["col"], block["col"] + BLOCK_WIDTH)],
                "booleanRule": {
                    "condition": {"type": "CUSTOM_FORMULA",
                                  "values": [{"userEnteredValue": f"={box}=TRUE"}]},
                    "format": {"textFormat": {
                        "strikethrough": True,
                        "foregroundColor": hex_to_rgb("#9a9a9a")}},
                },
            },
        }})

    # The drafted log gets a dropdown fed by the hidden "All Players" column, which
    # until now was written on every run and read by nothing at all. It turns the
    # log into a second way to record a pick: type a few letters of a name instead
    # of hunting for the row, which is what you actually want when someone takes a
    # player from a part of the board you have already scrolled past.
    #
    # Not strict -- the Apps Script writes into this column itself, and a warning
    # triangle on a name the board spelled slightly differently is worse than no
    # validation at all.
    if drafted_col is not None and names_col is not None:
        names_letter = column_letter(names_col)
        requests.append({"setDataValidation": {
            "range": rng(FIRST_DATA_ROW, DRAFTED_LIMIT, drafted_col, drafted_col + 1),
            "rule": {
                "condition": {"type": "ONE_OF_RANGE", "values": [{
                    "userEnteredValue":
                        f"=${names_letter}${FIRST_DATA_ROW + 1}:${names_letter}"}]},
                "showCustomUi": True,
                "strict": False,
            },
        }})

    return requests


# --------------------------------------------------------------------------
# The Apps Script behind the checkboxes. Generated from the same geometry
# constants the sheet is built from, so the two cannot drift apart.
# --------------------------------------------------------------------------

GS_TEMPLATE = '''/**
 * Draft Board -- one-click "player is gone", one-click "start over".
 *
 * Paste into the spreadsheet's Extensions > Apps Script editor and save. No
 * triggers to install and no authorisation prompt: onEdit and onOpen are simple
 * triggers, which may edit their own spreadsheet.
 *
 * The board's geometry is baked in below, written by the generator out of the same
 * constants it laid the board out with, so the two cannot drift. layout() is still
 * here and still the authority: it finds the header row, treats every bare "Rank"
 * header as a position block -- checkbox one column left, player name one column
 * right -- and locates the log by its "Drafted" header. Anything the baked answer
 * gets wrong falls through to it, so rebuilding the board with different columns,
 * widths or row offsets still does NOT require pasting this again.
 *
 * Ticking the checkbox beside a player writes their name into the hidden drafted
 * log and immediately unticks the box. Each position block is a FILTER over a
 * hidden source that excludes every name in that log, so the player vanishes from
 * their own column and everyone below slides up -- and because the box is already
 * clear, whoever slides into that row is not swept up with them.
 *
 * Ticking A1 resets the board. It copies the log to a backup column before
 * clearing, so a stray click is recoverable from the Draft Board menu.
 */

var SHEET_NAME = '%(sheet_name)s';
var RESET_ROW = 1, RESET_COL = 1;   // the reset box in A1
var SEARCH_ROWS = 8;                // the header row is within the first few

/**
 * Where everything is, written in at build time.
 *
 * Working it out at runtime instead is what layout() does, and it costs two round
 * trips to the spreadsheet -- getLastColumn(), then an eight-row probe across the
 * full width of the sheet -- before the script has done anything at all. That is
 * the wrong price on the one path whose whole point is that a tick feels instant,
 * and it was paid on every pick.
 *
 * Nothing trusts this blindly. The guards below fall back to layout() when the
 * baked columns do not match the edit, and the read that fetches the player's name
 * checks that it landed on a column headed "Player" before drafting anybody.
 */
var BAKED = %(baked)s;


/**
 * Work out where everything is by reading the header row.
 * Returns {headerRow, firstDataRow, checkboxCols, nameOffset, drafted, backup}
 * or null.
 */
function layout(sheet) {
  var probe = sheet.getRange(1, 1, SEARCH_ROWS, sheet.getLastColumn()).getValues();

  var headerRow = -1;
  for (var r = 0; r < probe.length && headerRow === -1; r++) {
    for (var c = 0; c < probe[r].length; c++) {
      if (probe[r][c] === 'Player') { headerRow = r; break; }
    }
  }
  if (headerRow === -1) return null;

  var head = probe[headerRow];
  var checkboxCols = [], drafted = -1, backup = -1;
  for (var i = 0; i < head.length; i++) {
    // A bare "Rank" heads a visible block. The hidden source columns are named
    // "QB Rank", "RB Rank" and so on, so they never match here.
    if (head[i] === 'Rank' && i > 0) checkboxCols.push(i);  // 1-indexed checkbox
    if (head[i] === 'Drafted') drafted = i + 1;
    if (head[i] === 'Drafted (backup)') backup = i + 1;
  }
  if (!checkboxCols.length || drafted === -1) return null;

  return {
    headerRow: headerRow + 1,            // 1-indexed, the way the sheet counts rows
    firstDataRow: headerRow + 2,
    checkboxCols: checkboxCols,
    nameOffset: 2,                       // checkbox | Rank | Player
    drafted: drafted,
    backup: backup === -1 ? drafted + 1 : backup
    // No logLastRow: nothing on the sheet says where the log ends, so logRange()
    // falls back to getLastRow() for a geometry that came from here.
  };
}


function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Draft Board')
    .addItem('Undo last pick', 'undoLastPick')
    .addItem('Undo reset (restore cleared board)', 'undoReset')
    .addSeparator()
    .addItem('Reset board (put everyone back)', 'resetBoard')
    .addToUi();
}


function onEdit(e) {
  if (!e || !e.range) return;
  if (e.value !== 'TRUE') return;              // only care about a box being ticked

  var range = e.range;
  var row = range.getRow();
  var col = range.getColumn();
  var sheet = range.getSheet();
  if (sheet.getName() !== SHEET_NAME) return;

  if (row === RESET_ROW && col === RESET_COL) {
    // Not a hot path -- it asks the sheet where everything is, and if the answer
    // is unreadable the baked geometry beats doing nothing.
    var reset = layout(sheet) || BAKED;
    range.setValue(false);                     // clear the box, then wipe the board
    clearBoard(sheet, reset, true);
    SpreadsheetApp.getActive().toast(
      'Board reset. Draft Board > Undo reset puts the picks back.');
    return;
  }

  // The TE premium toggle is the board's other checkbox and it is not a pick. It
  // is named explicitly so that it returns here for nothing, instead of falling
  // into the re-read below every time somebody switches scoring.
  if (row === BAKED.toggleRow && col === BAKED.toggleCol) return;

  // The guards themselves run off the baked geometry, before any read.
  //
  // The re-read is for a board that has been rebuilt with different columns, which
  // is the only thing that makes the baked answer wrong -- so it is on the path
  // that would otherwise silently do nothing on a click, and off the path of every
  // ordinary edit. Everything else that is not a pick has already returned: a name
  // typed into the log or a target list is not the string TRUE, and the reset box
  // and the toggle are both handled above.
  var L = BAKED;
  if (row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) {
    L = layout(sheet);                         // the baked answer does not fit
    if (!L) return;
    if (row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) return;
  }

  // BOTH reads before EITHER write, and that ordering is the whole difference
  // between one recalculation of the board per pick and two.
  //
  // Reads and writes do not interleave for free: a read flushes whatever writes
  // are queued, and the flush recalculates every FILTER, every value column and
  // every conditional format on the board. Unticking the box and THEN reading the
  // log -- which is what this used to do -- forced that recalculation in the
  // middle of the script, and the writes at the end of it forced a second. Reads
  // first, and the untick and the logged name flush together, once.
  var name = readName(sheet, L, row, col);
  if (name === null) {                         // not a name column after all
    L = layout(sheet);
    if (!L || row < L.firstDataRow || L.checkboxCols.indexOf(col) === -1) return;
    name = readName(sheet, L, row, col);
    if (name === null) return;
  }
  var logged = name ? logRange(sheet, L, L.drafted).getValues() : null;

  // Clear the box first. If the name came back empty -- a click on a blank row
  // past the end of a position -- that is all that happens.
  range.setValue(false);
  if (!name) return;

  // No toast, deliberately. It was worth one round trip while the wait was being
  // measured -- 400-800ms from click to toast, with the row collapsing straight
  // after it, which put the whole cost in Google's trigger dispatch and this
  // script rather than in the board's formulas. It is not worth one per pick to
  // repeat something the vanishing row already says.
  markDrafted(sheet, L, name, logged);
}


/**
 * The player beside a ticked checkbox, in a read that also proves the column.
 *
 * The range runs from the header row down to the player's own cell, so the first
 * value it returns is that column's heading and the last is the name. A block's
 * name column is headed "Player"; anything else means the geometry is pointing
 * somewhere it should not be, and null says so rather than drafting whatever
 * happened to be sitting there. It costs no more than reading the one cell: a
 * couple of hundred values down a single column is one round trip either way.
 */
function readName(sheet, L, row, col) {
  var span = sheet.getRange(L.headerRow, col + L.nameOffset,
                            row - L.headerRow + 1, 1).getValues();
  if (span[0][0] !== 'Player') return null;
  return span[span.length - 1][0];
}


/**
 * The drafted log, or its backup. Its last row is a build-time constant, so the
 * baked geometry knows it and getLastRow() -- one more round trip on the hot path
 * -- is only needed for a geometry that came off the sheet.
 */
function logRange(sheet, L, column) {
  var lastRow = L.logLastRow || Math.max(sheet.getLastRow(), L.firstDataRow);
  return sheet.getRange(L.firstDataRow, column, lastRow - L.firstDataRow + 1, 1);
}


/**
 * A whole column of the board proper, which runs PAST the end of the log: there
 * are more players than a draft has picks, so the checkboxes go further down than
 * the log does and a sweep over them cannot use logRange's bound.
 */
function boardRange(sheet, L, column) {
  var lastRow = Math.max(sheet.getLastRow(), L.firstDataRow);
  return sheet.getRange(L.firstDataRow, column, lastRow - L.firstDataRow + 1, 1);
}


/**
 * Append a name to the drafted log, if it is not already there.
 *
 * The log's values come in from onEdit, which read them before it wrote anything
 * -- see the note there on why the order matters. Read here instead when there is
 * nothing to pass in, which is every caller except the checkbox.
 */
function markDrafted(sheet, L, name, logged) {
  if (!logged) logged = logRange(sheet, L, L.drafted).getValues();

  var firstEmpty = -1;
  for (var i = 0; i < logged.length; i++) {
    if (logged[i][0] === name) return;         // already off the board
    if (firstEmpty === -1 && logged[i][0] === '') firstEmpty = i;
  }
  var target = firstEmpty === -1 ? logged.length : firstEmpty;
  sheet.getRange(L.firstDataRow + target, L.drafted).setValue(name);
}


/** Empty the drafted log and untick every box. Optionally back the log up first. */
function clearBoard(sheet, L, backup) {
  var log = logRange(sheet, L, L.drafted);
  if (backup) {
    var values = log.getValues();
    logRange(sheet, L, L.backup).clearContent();
    sheet.getRange(L.firstDataRow, L.backup, values.length, 1).setValues(values);
  }
  log.clearContent();

  // Clear any box left ticked -- only possible if a paste bypassed onEdit.
  //
  // Only where a checkbox actually exists. setValue(false) down the whole column
  // wrote a literal FALSE into every row past the end of the position -- ~190
  // under quarterback, ~155 under tight end -- so the one control that is always
  // on screen left the board covered in the word FALSE. Clearing the cells that
  // hold no checkbox, and unticking only the ones that do, leaves it blank.
  for (var i = 0; i < L.checkboxCols.length; i++) {
    var column = boardRange(sheet, L, L.checkboxCols[i]);
    var rules = column.getDataValidations();
    var out = [];
    for (var r = 0; r < rules.length; r++) {
      var rule = rules[r][0];
      var isBox = rule && rule.getCriteriaType() ===
                  SpreadsheetApp.DataValidationCriteria.CHECKBOX;
      out.push([isBox ? false : '']);
    }
    column.setValues(out);
  }
}


function active() {
  var sheet = SpreadsheetApp.getActive().getSheetByName(SHEET_NAME);
  return {sheet: sheet, L: layout(sheet) || BAKED};
}


function undoLastPick() {
  var a = active();
  var values = logRange(a.sheet, a.L, a.L.drafted).getValues();
  for (var i = values.length - 1; i >= 0; i--) {
    if (values[i][0] !== '') {
      a.sheet.getRange(a.L.firstDataRow + i, a.L.drafted).clearContent();
      SpreadsheetApp.getActive().toast(values[i][0] + ' is back on the board');
      return;
    }
  }
  SpreadsheetApp.getActive().toast('No picks to undo');
}


function undoReset() {
  var a = active();
  var backup = logRange(a.sheet, a.L, a.L.backup).getValues();
  var picks = backup.filter(function (row) { return row[0] !== ''; }).length;
  if (!picks) {
    SpreadsheetApp.getActive().toast('Nothing to restore');
    return;
  }
  var log = logRange(a.sheet, a.L, a.L.drafted);
  log.clearContent();
  a.sheet.getRange(a.L.firstDataRow, a.L.drafted, backup.length, 1).setValues(backup);
  SpreadsheetApp.getActive().toast('Restored ' + picks + ' pick(s)');
}


function resetBoard() {
  var ui = SpreadsheetApp.getUi();
  var answer = ui.alert('Reset the board?',
                        'This clears every pick and puts all players back.',
                        ui.ButtonSet.YES_NO);
  if (answer !== ui.Button.YES) return;

  var a = active();
  clearBoard(a.sheet, a.L, true);
  SpreadsheetApp.getActive().toast('Board reset');
}
'''


def write_apps_script(path, blocks, drafted_col, backup_col):
    """Write the onEdit script, with the board's geometry baked into it.

    The script can still work the geometry out for itself -- layout() is in there,
    and it is what a rebuild that moves columns falls through to, so the file does
    not need pasting into the editor again. But doing it on every tick of a
    checkbox cost two round trips to the spreadsheet before the pick could even be
    read, so the answer is written in as well, out of the same constants that
    placed the columns. One source of truth, no drift, no lookup on the hot path.
    """
    # Written out by hand rather than through json.dumps, which quotes every key
    # and puts a four-line array in the middle of it. This is a file someone opens.
    columns = ", ".join(str(block["col"] + 1) for block in blocks)
    baked = "\n".join([
        "{",
        f"  headerRow: {HEADER_ROW + 1},          "
        f"// 1-indexed, the way the sheet counts",
        f"  firstDataRow: {FIRST_DATA_ROW + 1},",
        f"  checkboxCols: [{columns}],   // one per position block",
        f"  nameOffset: 2,             // checkbox | Rank | Player",
        f"  toggleRow: {CONTROL_ROW + 1}, toggleCol: {TOGGLE_COL + 1},"
        f"   // the TE premium box",
        f"  drafted: {drafted_col + 1},",
        f"  backup: {backup_col + 1},",
        f"  logLastRow: {DRAFTED_LIMIT}          // DRAFTED_LIMIT",
        "}",
    ])
    source = GS_TEMPLATE % {"sheet_name": SHEET_TITLE, "baked": baked}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(source)
    return path


APPS_SCRIPT_MANIFEST = {
    "timeZone": "America/New_York",
    "dependencies": {},
    "exceptionLogging": "STACKDRIVER",
    "runtimeVersion": "V8",
}


def push_apps_script(script_id, path):
    """Upload the .gs into the spreadsheet's bound Apps Script project.

    Needs three one-time things, all of them the account holder's to grant: the
    Apps Script API switched on for the account, the same API enabled in the Cloud
    project behind the OAuth client, and a re-consent picking up the new
    script.projects scope. After that this replaces the copy-paste entirely.
    """
    from google.auth.transport.requests import AuthorizedSession

    credentials = google_credentials()
    if not credentials:
        print("Google credentials not configured; run --setup-auth.", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    body = {"files": [
        {"name": "appsscript", "type": "JSON",
         "source": json.dumps(APPS_SCRIPT_MANIFEST, indent=2)},
        {"name": "Code", "type": "SERVER_JS", "source": source},
    ]}

    session = AuthorizedSession(credentials)
    response = session.put(
        f"https://script.googleapis.com/v1/projects/{script_id}/content",
        json=body, timeout=60)

    if response.status_code == 200:
        print(f"pushed {os.path.basename(path)} to script {script_id}",
              file=sys.stderr)
        return 0

    print(f"push failed [{response.status_code}]: {response.text[:400]}",
          file=sys.stderr)
    if response.status_code == 403:
        print("\nIf this mentions the Apps Script API, switch it on at\n"
              "  https://script.google.com/home/usersettings\n"
              "and enable it for the Cloud project at\n"
              "  https://console.cloud.google.com/apis/library/script.googleapis.com",
              file=sys.stderr)
    return 1


def write_sheet(grid, blocks, total_rows, width, drafted_col, url, name,
                keep_picks=False):
    """Create or rewrite the draft board. Returns the spreadsheet URL, or None."""
    import gspread

    credentials = google_credentials()
    if not credentials:
        print("\nGoogle credentials not configured.\n"
              f"Run:  python {os.path.basename(__file__)} --setup-auth\n"
              f"and sign in as {SHEETS_ACCOUNT}.", file=sys.stderr)
        return None

    client = gspread.authorize(credentials)
    spreadsheet, created = open_spreadsheet(client, url, name)
    if created:
        print(f"created new spreadsheet '{name}'", file=sys.stderr)
        # Pin the URL so later runs open it directly; searching Drive by name needs
        # a broader scope than drive.file gives us.
        save_env_value(ENV_PATH, "TIERS_SHEET_URL", spreadsheet.url)
        print(f"saved TIERS_SHEET_URL to {ENV_PATH}", file=sys.stderr)

    drafted_letter = column_letter(drafted_col)
    last_row = max(total_rows, DRAFTED_LIMIT)

    try:
        worksheet = spreadsheet.worksheet(SHEET_TITLE)
        end_row = max(last_row, worksheet.row_count)
        end_col = column_letter(max(width + 5, worksheet.col_count))
        if keep_picks:
            # Everything except the drafted log, so a rebuild mid-draft keeps who
            # is already off the board. Only safe when the layout has not moved.
            worksheet.batch_clear([
                f"A1:{column_letter(drafted_col - 1)}{end_row}",
                f"{drafted_letter}1:{drafted_letter}{HEADER_ROW + 1}",
                f"{column_letter(width)}1:{end_col}{end_row}",
            ])
        else:
            # A rebuild starts from a clean board: if the column layout changed,
            # a stale log would sit in a column that now means something else.
            worksheet.batch_clear([f"A1:{end_col}{end_row}"])
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=SHEET_TITLE, rows=total_rows + 20, cols=width + 2)

    if worksheet.row_count < DRAFTED_LIMIT or worksheet.col_count < width:
        worksheet.resize(rows=max(total_rows + 20, DRAFTED_LIMIT),
                         cols=max(width, worksheet.col_count))

    # Unmerge before writing, not just before formatting: a merge left over from
    # the last run swallows every value written into it but the top-left cell, and
    # the banner now starts at B1 inside the old A1 merge.
    spreadsheet.batch_update(
        {"requests": [{"unmergeCells": {"range": {"sheetId": worksheet.id}}}]})

    # Everything left of the drafted log, then the log's own headers, so the
    # blank cells in the payload never overwrite live picks or their backup.
    worksheet.update(values=[row[:drafted_col] for row in grid],
                     range_name="A1", value_input_option="USER_ENTERED")
    worksheet.update(values=[grid[HEADER_ROW][drafted_col:]],
                     range_name=f"{drafted_letter}{HEADER_ROW + 1}",
                     value_input_option="RAW")

    clear_conditional_formats(spreadsheet, worksheet.id)
    spreadsheet.batch_update({"requests": format_requests(
        worksheet.id, blocks, total_rows, width, drafted_col,
        blocks[-1]["src"] + SOURCE_WIDTH)})

    # Drop the empty default tab a new spreadsheet ships with. Checked on every run,
    # not just creation, since a failed first run can leave it behind. Only an
    # untouched default tab qualifies -- never a tab with anything in it.
    for other in spreadsheet.worksheets():
        if other.title != SHEET_TITLE and other.title.replace(" ", "") == "Sheet1":
            try:
                if not any(any(cell for cell in row) for row in other.get_all_values()):
                    spreadsheet.del_worksheet(other)
            except Exception:
                pass

    return spreadsheet.url


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="rankings CSV to read")
    ap.add_argument("--sheet-name", default=DEFAULT_SPREADSHEET_NAME,
                    help="spreadsheet name to create on a first run")
    ap.add_argument("--sheet-url", default=None,
                    help="open this spreadsheet instead of TIERS_SHEET_URL")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="only rank this many players overall (0 = all)")
    ap.add_argument("--teams", type=int, default=DEFAULT_TEAMS,
                    help="teams in the draft; seeds the sheet input (default 12)")
    ap.add_argument("--pick", type=int, default=DEFAULT_PICK,
                    help="your pick in the draft order, 1 to --teams; seeds the sheet "
                         f"input (default {DEFAULT_PICK}, editable on the board)")
    ap.add_argument("--break-p", type=float, default=BREAK_P,
                    help=f"how improbable a gap must be to end a tier (default "
                         f"{BREAK_P:g}; lower = fewer, larger tiers)")
    # These three are rank-basis only. The ppg basis reads its own PPG_GAP_WINDOW
    # / MIN_DROP / MAX_DROP, because a window in rank slots and a floor in rank
    # slots mean nothing to a series of points. Said out loud rather than left for
    # someone to discover by passing --min-gap and watching nothing happen.
    ap.add_argument("--gap-window", type=int, default=GAP_WINDOW,
                    help="[--tier-on rank only] gaps either side used to judge a "
                         f"gap (default {GAP_WINDOW})")
    ap.add_argument("--min-gap", type=int, default=MIN_GAP,
                    help="[--tier-on rank only] smallest rank gap that may end a "
                         f"tier (default {MIN_GAP})")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP,
                    help="[--tier-on rank only] rank gap that always ends a tier "
                         f"(default {MAX_GAP})")
    ap.add_argument("--tier-max-spread", type=float, default=TIER_MAX_SPREAD,
                    help="[--tier-on ppg only] widest PPG spread one tier may hold "
                         f"before it is split (default {TIER_MAX_SPREAD:g}; 0 "
                         "disables the cap and leaves the gap test alone)")
    ap.add_argument("--tier-split-drop", type=float, default=TIER_SPLIT_DROP,
                    help="[--tier-on ppg only] smallest PPG drop a spread split may "
                         f"land on (default {TIER_SPLIT_DROP:g}); a too-wide tier "
                         "with no drop this large is left wide")
    ap.add_argument("--games", type=int, default=GAMES, help="games in the season (default 17)")
    ap.add_argument("--receptions", default=DEFAULT_RECEPTIONS_CSV,
                    help="CSV with projected receptions (any file with a name and a "
                         "receptions column); makes the TE premium per-player instead "
                         "of a flat multiplier")
    ap.add_argument("--te-premium", type=float, default=TE_PREMIUM,
                    help="extra points per TE reception over standard PPR "
                         "(default 0.5, i.e. 1.5 PPR; 0 disables)")
    ap.add_argument("--te-points-per-reception", type=float, default=TE_POINTS_PER_RECEPTION,
                    help="PPR points per catch used to estimate receptions when none "
                         "are supplied (default 2.5)")
    ap.add_argument("--starters", default=",".join(f"{k}:{v:g}" for k, v in LINEUP.items()),
                    help="mandatory starting slots per team, e.g. QB:1,RB:2,WR:2,TE:1")
    ap.add_argument("--bench", type=int, default=ROSTER_BENCH_ROWS,
                    help=f"bench slots per team; sizes the MY TEAM panel past the "
                         f"starters (default {ROSTER_BENCH_ROWS})")
    ap.add_argument("--flex", type=float, default=FLEX_SLOTS,
                    help=f"flex slots per team, filled on merit from "
                         f"{'/'.join(FLEX_POSITIONS)} (default {FLEX_SLOTS:g})")
    ap.add_argument("--te-starters", type=int, default=None,
                    help="override the replacement tight end (e.g. 20 for TE20); by "
                         "default it is derived from the lineup, with the flex resolved "
                         "on projected points")
    ap.add_argument("--te-shift-model", choices=("curve", "linear"), default="curve",
                    help="how a tight end's gain becomes a move up the board: 'curve' "
                         "inverts a monotone value curve fitted to the RB/WR field, so "
                         "the exchange rate varies with depth the way the market's does; "
                         "'linear' uses one flat rate (default curve). Rank and ADP each "
                         "get their own fit -- picks and queue slots are not the same "
                         "scale")
    ap.add_argument("--te-slots-per-ppg", type=float, default=TE_SLOTS_PER_PPG,
                    help="board slots one PPG is worth under --te-shift-model linear "
                         f"(default {TE_SLOTS_PER_PPG:g}; the same rate is used for ADP "
                         "picks, which is exactly why linear is not the default; 0 "
                         "freezes both axes)")
    ap.add_argument("--positions", default=",".join(DEFAULT_BOARD_POSITIONS),
                    help="comma-separated positions that get a block on the board "
                         "(default QB,RB,WR,TE; everything else is still parsed and "
                         "still holds its rank slots, it just does not earn screen)")
    ap.add_argument("--tier-on", choices=TIER_BASES, default="ppg",
                    help="what tier boundaries are cut from: 'ppg' re-sorts each "
                         "position by premium-adjusted projected points and cuts on the "
                         "value drops, so a tier means 'these are interchangeable'; "
                         "'rank' uses gaps in PFF's overall rank instead, which measures "
                         "how many other positions sit in between (default ppg)")
    ap.add_argument("--keep-picks", action="store_true",
                    help="rebuild without clearing the drafted log (same layout only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the tiers and exit, without touching Sheets")
    ap.add_argument("--push-script", action="store_true",
                    help="upload draft_board.gs into the sheet's Apps Script project")
    ap.add_argument("--script-id", default=None,
                    help="Apps Script project id; saved to .env as TIERS_SCRIPT_ID")
    ap.add_argument("--setup-auth", action="store_true",
                    help=f"one-time Google OAuth flow; sign in as {SHEETS_ACCOUNT}")
    ap.add_argument("--client-secret", default=DEFAULT_CLIENT_SECRET,
                    help="path to the OAuth client_secret.json for --setup-auth")
    args = ap.parse_args()

    if args.setup_auth:
        return setup_google_auth(args.client_secret, ENV_PATH)

    for path in ENV_CANDIDATES:
        load_env_file(path)

    if args.script_id:
        save_env_value(ENV_PATH, "TIERS_SCRIPT_ID", args.script_id)
        os.environ["TIERS_SCRIPT_ID"] = args.script_id
        print(f"saved TIERS_SCRIPT_ID to {ENV_PATH}", file=sys.stderr)

    if args.push_script:
        script_id = os.environ.get("TIERS_SCRIPT_ID")
        if not script_id:
            print("No script id. Open the sheet, Extensions > Apps Script, then\n"
                  "Project Settings and copy the Script ID; pass it once with\n"
                  "--script-id <id>.", file=sys.stderr)
            return 1
        # Pushes the file as it stands rather than rewriting it first. The script
        # now carries the board's geometry, and the only run that knows the
        # geometry is the run that built the board -- so the copy on disk is the
        # one that matches the live sheet, and regenerating it here from a guessed
        # set of positions could only make it disagree.
        path = os.path.join(SCRIPT_DIR, "draft_board.gs")
        if not os.path.exists(path):
            print(f"{path} does not exist yet. Build the sheet once first: a "
                  f"normal run writes it.", file=sys.stderr)
            return 1
        return push_apps_script(script_id, path)

    # The default receptions file is a convenience, not a requirement: only
    # complain about one that was asked for by name and is not there.
    lineup = {}
    for part in args.starters.split(","):
        if not part.strip():
            continue
        position, _, count = part.partition(":")
        try:
            lineup[position.strip().upper()] = float(count)
        except ValueError:
            print(f"bad --starters entry: {part!r} (want e.g. RB:2)", file=sys.stderr)
            return 1
    # Before anything reads it: the panel, the caps and the need gate all take the
    # lineup from the module, not from an argument.
    set_lineup(lineup, args.flex, args.bench)

    receptions = None
    if args.receptions and os.path.exists(args.receptions):
        receptions = read_receptions(args.receptions)
    elif args.receptions and args.receptions != DEFAULT_RECEPTIONS_CSV:
        print(f"no such receptions file: {args.receptions}", file=sys.stderr)
        return 1
    by_position = read_players(
        args.csv, games=args.games, limit=args.limit, receptions=receptions,
        te_premium=args.te_premium,
        te_points_per_reception=args.te_points_per_reception,
        te_slots_per_ppg=args.te_slots_per_ppg, teams=args.teams,
        te_shift_model=args.te_shift_model, te_starters=args.te_starters,
        lineup=lineup, flex=args.flex)

    # Read the board a second time with the premium switched off, purely to get the
    # standard-scoring numbers. Both sets go onto the sheet so the toggle can pick
    # between them without a rebuild; the parse is cheap and deriving one from the
    # other in-sheet would mean duplicating the replacement maths in a formula.
    #
    # Every position is carried, not just the tight ends: PPG only ever moves at TE,
    # but rank is a queue and ADP is a draft, and the premium shuffles everyone's
    # place in both.
    # Keyed on (position, name) because bare names are not unique across a board --
    # a running back sharing a tight end's name would otherwise pick up his switch.
    standard, standard_board = {}, {}
    if args.te_premium:
        standard_board = read_players(args.csv, games=args.games, limit=args.limit,
                                      receptions=receptions, te_premium=0.0,
                                      teams=args.teams)
        standard = {(position, p.name): (p.ppg, p.adp, p.rank)
                    for position, players in standard_board.items()
                    for p in players}
    if not by_position:
        print(f"no players parsed from {args.csv}", file=sys.stderr)
        return 1

    # Everything was parsed, so every position held its rank slots while the
    # premium re-slotted the board and while the rank gaps were measured. Only now
    # do the ones that do not earn a block drop out.
    wanted = [p.strip().upper() for p in args.positions.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in by_position]
    if unknown:
        print(f"no players at: {', '.join(unknown)}", file=sys.stderr)
    dropped = sorted(p for p in by_position if p not in wanted)
    positions = ([p for p in POSITIONS if p in wanted]
                 + [p for p in sorted(by_position) if p in wanted and p not in POSITIONS])

    def cut(players):
        return assign_tiers(players, p=args.break_p, window=args.gap_window,
                            min_gap=args.min_gap, max_gap=args.max_gap,
                            basis=args.tier_on, max_spread=args.tier_max_spread,
                            split_drop=args.tier_split_drop)

    tiered, all_names = {}, []
    total = 0
    for position in positions:
        players = by_position.get(position)
        if not players:
            continue
        rows, k = cut(players)
        tiered[position] = rows
        all_names.extend(row.name for row in rows)
        total += len(rows)
        sizes = [sum(1 for row in rows if row.tier == t) for t in range(1, k + 1)]
        widest = max((max(r.ppg for r in rows if r.tier == t)
                      - min(r.ppg for r in rows if r.tier == t))
                     for t in range(1, k + 1))
        print(f"{position:>4}: {len(rows):>3} players, {k} tiers, "
              f"widest {widest:.2f} PPG, sizes {sizes}", file=sys.stderr)

    # The same cut on the standard-scoring board, so the premium toggle moves the
    # tier bands with the numbers instead of leaving last format's bands behind.
    standard_tiers = {}
    if standard_board:
        for position in positions:
            players = standard_board.get(position)
            if players:
                rows, _ = cut(players)
                standard_tiers.update(
                    ((position, row.name), row.tier) for row in rows)
        moved = sum(1 for (position, name), tier in standard_tiers.items()
                    if any(r.name == name and r.tier != tier
                           for r in tiered.get(position, ())))
        print(f"standard scoring: {moved} player(s) change tier when the premium "
              f"is switched off", file=sys.stderr)
        for position, backwards in tier_striping(tiered, standard_tiers):
            print(f"  warning: {position} standard tiers run backwards at "
                  f"{backwards} row(s); its colour bands will stripe with the "
                  f"premium off", file=sys.stderr)
    if dropped:
        held = sum(len(by_position[p]) for p in dropped)
        print(f"off the board: {', '.join(dropped)} "
              f"({held} players, still holding their rank slots)", file=sys.stderr)
    print(f"total: {total} players across {len(tiered)} positions, "
          f"tiers cut on {args.tier_on}", file=sys.stderr)

    # Depths for the VOR baselines, off the same premium-adjusted board the sheet
    # shows, so the replacement each column prices against is the one this league
    # would actually be left with.
    depths = starter_depths(by_position, args.teams, lineup, args.flex)
    if args.te_starters:
        depths[TE_POSITION] = args.te_starters
    # And the same depths at standard scoring, so VOR can follow the premium
    # toggle. Without this the sheet keeps pricing tight ends against TE18 with
    # the premium switched off, which is a replacement level that only exists
    # because of the premium -- the toggle would change the numerator and leave
    # the denominator behind.
    standard_depths = starter_depths(
        {position: [p._replace(ppg=round(p.ppg - p.bonus, 2)) for p in players]
         for position, players in by_position.items()},
        args.teams, lineup, args.flex)

    generated_at = datetime.date.today().isoformat()
    grid, blocks, total_rows, width, drafted_col, backup_col = build_sheet_payload(
        tiered, positions, generated_at, teams=args.teams, pick=args.pick,
        te_premium=args.te_premium, standard=standard, depths=depths,
        standard_depths=standard_depths,
        # Positions with no block still reach the drafted dropdown, so every pick
        # in the room is loggable and the roster panel's round parity holds.
        off_board=[(position, p.name) for position in dropped
                   for p in by_position[position]],
        standard_tiers=standard_tiers, tier_on=args.tier_on)
    gs_path = write_apps_script(
        os.path.join(SCRIPT_DIR, "draft_board.gs"), blocks, drafted_col, backup_col)

    if args.dry_run:
        for position in positions:
            rows = tiered.get(position)
            if not rows:
                continue
            print(f"\n=== {position} ===")
            current = None
            for row in rows:
                if row.tier != current:
                    print(f"-- Tier {row.tier} --")
                    current = row.tier
                print(f"   {row.name:<26} {row.team:<4} #{row.rank:<4} "
                      f"{row.ppg:>6.2f}  ADP {row.adp:>5.1f}")
        return 0

    print("\nwriting to Google Sheets...", file=sys.stderr)
    url = write_sheet(grid, blocks, total_rows, width, drafted_col,
                      args.sheet_url or os.environ.get("TIERS_SHEET_URL"),
                      args.sheet_name, keep_picks=args.keep_picks)
    if url:
        print(f"\ndone: {url}", file=sys.stderr)
        print(f"\nThe checkboxes need the Apps Script installed once:\n"
              f"  Extensions > Apps Script, paste {gs_path}, save, reload the sheet.\n"
              f"  (Once only -- it falls back to reading the board's layout at "
              f"runtime, so a rebuild that moves columns still works. It works the "
              f"slow way, though: the geometry is baked into this file, and a paste "
              f"after a move that changed it puts the fast path back.)",
              file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
