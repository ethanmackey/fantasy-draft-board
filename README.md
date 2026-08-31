# Fantasy Draft Board

A tiered fantasy football draft board that lives in Google Sheets, plus a draft
simulator that tells you what shape of roster to aim for from the slot you drew.

Built for a 12-team tight end premium league.

## What's here

| File | What it does |
| --- | --- |
| `draft_tiers.py` | Reads a rankings export + projections CSV, prices every player, cuts each position into tiers where the drop to the next player is unusually large, and writes the board to a Google Sheet. |
| `draft_board.gs` | Apps Script pasted into the sheet. A checkbox beside a player marks them drafted; they drop out of their position column and everyone below slides up. Ticking A1 resets the board. |
| `simulate_drafts.py` | Drafts against eleven simulated opponents a few hundred thousand times to grade positional plans for every draft slot. |
| `draftsim/` | The simulator's model: player pool, rosters, bots, season scoring, plan search, reporting. |
| `tests/` | pytest suite over the simulator. |
| `docs/superpowers/specs/` | Design notes for the simulator, including its limits. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill it in
python draft_tiers.py --setup-auth   # one-time Google sign-in, writes the refresh token
```

`.env` holds the Google OAuth client and refresh token plus the target sheet and
Apps Script ids. It is gitignored — never commit it.

## Use

```bash
python draft_tiers.py --dry-run     # print the tiers, touch nothing
python draft_tiers.py               # write / rewrite the sheet

python simulate_drafts.py --quick   # smoke test, ~30 seconds
python simulate_drafts.py           # full run, all 12 slots
python simulate_drafts.py --plan RB-WR-WR-TE-RB-WR-QB-RB   # grade one plan
```

Simulator output lands in `sim_out/` (gitignored, regenerated on every run).

```bash
pytest
```

## Data

The `Draft-rankings-export-*.csv` and `projections*.csv` files are dated snapshots
of the inputs; the scripts default to the most recent pair.
