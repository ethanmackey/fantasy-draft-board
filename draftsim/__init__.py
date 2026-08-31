"""Monte Carlo draft simulator for the 2026 tight-end-premium league.

The draft board in ``draft_tiers.py`` prices players. This package answers the
question the board cannot: from the slot you actually drew, what shape of roster
should you be trying to build? It drafts against ESPN-ADP opponents thousands of
times per slot, plays a season out of every roster, and reports which draft plans
win.

Entry point is ``simulate_drafts.py`` at the repository root. See
``docs/superpowers/specs/2026-08-19-draft-simulator-design.md``.
"""
