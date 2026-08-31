"""League validation, and that the report actually renders every table it promises."""

import os

import pytest

from draftsim.config import Budget, League, METRICS, SimConfig


# ---- League.validate ------------------------------------------------------

def test_the_default_league_is_the_board_plus_a_kicker_and_a_defense():
    league = League()
    assert league.mandatory == 10        # QB RB RB WR WR TE FLEX FLEX K DST
    assert league.bench == 6
    assert league.picks == 192
    assert league.playoff_weeks == 3


def test_too_few_rounds_to_field_a_lineup_is_rejected():
    """Silently producing zero-scoring starting slots would be far worse than failing."""
    with pytest.raises(ValueError, match="cannot fill"):
        League(rounds=8)


def test_a_cap_below_a_starting_requirement_is_rejected():
    with pytest.raises(ValueError, match="cap"):
        League(caps=(2, 1, 7, 3, 1, 1))


def test_a_bracket_that_does_not_fit_the_playoff_weeks_is_rejected():
    with pytest.raises(ValueError, match="playoff weeks"):
        League(regular_weeks=16, total_weeks=17, playoff_teams=6)


def test_more_playoff_teams_than_teams_is_rejected():
    with pytest.raises(ValueError, match="more playoff teams"):
        League(teams=4, playoff_teams=6)


def test_a_starters_tuple_of_the_wrong_length_is_rejected():
    with pytest.raises(ValueError, match="starters must have"):
        League(starters=(1, 2, 2, 1))


# ---- SimConfig ------------------------------------------------------------

def test_slots_default_to_every_slot():
    cfg = SimConfig()
    assert cfg.slots == tuple(range(1, 13))


def test_a_slot_outside_the_league_is_rejected():
    with pytest.raises(ValueError, match="slots outside"):
        SimConfig(slots=(0, 13))


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="metric must be"):
        SimConfig(metric="vibes")
    for metric in METRICS:
        SimConfig(metric=metric)


def test_an_unknown_adp_model_is_rejected():
    with pytest.raises(ValueError, match="adp_model"):
        SimConfig(adp_model="yahoo")


def test_more_finalists_than_candidates_is_rejected():
    with pytest.raises(ValueError, match="finalists"):
        Budget(candidates=4, finalists=8)


def test_injuries_are_on_and_the_switch_reaches_the_rates():
    """On, now that the waiver wire exists to respond to them.

    They were off for one run because modelling attrition without modelling the
    response to it hands bench depth a value it does not have. That reason is gone.
    """
    from draftsim.config import INJURY_MISS_ANY_PROB, NO_INJURIES
    assert SimConfig().injuries is True
    assert SimConfig().miss_any_prob == INJURY_MISS_ANY_PROB
    assert SimConfig(injuries=False).miss_any_prob == NO_INJURIES


def test_the_injury_rates_are_expressed_as_expected_games_missed():
    """The old rates under-injured by about half, and the reason was the parameterisation.

    P(any) = 0.55 with a three-week mean block gives a back 1.65 expected missed games
    where reality is nearer three. Stating the expectation and backing out P(any) makes
    the number checkable against a season's snap counts.
    """
    from draftsim.config import (EXPECTED_GAMES_MISSED, INJURY_MISS_ANY_PROB,
                                 MISS_MEAN_WEEKS, POSITIONS, RB, DST)
    assert EXPECTED_GAMES_MISSED[RB] >= 3.0
    assert EXPECTED_GAMES_MISSED[DST] == 0.0
    for i, expected in enumerate(EXPECTED_GAMES_MISSED):
        assert INJURY_MISS_ANY_PROB[i] == pytest.approx(
            min(1.0, expected / MISS_MEAN_WEEKS))


def test_projection_error_is_on_and_the_switch_reaches_it():
    """Without it the season is scored off the numbers the draft optimised against, so
    any bias in the source is risk-free profit and reaching for an outlier is free."""
    from draftsim.config import PROJECTION_SIGMA
    assert SimConfig().projection_error is True
    assert SimConfig().proj_sigma == PROJECTION_SIGMA
    assert set(SimConfig(projection_error=False).proj_sigma) == {0.0}


def test_replacement_defaults_to_the_waiver_wire():
    """Settled by experiment: a wire-priced drafter beats a board-priced room by
    2.0 +/- 0.4 points of playoff rate, paired across all twelve seats."""
    assert SimConfig().replacement_from_wire is True
    assert SimConfig().replacement == ()          # filled in by calibration


def test_the_weekly_cvs_are_the_raised_ones():
    """Pinned deliberately. These were raised after the first run was too confident,
    and a silent revert to the tamer set would quietly narrow every interval."""
    from draftsim.config import WEEKLY_CV, POSITIONS
    by_pos = dict(zip(POSITIONS, WEEKLY_CV))
    assert by_pos["QB"] == 0.35
    assert by_pos["RB"] == 0.65
    assert by_pos["WR"] == 0.75
    assert by_pos["TE"] == 0.80


def test_an_unknown_room_is_rejected():
    with pytest.raises(ValueError, match="room must be"):
        SimConfig(room="auction")


def test_a_negative_perception_sigma_is_rejected():
    with pytest.raises(ValueError, match="perception_sigma"):
        SimConfig(perception_sigma=-0.1)


def test_a_plan_reaching_into_the_kicker_rounds_is_rejected():
    """The plan space stops before the reserved rounds by construction; letting it
    overlap would mean a plan naming a position that cannot legally be drafted."""
    with pytest.raises(ValueError, match="K/DST rounds"):
        SimConfig(plan_rounds=15)


# ---- report ---------------------------------------------------------------

@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    from draftsim import report, search
    from draftsim.pool import load_pool

    pool = load_pool()
    cfg = SimConfig(league=League(),
                    budget=Budget(stage1=100, stage2=30, stage2b=20, stage3=60,
                                  candidates=9, mutate_top=1, mutate_passes=1,
                                  finalists=4, min_stage1_samples=1),
                    slots=(1, 12), jobs=1, baseline=48)
    baseline = search.run_baseline(pool, cfg, jobs=1)
    stage1 = search.run_stage1(pool, cfg, jobs=1)
    stage2 = search.run_stage2(pool, cfg, stage1, jobs=1)
    final = search.run_stage3(pool, cfg, stage2, jobs=1)

    lines = []
    report.print_header(pool, cfg, out=lines.append)
    report.print_baseline(cfg, baseline, out=lines.append)
    report.print_summary(cfg, final, baseline, stage1, out=lines.append)
    report.print_marginals(cfg, stage1, out=lines.append)
    report.print_slot_detail(cfg, pool, final, baseline, stage1, top=4,
                             out=lines.append)

    outdir = str(tmp_path_factory.mktemp("sim_out"))
    csvs = report.write_csvs(outdir, cfg, pool, stage1, stage2, final, baseline)
    md = report.write_markdown(outdir, cfg, pool, stage1, final, baseline, elapsed=61.0)
    return "\n".join(lines), csvs, md, cfg


def test_the_console_report_names_every_section(rendered):
    text, _, _, _ = rendered
    for heading in ("BPA BASELINE", "RECOMMENDED PLAN BY DRAFT SLOT", "ROUND 1",
                    "SLOT 1", "SLOT 12"):
        assert heading in text, heading


def test_the_console_report_shows_the_edge_not_only_the_rate(rendered):
    """The edge column is the one that stops the table being misread, so its absence is
    a bug rather than a formatting quibble."""
    text, _, _, _ = rendered
    assert "edge" in text
    assert "BPAbase" in text


def test_the_console_report_declares_ties_rather_than_a_bare_winner(rendered):
    """The point of stage 3. A table that names one best plan per slot when several are
    indistinguishable is presenting noise as a decision."""
    text, _, _, _ = rendered
    assert "tied" in text
    assert "indistinguishable from the leader" in text
    assert "recommended" in text


def test_the_console_header_states_the_model_it_ran(rendered):
    """Absolute rates are not comparable across configurations, so a table without its
    configuration attached is a trap."""
    text, _, _, _ = rendered
    assert "room" in text
    assert "injuries    on" in text
    assert "proj error  on" in text
    assert "weekly CV" in text


def test_every_csv_has_a_header_and_at_least_one_row(rendered):
    _, csvs, _, _ = rendered
    import csv as csvmod
    for path in csvs:
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            rows = list(csvmod.reader(fh))
        assert len(rows) >= 2, path
        assert all(rows[0]), path


def test_the_csvs_are_the_ones_the_cli_advertises(rendered):
    _, csvs, _, _ = rendered
    names = {os.path.basename(p) for p in csvs}
    assert names == {"baseline.csv", "stage1_marginals.csv", "stage1_shapes.csv",
                     "stage2_screen.csv", "finalists.csv", "anchors.csv"}


def test_the_markdown_carries_the_caveats(rendered):
    """The numbers are seductive and the model is a model. A reader who takes a 1.4
    point difference as a fact about football has been misled by the report."""
    _, _, md, _ = rendered
    text = open(md, encoding="utf-8").read()
    for phrase in ("BPA baseline", "Recommended plan by slot",
                   "Round-by-round marginals", "How to read this", "Read the edge",
                   "shape*, not a script", "Projected points are taken as truth",
                   "Injuries are off", "There is no waiver wire",
                   "Respect the tied column", "How the room was modelled"):
        assert phrase in text, phrase


def test_the_markdown_reports_the_calibration_mean(rendered):
    _, _, md, cfg = rendered
    text = open(md, encoding="utf-8").read()
    assert "50.0%" in text, "the baseline mean must be stated, it is the sanity check"
