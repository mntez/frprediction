from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from ml.baselines.poisson import TEAM_FEATURES
from ml.features.feature_engineering import (
    COMPLETED_STATUSES,
    CompletedMatch as V1CompletedMatch,
    FEATURE_COLUMNS as V1_COLUMNS,
    build_training_rows as build_training_rows_v1,
)
from ml.features.feature_engineering_v2 import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    SourceMatch,
    _Appearance,
    _CompetitionGoals,
    _EloRating,
    _season_rating,
    _strengths,
    assert_comparator_ids_match,
    build_training_rows_v2,
    venue_residuals,
)
from ml.train_poisson_baseline import (
    DEFAULT_DATASET as V1_DATASET,
    EVALUATION_DIRECTORY as V1_EVALUATION_DIRECTORY,
    MODEL_DIRECTORY as V1_MODEL_DIRECTORY,
    VERSION as V1_VERSION,
)


def match(
    provider_id: int,
    day: int,
    *,
    home_id: int = 1,
    away_id: int | None = None,
    home_goals: int = 2,
    away_goals: int = 1,
    competition: str = "PL",
    season_year: int = 2025,
    status: str = "FINISHED",
) -> SourceMatch:
    if away_id is None:
        away_id = provider_id + 1000
    return SourceMatch(
        provider_match_id=provider_id,
        kickoff_at=datetime(2025, 1, day, 15, tzinfo=timezone.utc),
        status=status,
        competition=competition,
        season_provider_id=season_year * 10 + (1 if competition == "PL" else 2),
        season_start_date=date(season_year, 7, 1),
        home_team_provider_id=home_id,
        home_team_name=f"Team {home_id}",
        away_team_provider_id=away_id,
        away_team_name=f"Team {away_id}",
        home_goals=home_goals,
        away_goals=away_goals,
    )


def row_by_id(rows, provider_id):
    return next(row for row in rows if row["provider_match_id"] == provider_id)


def features(row):
    ignored = set(TARGET_COLUMNS) | {
        "provider_match_id", "competition", "season_provider_id", "season_start_year",
        "match_date", "home_team_provider_id", "home_team", "away_team_provider_id", "away_team",
    }
    return {key: value for key, value in row.items() if key not in ignored}


def test_team_strength_formulas_apply_neutral_pseudo_match_shrinkage():
    history = [
        _Appearance(date(2025, 1, 1), 2, 2, 3, "home", 2.0, 2.0),
        _Appearance(date(2025, 1, 1), 1, 1, 1, "away", 0.5, 0.5),
    ]

    attack, defence = _strengths(
        history, target_date=date(2025, 1, 2), baseline_ready=True
    )

    weight = 2 ** (-1 / 365)
    assert attack == pytest.approx((5 + weight * 2.5) / (5 + 2 * weight))
    assert defence == pytest.approx((5 + 2 * weight) / (5 + weight * 2.5))


def test_competition_baseline_requires_twenty_observations_per_venue():
    baseline = _CompetitionGoals()
    assert baseline.baselines() == (None, None)
    for index in range(19):
        baseline.add(match(index + 1, index + 1, home_goals=2, away_goals=1))
    assert baseline.baselines() == (None, None)
    baseline.add(match(20, 20, home_goals=4, away_goals=0))
    assert baseline.baselines() == pytest.approx((2.1, 0.95))


def test_competition_baselines_update_after_day_and_never_backfill():
    history = [match(index + 1, index + 1, home_goals=2, away_goals=1) for index in range(20)]
    history.extend([
        match(21, 21, home_id=1, away_id=2001, home_goals=4, away_goals=0),
        match(22, 22, home_id=1, away_id=2002),
    ])

    rows = build_training_rows_v2(history)

    assert row_by_id(rows, 20)["home_attack_strength"] is None
    # The baseline first becomes ready on day 21, after day 20's features were emitted.
    assert row_by_id(rows, 21)["home_attack_strength"] == pytest.approx(1.0)
    # Day 21's 4-goal performance is normalized by the then-current home baseline of 2.
    weight = 2 ** (-1 / 365)
    assert row_by_id(rows, 22)["home_attack_strength"] == pytest.approx((5 + weight * 2) / (5 + weight))


def test_recent_form_averages_available_matches_without_zero_filling():
    rows = build_training_rows_v2([
        match(1, 1, home_id=10, away_id=11, home_goals=2, away_goals=0),
        match(2, 2, home_id=10, away_id=12, home_goals=1, away_goals=1),
        match(3, 3, home_id=10, away_id=13),
    ])

    target = row_by_id(rows, 3)
    assert target["home_form_points_5"] == pytest.approx(2.0)
    assert target["home_goals_for_5"] == pytest.approx(1.5)
    assert target["home_goals_against_5"] == pytest.approx(0.5)
    assert target["home_history_matches"] == 2


def test_missing_history_is_explicit_and_elo_difference_starts_neutral():
    row = build_training_rows_v2([match(1, 1)])[0]

    assert row["home_attack_strength"] is None
    assert row["home_defense_strength"] is None
    assert row["home_form_points_5"] is None
    assert row["home_goals_for_5"] is None
    assert row["home_history_matches"] == 0
    assert row["away_history_matches"] == 0
    assert row["elo_difference"] == 0


def test_competition_boundary_keeps_team_history_and_elo_pools_separate():
    pl_history = [
        match(index + 1, index + 1, home_id=1, away_id=1000 + index, competition="PL")
        for index in range(20)
    ]
    pl_history.append(match(30, 21, home_id=1, away_id=2000, competition="PD"))

    target = row_by_id(build_training_rows_v2(pl_history), 30)
    assert target["home_history_matches"] == 0
    assert target["home_attack_strength"] is None
    assert target["elo_difference"] == 0


def test_venue_residuals_require_five_observations_in_the_requested_venue():
    history = [
        _Appearance(date(2025, 1, 1), 2, 1, 3, "home", 2.0, 0.5),
        _Appearance(date(2025, 1, 2), 2, 1, 3, "home", 2.0, 0.5),
        _Appearance(date(2025, 1, 3), 2, 1, 3, "home", 2.0, 0.5),
        _Appearance(date(2025, 1, 4), 2, 1, 3, "home", 2.0, 0.5),
        _Appearance(date(2025, 1, 5), 2, 1, 3, "home", 2.0, 0.5),
        _Appearance(date(2025, 1, 6), 4, 3, 0, "away", 4.0, 3.0),
    ]
    insufficient = venue_residuals(
        history[:4], venue="home", target_date=date(2025, 1, 7),
        overall_attack=1.0, overall_defence=1.0,
    )
    home_residual = venue_residuals(
        history, venue="home", target_date=date(2025, 1, 7),
        overall_attack=1.0, overall_defence=1.0,
    )

    assert insufficient == (None, None)
    assert home_residual[0] > 0
    assert home_residual[1] > 0


def test_elo_initialization_and_multi_season_regression():
    ratings = {}
    key = ("PL", 42)
    assert _season_rating(ratings, key, 2025) == 1500
    ratings[key].rating = 1700
    assert _season_rating(ratings, key, 2026) == pytest.approx(1650)
    assert _season_rating(ratings, key, 2028) == pytest.approx(1500 + 150 * (0.75**2))


def test_all_fixtures_on_same_utc_date_use_opening_day_state():
    local_east_africa = timezone(timedelta(hours=3))
    first_late_fixture = replace(
        match(2, 2, home_id=1, away_id=3, home_goals=10, away_goals=0),
        kickoff_at=datetime(2025, 1, 2, 22, 30, tzinfo=local_east_africa),
    )
    second_after_midnight_fixture = replace(
        match(3, 3, home_id=1, away_id=4, home_goals=0, away_goals=9),
        kickoff_at=datetime(2025, 1, 3, 0, 30, tzinfo=local_east_africa),
    )
    rows = build_training_rows_v2([
        match(1, 1, home_id=1, away_id=2),
        first_late_fixture,
        second_after_midnight_fixture,
    ])

    first = row_by_id(rows, 2)
    second = row_by_id(rows, 3)
    assert first["match_date"] == second["match_date"] == "2025-01-02"
    for key in (
        "home_attack_strength", "home_defense_strength", "home_form_points_5",
        "home_goals_for_5", "home_goals_against_5", "home_history_matches",
        "elo_difference",
    ):
        assert first[key] == second[key]


def test_own_result_and_future_matches_cannot_change_existing_features():
    target = match(10, 5, home_goals=1, away_goals=0)
    future = match(11, 6, home_id=1, away_id=40, home_goals=15, away_goals=0)
    changed_target = match(10, 5, home_goals=0, away_goals=12)

    base_row = row_by_id(build_training_rows_v2([target]), 10)
    with_future = row_by_id(build_training_rows_v2([target, future]), 10)
    with_changed_result = row_by_id(build_training_rows_v2([changed_target]), 10)

    assert features(base_row) == features(with_future) == features(with_changed_result)


def test_awarded_fixture_542704_is_absent_and_cannot_change_later_state():
    prior = match(1, 1, home_id=543, away_id=511, home_goals=2, away_goals=1)
    awarded = match(
        542704, 2, home_id=543, away_id=511, home_goals=0, away_goals=0,
        status="AWARDED",
    )
    later = match(3, 3, home_id=543, away_id=511, home_goals=1, away_goals=1)

    with_awarded = build_training_rows_v2([prior, awarded, later])
    without_awarded = build_training_rows_v2([prior, later])

    assert 542704 not in {row["provider_match_id"] for row in with_awarded}
    assert row_by_id(with_awarded, 3) == row_by_id(without_awarded, 3)


def test_output_is_deterministic_and_schema_excludes_v1_redundant_features():
    matches = [match(2, 2), match(1, 1)]
    first = build_training_rows_v2(matches)
    second = build_training_rows_v2(matches)

    assert first == second
    assert list(first[0]) == DATASET_COLUMNS
    assert len(FEATURE_COLUMNS) == 13
    assert not any(column.startswith("h2h_") for column in DATASET_COLUMNS)
    assert "home_recent_form" not in DATASET_COLUMNS
    assert "home_previous_wins" not in DATASET_COLUMNS
    assert "home_previous_goals_scored" not in DATASET_COLUMNS


def test_expanded_history_and_v2_comparator_splits_have_identical_ids():
    cutoff = date(2026, 10, 31)
    matches = [
        match(1, 1),
        SourceMatch(
            provider_match_id=2, kickoff_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
            status="FINISHED", competition="PL", season_provider_id=2025,
            season_start_date=date(2025, 7, 1), home_team_provider_id=1,
            home_team_name="Team 1", away_team_provider_id=2, away_team_name="Team 2",
            home_goals=2, away_goals=1,
        ),
        SourceMatch(
            provider_match_id=3, kickoff_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
            status="FINISHED", competition="PL", season_provider_id=2026,
            season_start_date=date(2026, 7, 1), home_team_provider_id=1,
            home_team_name="Team 1", away_team_provider_id=2, away_team_name="Team 2",
            home_goals=1, away_goals=1,
        ),
    ]
    v2_rows = build_training_rows_v2(matches)
    legacy_rows = [{"provider_match_id": str(row["provider_match_id"]), "match_date": row["match_date"]} for row in v2_rows]

    training_ids, test_ids = assert_comparator_ids_match(v2_rows, legacy_rows, cutoff)

    assert training_ids == [1, 2]
    assert test_ids == [3]


def test_legacy_v1_feature_and_artifact_contract_is_unchanged():
    legacy_row = build_training_rows_v1([
        V1CompletedMatch(
            provider_match_id=542704,
            kickoff_at=datetime(2026, 5, 17, 19, tzinfo=timezone.utc),
            competition="FL1",
            season_provider_id=2420,
            season_start_date=date(2025, 8, 17),
            home_team_provider_id=543,
            home_team_name="FC Nantes",
            away_team_provider_id=511,
            away_team_name="Toulouse FC",
            home_goals=0,
            away_goals=0,
        )
    ])[0]

    assert COMPLETED_STATUSES == frozenset({"FINISHED", "AWARDED"})
    assert set(legacy_row) == set(V1_COLUMNS)
    assert "h2h_previous_matches" in legacy_row
    assert "home_previous_wins" in legacy_row
    assert TEAM_FEATURES == (
        "home_home_matches_played", "home_home_goals_scored_avg",
        "home_home_goals_conceded_avg", "away_away_matches_played",
        "away_away_goals_scored_avg", "away_away_goals_conceded_avg",
    )
    assert V1_VERSION == "v1"
    assert V1_DATASET.name == "training_dataset.csv"
    assert V1_DATASET.parent.name == "processed"
    assert V1_MODEL_DIRECTORY.parts[-2:] == ("poisson_baseline", "v1")
    assert V1_EVALUATION_DIRECTORY.parts[-2:] == ("poisson_baseline", "v1")
