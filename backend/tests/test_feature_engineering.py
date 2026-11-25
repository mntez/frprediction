from datetime import date, datetime, timezone

from app.db.models import (
    CompetitionRecord,
    FixtureRecord,
    MatchResultRecord,
    SeasonRecord,
    TeamRecord,
    utc_now,
)
from app.jobs.build_training_dataset import load_completed_matches
from ml.features.feature_engineering import CompletedMatch, FEATURE_COLUMNS, TARGET_COLUMNS, build_training_rows


def match(
    provider_id: int,
    day: int,
    home_id: int = 1,
    away_id: int | None = None,
    home_goals: int = 1,
    away_goals: int = 0,
) -> CompletedMatch:
    away_id = away_id if away_id is not None else provider_id + 100
    return CompletedMatch(
        provider_match_id=provider_id,
        kickoff_at=datetime(2025, 1, day, 15, tzinfo=timezone.utc),
        competition="PL",
        season_provider_id=1,
        season_start_date=date(2024, 8, 1),
        home_team_provider_id=home_id,
        home_team_name=f"Team {home_id}",
        away_team_provider_id=away_id,
        away_team_name=f"Team {away_id}",
        home_goals=home_goals,
        away_goals=away_goals,
    )


def row_by_id(rows, provider_id):
    return next(row for row in rows if row["provider_match_id"] == provider_id)


def feature_values(row):
    return {key: value for key, value in row.items() if key not in TARGET_COLUMNS}


def test_match_cannot_use_its_own_result_as_a_feature():
    target = match(10, 5, home_goals=8, away_goals=7)
    row = build_training_rows([target])[0]

    assert row["home_previous_matches_played"] == 0
    assert row["home_previous_goals_scored"] == 0
    assert row["home_rolling_avg_goals_scored"] is None
    assert row["home_home_win_rate"] is None
    assert row["home_win"] == 1
    assert row["total_goals"] == 15
    assert "home_goals" not in feature_values(row)
    assert "away_goals" not in feature_values(row)


def test_future_match_cannot_change_an_earlier_rows_features():
    target = match(10, 5, home_goals=1, away_goals=0)
    future = match(11, 6, home_id=1, home_goals=20, away_goals=0)

    earlier_only = row_by_id(build_training_rows([target]), 10)
    with_future = row_by_id(build_training_rows([future, target]), 10)
    future_row = row_by_id(build_training_rows([target, future]), 11)

    assert feature_values(earlier_only) == feature_values(with_future)
    assert future_row["home_previous_goals_scored"] == 1


def test_rolling_statistics_include_only_prior_matches():
    history = [match(index, index, home_goals=index) for index in range(1, 7)]
    target = match(7, 7, home_goals=99)

    row = row_by_id(build_training_rows(history + [target]), 7)

    assert row["home_previous_matches_played"] == 6
    assert row["home_previous_goals_scored"] == sum(range(1, 7))
    assert row["home_rolling_avg_goals_scored"] == 4.0
    assert row["home_recent_form"] == "WWWWW"


def test_chronological_output_and_same_day_results_are_not_used():
    later = match(2, 4, home_id=1, home_goals=5)
    same_day = match(3, 3, home_id=1, home_goals=9)
    earlier = match(1, 2, home_id=1, home_goals=2)

    rows = build_training_rows([later, same_day, earlier])

    assert [row["provider_match_id"] for row in rows] == [1, 3, 2]
    assert row_by_id(rows, 3)["home_previous_goals_scored"] == 2
    assert row_by_id(rows, 2)["home_previous_goals_scored"] == 11


def test_output_columns_are_stable_and_targets_are_complete():
    row = build_training_rows([match(1, 2, home_goals=2, away_goals=1)])[0]
    assert list(row) == FEATURE_COLUMNS
    assert set(TARGET_COLUMNS) <= row.keys()
    assert {key: row[key] for key in TARGET_COLUMNS} == {
        "home_goals": 2, "away_goals": 1, "home_win": 1, "draw": 0, "away_win": 0,
        "total_goals": 3, "over_1_5": 1, "over_2_5": 1, "over_3_5": 0, "btts": 1,
    }


def test_processed_source_excludes_cancelled_results(db_factory):
    retrieved_at = utc_now()
    with db_factory.begin() as session:
        competition = CompetitionRecord(provider_id=2021, code="PL", name="Premier League", source="test", retrieved_at=retrieved_at)
        season = SeasonRecord(provider_id=1, competition=competition, start_date=date(2024, 8, 1), source="test", retrieved_at=retrieved_at)
        home = TeamRecord(provider_id=10, name="Home", source="test", retrieved_at=retrieved_at)
        away = TeamRecord(provider_id=20, name="Away", source="test", retrieved_at=retrieved_at)
        session.add_all([competition, season, home, away])
        session.flush()
        for match_id, status in ((101, "FINISHED"), (102, "AWARDED"), (103, "CANCELLED")):
            fixture = FixtureRecord(
                provider_id=match_id,
                competition_id=competition.id,
                season_id=season.id,
                kickoff_at=datetime(2025, 1, match_id - 100, 15, tzinfo=timezone.utc),
                status=status,
                home_team_id=home.id,
                away_team_id=away.id,
                source="test",
                retrieved_at=retrieved_at,
            )
            session.add(fixture)
            session.flush()
            session.add(MatchResultRecord(
                fixture_id=fixture.id,
                home_score=0,
                away_score=0,
                source="test",
                retrieved_at=retrieved_at,
            ))

    processed = load_completed_matches(db_factory)
    assert [item.provider_match_id for item in processed] == [101, 102]
