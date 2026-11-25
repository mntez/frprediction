import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.jobs.build_comparator_b_dataset import build_manifest
from ml.baselines.poisson import TEAM_FEATURES
from ml.features.comparator_b import COMPARATOR_B_COLUMNS, build_comparator_b_rows
from ml.features.feature_engineering import CompletedMatch
from ml.training.fold_preprocessing import (
    ROLLING_ORIGIN_FOLDS,
    FoldPreprocessor,
    comparator_b_preprocessor,
    prepare_rolling_fold,
    split_rolling_fold,
)
from ml.training.model_outputs import OUTPUTS, assert_unprotected_output_path, get_output_directory

ROOT = Path(__file__).resolve().parents[2]


def match(provider_id, day, *, home=1, away=None, hg=2, ag=1, year=2025):
    away = provider_id + 1000 if away is None else away
    return CompletedMatch(
        provider_match_id=provider_id,
        kickoff_at=datetime(year, 1, day, 15, tzinfo=timezone.utc),
        competition="PL", season_provider_id=2403,
        season_start_date=date(year, 7, 1),
        home_team_provider_id=home, home_team_name=f"Team {home}",
        away_team_provider_id=away, away_team_name=f"Team {away}",
        home_goals=hg, away_goals=ag,
    )


def test_comparator_b_has_exact_original_six_inputs_and_no_v2_features():
    assert TEAM_FEATURES == (
        "home_home_matches_played", "home_home_goals_scored_avg",
        "home_home_goals_conceded_avg", "away_away_matches_played",
        "away_away_goals_scored_avg", "away_away_goals_conceded_avg",
    )
    feature_columns = [column for column in COMPARATOR_B_COLUMNS if column in TEAM_FEATURES]
    assert tuple(feature_columns) == TEAM_FEATURES
    assert not set(TEAM_FEATURES) & {
        "home_attack_strength", "home_defense_strength", "away_attack_strength",
        "away_defense_strength", "elo_difference",
    }


def test_awarded_fixture_is_removed_from_all_b_history_and_targets():
    earlier = match(100, 1, home=8, away=9)
    awarded = match(542704, 2, home=77, away=10, hg=0, ag=0)
    later = match(102, 3, home=8, away=10, hg=1, ag=0)
    rows = build_comparator_b_rows([earlier, awarded, later])
    later_row = next(row for row in rows if row["provider_match_id"] == 102)
    assert 542704 not in [row["provider_match_id"] for row in rows]
    assert later_row["away_away_matches_played"] == 0
    assert later_row["away_away_goals_scored_avg"] is None
    assert later_row["away_away_goals_conceded_avg"] is None


def test_same_utc_date_results_do_not_enter_each_others_features():
    first = match(10, 5, home=1, away=2, hg=9, ag=0)
    second = match(11, 5, home=2, away=3, hg=0, ag=4)
    later = match(12, 6, home=1, away=3)
    rows = build_comparator_b_rows([first, second, later])
    same_day_second = next(row for row in rows if row["provider_match_id"] == 11)
    assert same_day_second["away_away_matches_played"] == 0
    later_row = next(row for row in rows if row["provider_match_id"] == 12)
    assert later_row["home_home_matches_played"] == 1


def test_future_result_cannot_change_an_earlier_comparator_row():
    target = match(30, 5, home=1, away=2, hg=1, ag=0)
    future = match(31, 6, home=1, away=3, hg=8, ag=0)
    isolated = next(row for row in build_comparator_b_rows([target]) if row["provider_match_id"] == 30)
    with_future = next(row for row in build_comparator_b_rows([future, target]) if row["provider_match_id"] == 30)
    feature_keys = set(TEAM_FEATURES)
    assert {key: isolated[key] for key in feature_keys} == {key: with_future[key] for key in feature_keys}


def test_comparator_b_build_is_deterministic_and_matches_v2_ordered_ids():
    matches = [match(4, 2), match(2, 1), match(3, 1)]
    assert build_comparator_b_rows(matches) == build_comparator_b_rows(list(reversed(matches)))

    manifest = json.loads((ROOT / "ml/data/processed/v2/manifest.json").read_text(encoding="utf-8"))
    b_manifest = json.loads((ROOT / "ml/data/processed/comparator_b/v1/manifest.json").read_text(encoding="utf-8"))
    ids = [int(row["provider_match_id"]) for row in csv.DictReader(
        (ROOT / "ml/data/processed/comparator_b/v1/training_dataset.csv").open(encoding="utf-8", newline="")
    )]
    assert ids == manifest["ordered_fixture_ids"]
    assert b_manifest["ordered_fixture_ids"] == ids
    assert b_manifest["split"]["training_fixture_ids"] == manifest["split"]["training_fixture_ids"]
    assert b_manifest["split"]["test_fixture_ids"] == manifest["split"]["test_fixture_ids"]
    assert manifest["comparator_manifests"]["expanded_history_poisson"] == manifest["comparator_manifests"]["redesigned_poisson_v2"]


def test_audit_identified_toulouse_fixtures_use_award_free_history():
    path = ROOT / "ml/data/processed/comparator_b/v1/training_dataset.csv"
    rows = {int(row["provider_match_id"]): row for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}
    expected_counts = {559710: 16, 559684: 17, 559673: 18, 559661: 19, 559645: 20, 559623: 21}
    assert {fixture_id: int(rows[fixture_id]["away_away_matches_played"]) for fixture_id in expected_counts} == expected_counts


def test_original_six_features_retain_v1_venue_definitions():
    rows = build_comparator_b_rows([
        match(20, 1, home=10, away=11, hg=3, ag=2),
        match(21, 2, home=12, away=20, hg=2, ag=1),
        match(22, 3, home=10, away=20, hg=1, ag=0),
    ])
    row = next(item for item in rows if item["provider_match_id"] == 22)
    assert row["home_home_matches_played"] == 1
    assert row["home_home_goals_scored_avg"] == 3.0
    assert row["home_home_goals_conceded_avg"] == 2.0
    assert row["away_away_matches_played"] == 1
    assert row["away_away_goals_scored_avg"] == 1.0
    assert row["away_away_goals_conceded_avg"] == 2.0


def test_four_folds_are_chronological_and_preserve_date_groups():
    rows = [
        {"provider_match_id": 1, "match_date": "2025-12-31"},
        {"provider_match_id": 2, "match_date": "2026-01-01"},
        {"provider_match_id": 3, "match_date": "2026-03-31"},
    ]
    train, validation = split_rolling_fold(rows, ROLLING_ORIGIN_FOLDS[0])
    assert [row["provider_match_id"] for row in train] == [1]
    assert [row["provider_match_id"] for row in validation] == [2, 3]
    assert len(ROLLING_ORIGIN_FOLDS) == 4


def test_preprocessor_fits_only_training_rows_and_validation_cannot_change_state():
    train = [
        {"x": 1.0, "category": "A"},
        {"x": None, "category": "A"},
        {"x": 3.0, "category": "B"},
    ]
    validation = [{"x": 10000.0, "category": "UNSEEN"}]
    transformer = FoldPreprocessor(("x",), ("category",)).fit(train)
    before = transformer.fitted_state()
    transformed = transformer.transform(validation)
    after = transformer.fitted_state()
    assert before == after
    assert before["numeric_medians"] == [2.0]
    assert before["categories"] == [["A", "B"]]
    assert transformed.shape[0] == 1
    assert transformer.get_feature_names_out().size == transformed.shape[1]


def test_fold_workflow_fits_train_and_only_transforms_validation():
    rows = [
        {"provider_match_id": 1, "match_date": "2025-12-31", "x": 2.0},
        {"provider_match_id": 2, "match_date": "2026-01-01", "x": 9000.0},
        {"provider_match_id": 3, "match_date": "2026-03-31", "x": 10000.0},
    ]
    prepared = prepare_rolling_fold(rows, ROLLING_ORIGIN_FOLDS[0], FoldPreprocessor(("x",)))
    assert [row["provider_match_id"] for row in prepared.training_rows] == [1]
    assert [row["provider_match_id"] for row in prepared.validation_rows] == [2, 3]
    assert prepared.preprocessor.fitted_state()["numeric_medians"] == [2.0]


def test_targets_and_identifiers_are_rejected_as_preprocessing_inputs():
    with pytest.raises(ValueError, match="Targets or identifiers"):
        FoldPreprocessor(("home_goals",))


def test_comparator_preprocessor_uses_exact_six_inputs():
    processor = comparator_b_preprocessor()
    assert processor.features == TEAM_FEATURES


def test_v1_artifact_paths_are_rejected_and_future_paths_are_isolated():
    with pytest.raises(ValueError, match="protected v1"):
        assert_unprotected_output_path(ROOT / "ml/models/poisson_baseline/v1/new.json")
    assert_unprotected_output_path(OUTPUTS["comparator_b"]["model"])
    assert_unprotected_output_path(OUTPUTS["poisson_v2"]["evaluation"])
    with pytest.raises(ValueError, match="protected v1"):
        assert_unprotected_output_path(ROOT / "ml/evaluation/xgboost/v1/new.json")
    assert get_output_directory("comparator_b", "evaluation") == OUTPUTS["comparator_b"]["evaluation"].resolve()


def test_protected_v1_and_approved_v2_artifacts_keep_their_audit_hashes():
    expected = {
        "ml/models/poisson_baseline/v1/model.json": "77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a",
        "ml/evaluation/poisson_baseline/v1/evaluation.json": "9965f5209a9630fc2bac23946e9cf4e85b017ad6cba65cc9bdd6be053d4ad41b",
        "ml/evaluation/poisson_baseline/v1/test_predictions.csv": "cf61138fbf45bba36dbcc3e9df8f73fd2b634584867b2fe2554766ce1afcea7e",
        "ml/data/processed/v2/training_dataset.csv": "7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5",
        "ml/data/processed/v2/manifest.json": "67daae2ea0361ccdc8ede1f1630b9cf0e518068dee0103f9fc040733b59317e6",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

