from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import aliased

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.db.models import (  # noqa: E402
    CompetitionRecord,
    FixtureRecord,
    MatchResultRecord,
    SeasonRecord,
    TeamRecord,
)
from app.db.session import get_session_factory  # noqa: E402
from ml.features.feature_engineering_v2 import (  # noqa: E402
    DATASET_COLUMNS,
    ELIGIBLE_STATUS,
    FEATURE_COLUMNS,
    MATCH_INFORMATION_COLUMNS,
    SCHEMA_VERSION,
    SourceMatch,
    TARGET_COLUMNS,
    assert_comparator_ids_match,
    build_training_rows_v2,
)

logger = logging.getLogger(__name__)
DEFAULT_CUTOFF = date(2026, 10, 31)
DEFAULT_DATASET = REPOSITORY_ROOT / "ml" / "data" / "processed" / "v2" / "training_dataset.csv"
LEGACY_DATASET = REPOSITORY_ROOT / "ml" / "data" / "processed" / "training_dataset.csv"
V1_TEST_PREDICTIONS = REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v1" / "test_predictions.csv"

CONFIGURATION: dict[str, Any] = {
    "feature_schema_version": SCHEMA_VERSION,
    "eligible_fixture_statuses": [ELIGIBLE_STATUS],
    "same_date_policy": "All fixtures on one UTC date are featurized before any result from that date updates state.",
    "competition_baseline": {
        "home": "mean home-team goals in prior same-competition FINISHED fixtures",
        "away": "mean away-team goals in prior same-competition FINISHED fixtures",
        "minimum_observations_per_venue": 20,
        "update_policy": "Expanding point-in-time mean, applied after each UTC date batch.",
        "historical_normalization": "Use baseline available before each historical appearance; never backfill later means.",
    },
    "team_strength": {
        "decay_half_life_days": 365,
        "neutral_prior_strength": 1.0,
        "prior_pseudo_matches": 5.0,
        "attack": "(5 + sum(w * goals_for / venue_baseline)) / (5 + sum(w))",
        "defence": "(5 + sum(w)) / (5 + sum(w * goals_against / opponent_venue_baseline))",
        "scope": "Same competition, all prior venues, FINISHED fixtures only.",
    },
    "recent_form": {
        "window": 5,
        "scope": "Prior same-competition FINISHED appearances across venues and seasons.",
        "partial_window": "Average over available observations; missing only when none exist.",
    },
    "elo": {
        "initial_rating": 1500,
        "scale": 400,
        "k_factor": 10,
        "home_advantage": 60,
        "offseason_regression": 0.75,
        "margin_of_victory_multiplier": 1,
        "scope": "Separate competition pools; FINISHED fixtures only.",
    },
    "history_exposure_features": ["home_history_matches", "away_history_matches"],
    "venue_residual_features_enabled": False,
    "match_information_columns": MATCH_INFORMATION_COLUMNS,
    "feature_columns": FEATURE_COLUMNS,
    "target_columns": TARGET_COLUMNS,
    "missing_values": {
        "dataset": "Preserve undefined historical features as empty CSV cells/null values.",
        "model_preprocessing": "Fit numeric medians, missing indicators, scaling and categorical encoders on each training fold only.",
    },
    "excluded_statuses": ["AWARDED", "IN_PLAY", "CANCELLED", "POSTPONED", "SCHEDULED", "TIMED"],
    "cutoff_date": DEFAULT_CUTOFF.isoformat(),
}


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_record(match: SourceMatch) -> dict[str, Any]:
    return {
        "provider_match_id": match.provider_match_id,
        "kickoff_utc": match.kickoff_utc.isoformat(),
        "status": match.status,
        "competition": match.competition,
        "season_provider_id": match.season_provider_id,
        "season_start_date": match.season_start_date.isoformat() if match.season_start_date else None,
        "home_team_provider_id": match.home_team_provider_id,
        "home_team_name": match.home_team_name,
        "away_team_provider_id": match.away_team_provider_id,
        "away_team_name": match.away_team_name,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "winner": match.winner,
    }


def load_source_matches(session_factory=None) -> list[SourceMatch]:
    home_team = aliased(TeamRecord)
    away_team = aliased(TeamRecord)
    statement = (
        select(
            FixtureRecord.provider_id,
            FixtureRecord.kickoff_at,
            FixtureRecord.status,
            CompetitionRecord.code,
            SeasonRecord.provider_id,
            SeasonRecord.start_date,
            home_team.provider_id,
            home_team.name,
            away_team.provider_id,
            away_team.name,
            MatchResultRecord.home_score,
            MatchResultRecord.away_score,
            MatchResultRecord.winner,
        )
        .join(MatchResultRecord, MatchResultRecord.fixture_id == FixtureRecord.id)
        .join(CompetitionRecord, CompetitionRecord.id == FixtureRecord.competition_id)
        .join(SeasonRecord, SeasonRecord.id == FixtureRecord.season_id)
        .join(home_team, home_team.id == FixtureRecord.home_team_id)
        .join(away_team, away_team.id == FixtureRecord.away_team_id)
        .order_by(FixtureRecord.kickoff_at, FixtureRecord.provider_id)
    )
    factory = session_factory or get_session_factory()
    with factory() as session:
        records = session.execute(statement).all()
    return [
        SourceMatch(
            provider_match_id=row[0],
            kickoff_at=row[1],
            status=row[2],
            competition=row[3],
            season_provider_id=row[4],
            season_start_date=row[5],
            home_team_provider_id=row[6],
            home_team_name=row[7],
            away_team_provider_id=row[8],
            away_team_name=row[9],
            home_goals=row[10],
            away_goals=row[11],
            winner=row[12],
        )
        for row in records
    ]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no CSV header: {path}")
        return list(reader)


def write_dataset(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty v2 dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=DATASET_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_manifest(
    source_matches: list[SourceMatch],
    rows: list[dict[str, Any]],
    expanded_history_rows: list[dict[str, str]],
    saved_v1_test_ids: list[int],
    dataset_path: Path,
    cutoff_date: date = DEFAULT_CUTOFF,
) -> dict[str, Any]:
    # The explicit invariant guards against accidentally relaxing the status filter.
    if any(int(row["provider_match_id"]) == 542704 for row in rows):
        raise ValueError("Awarded fixture 542704 must not appear in the v2 dataset")
    training_ids, test_ids = assert_comparator_ids_match(rows, expanded_history_rows, cutoff_date)
    if test_ids != saved_v1_test_ids:
        raise ValueError("V2 test IDs do not match the preserved Poisson v1 test fixture IDs")

    all_ids = [int(row["provider_match_id"]) for row in rows]
    status_counts = Counter(match.status for match in source_matches)
    excluded_ids: dict[str, list[int]] = defaultdict(list)
    eligible_ids = set(all_ids)
    for match in source_matches:
        if match.provider_match_id not in eligible_ids:
            excluded_ids[match.status].append(match.provider_match_id)
    source_records = sorted((_source_record(match) for match in source_matches), key=lambda row: row["provider_match_id"])
    training_manifest = {
        "cutoff_date": cutoff_date.isoformat(),
        "training_fixture_ids": training_ids,
        "test_fixture_ids": test_ids,
    }
    configuration = {**CONFIGURATION, "cutoff_date": cutoff_date.isoformat()}
    config_hash = _canonical_hash(configuration)
    try:
        relative_dataset_path = str(dataset_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    except ValueError:
        relative_dataset_path = str(dataset_path)
    return {
        "feature_schema_version": SCHEMA_VERSION,
        "configuration": configuration,
        "configuration_sha256": config_hash,
        "source_data_sha256": _canonical_hash(source_records),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_path": relative_dataset_path,
        "row_count": len(rows),
        "column_count": len(DATASET_COLUMNS),
        "match_information_columns": MATCH_INFORMATION_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "missing_value_rates_by_feature": {
            column: sum(row.get(column) is None for row in rows) / len(rows)
            for column in FEATURE_COLUMNS
        },
        "ordered_fixture_ids": all_ids,
        "split": training_manifest,
        "comparator_manifests": {
            "expanded_history_poisson": training_manifest,
            "redesigned_poisson_v2": training_manifest,
        },
        "status_counts_in_source": dict(sorted(status_counts.items())),
        "excluded_fixture_ids_by_status": {
            status: sorted(ids) for status, ids in sorted(excluded_ids.items())
        },
        "excluded_fixture_count_by_status": {
            status: len(ids) for status, ids in sorted(excluded_ids.items())
        },
    }


def run(
    output_path: Path = DEFAULT_DATASET,
    *,
    session_factory=None,
    legacy_dataset_path: Path = LEGACY_DATASET,
    v1_predictions_path: Path = V1_TEST_PREDICTIONS,
    cutoff_date: date = DEFAULT_CUTOFF,
) -> dict[str, Any]:
    source_matches = load_source_matches(session_factory)
    rows = build_training_rows_v2(source_matches)
    legacy_rows = load_csv(legacy_dataset_path)
    saved_predictions = load_csv(v1_predictions_path)
    saved_v1_test_ids = [int(row["provider_match_id"]) for row in saved_predictions]

    output_path = output_path.resolve()
    write_dataset(rows, output_path)
    manifest = build_manifest(
        source_matches,
        rows,
        legacy_rows,
        saved_v1_test_ids,
        output_path,
        cutoff_date,
    )
    manifest_path = output_path.parent / "manifest.json"
    _write_json(manifest_path, manifest)
    logger.info(
        "Generated v2 dataset rows=%d columns=%d train=%d test=%d source_sha256=%s dataset_sha256=%s",
        manifest["row_count"],
        manifest["column_count"],
        len(manifest["split"]["training_fixture_ids"]),
        len(manifest["split"]["test_fixture_ids"]),
        manifest["source_data_sha256"],
        manifest["dataset_sha256"],
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the versioned v2 feature dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    run(output_path=args.output, cutoff_date=args.cutoff_date)


if __name__ == "__main__":
    main()
