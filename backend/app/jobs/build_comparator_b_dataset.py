from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import date, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import aliased

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.db.models import (  # noqa: E402
    CompetitionRecord, FixtureRecord, MatchResultRecord, SeasonRecord, TeamRecord,
)
from app.db.session import get_session_factory  # noqa: E402
from ml.features.comparator_b import (  # noqa: E402
    COMPARATOR_B_COLUMNS, CUTOFF_DATE, EXCLUDED_FIXTURE_IDS, FEATURE_DEFINITIONS,
    build_comparator_b_rows,
)
from ml.features.feature_engineering import CompletedMatch  # noqa: E402

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ml" / "data" / "processed" / "comparator_b" / "v1" / "training_dataset.csv"
V2_DATASET = REPOSITORY_ROOT / "ml" / "data" / "processed" / "v2" / "training_dataset.csv"
V2_MANIFEST = V2_DATASET.with_name("manifest.json")
V1_TEST_PREDICTIONS = REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v1" / "test_predictions.csv"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_eligible_matches(session_factory=None) -> list[CompletedMatch]:
    home = aliased(TeamRecord)
    away = aliased(TeamRecord)
    statement = (
        select(
            FixtureRecord.provider_id, FixtureRecord.kickoff_at, CompetitionRecord.code,
            SeasonRecord.provider_id, SeasonRecord.start_date,
            home.provider_id, home.name, away.provider_id, away.name,
            MatchResultRecord.home_score, MatchResultRecord.away_score,
        )
        .join(MatchResultRecord, MatchResultRecord.fixture_id == FixtureRecord.id)
        .join(CompetitionRecord, CompetitionRecord.id == FixtureRecord.competition_id)
        .join(SeasonRecord, SeasonRecord.id == FixtureRecord.season_id)
        .join(home, home.id == FixtureRecord.home_team_id)
        .join(away, away.id == FixtureRecord.away_team_id)
        .where(FixtureRecord.status == "FINISHED")
        .where(FixtureRecord.provider_id.not_in(EXCLUDED_FIXTURE_IDS))
        .order_by(FixtureRecord.kickoff_at, FixtureRecord.provider_id)
    )
    factory = session_factory or get_session_factory()
    with factory() as session:
        records = session.execute(statement).all()
    return [CompletedMatch(
        provider_match_id=row[0], kickoff_at=row[1], competition=row[2],
        season_provider_id=row[3], season_start_date=row[4],
        home_team_provider_id=row[5], home_team_name=row[6],
        away_team_provider_id=row[7], away_team_name=row[8],
        home_goals=row[9], away_goals=row[10],
    ) for row in records]


def _source_record(match: CompletedMatch) -> dict[str, Any]:
    kickoff = match.kickoff_utc.isoformat()
    return {
        "provider_match_id": match.provider_match_id,
        "kickoff_utc": kickoff,
        "status": "FINISHED",
        "competition": match.competition,
        "season_provider_id": match.season_provider_id,
        "season_start_date": match.season_start_date.isoformat() if match.season_start_date else None,
        "home_team_provider_id": match.home_team_provider_id,
        "home_team_name": match.home_team_name,
        "away_team_provider_id": match.away_team_provider_id,
        "away_team_name": match.away_team_name,
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
    }


def _write_dataset(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=COMPARATOR_B_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_manifest(
    matches: list[CompletedMatch], rows: list[dict[str, Any]], output_path: Path,
    v2_manifest: dict[str, Any], v2_dataset_path: Path, v1_prediction_path: Path,
) -> dict[str, Any]:
    all_ids = [int(row["provider_match_id"]) for row in rows]
    split = v2_manifest["split"]
    train_ids = [int(value) for value in split["training_fixture_ids"]]
    test_ids = [int(value) for value in split["test_fixture_ids"]]
    v2_ordered_ids = [int(value) for value in v2_manifest["ordered_fixture_ids"]]
    if all_ids != v2_ordered_ids:
        raise ValueError("Comparator B ordered fixture IDs differ from the approved v2 dataset")
    train_actual = [int(row["provider_match_id"]) for row in rows if str(row["match_date"]) < CUTOFF_DATE.isoformat()]
    test_actual = [int(row["provider_match_id"]) for row in rows if str(row["match_date"]) >= CUTOFF_DATE.isoformat()]
    if train_actual != train_ids or test_actual != test_ids:
        raise ValueError("Comparator B split fixture IDs differ from approved v2 manifests")
    if any(match.provider_match_id in EXCLUDED_FIXTURE_IDS for match in matches) or any(
        fixture_id in EXCLUDED_FIXTURE_IDS for fixture_id in all_ids
    ):
        raise ValueError("Awarded fixture 542704 must be absent from B source, state and targets")
    v1_test_ids = []
    with v1_prediction_path.open(newline="", encoding="utf-8") as f:
        v1_test_ids = [int(row["provider_match_id"]) for row in csv.DictReader(f)]
    if test_ids != v1_test_ids:
        raise ValueError("Comparator B test IDs differ from protected Poisson v1 IDs")

    configuration = {
        "schema_version": "comparator_b_v1",
        "source_statuses": ["FINISHED"],
        "excluded_fixture_ids": sorted(EXCLUDED_FIXTURE_IDS),
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "utc_date_batching": "Emit all same-UTC-date features before updating team history.",
        "history_scope": "Original v1 behavior: provider team ID history across all competitions.",
        "features": list(FEATURE_DEFINITIONS.items()),
        "feature_columns": [column for column in COMPARATOR_B_COLUMNS if column not in v2_manifest["match_information_columns"] and column not in v2_manifest["target_columns"]],
        "columns": list(COMPARATOR_B_COLUMNS),
    }
    source_records = sorted((_source_record(match) for match in matches), key=lambda row: row["provider_match_id"])
    missing_rates = {
        feature: sum(row.get(feature) is None for row in rows) / len(rows)
        for feature in configuration["feature_columns"]
    }
    try:
        output_relative = str(output_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    except ValueError:
        output_relative = str(output_path)
    try:
        source_relative = str(v2_dataset_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
    except ValueError:
        source_relative = str(v2_dataset_path)
    return {
        "schema_version": "comparator_b_v1",
        "dataset_path": output_relative,
        "source_dataset_path": source_relative,
        "source_dataset_sha256": hashlib.sha256(v2_dataset_path.read_bytes()).hexdigest(),
        "source_data_sha256": _canonical_hash(source_records),
        "dataset_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "configuration": configuration,
        "configuration_sha256": _canonical_hash(configuration),
        "row_count": len(rows),
        "column_count": len(COMPARATOR_B_COLUMNS),
        "feature_columns": list(configuration["feature_columns"]),
        "feature_definitions": FEATURE_DEFINITIONS,
        "missing_value_rates_by_feature": missing_rates,
        "ordered_fixture_ids": all_ids,
        "ordered_fixture_ids_sha256": _canonical_hash(all_ids),
        "split": {
            "cutoff_date": CUTOFF_DATE.isoformat(),
            "training_count": len(train_ids),
            "training_fixture_ids": train_ids,
            "training_fixture_ids_sha256": _canonical_hash(train_ids),
            "test_count": len(test_ids),
            "test_fixture_ids": test_ids,
            "test_fixture_ids_sha256": _canonical_hash(test_ids),
        },
        "excluded_awarded_fixture_ids": sorted(EXCLUDED_FIXTURE_IDS),
        "reproducibility": {
            "source_order": "sorted by provider fixture ID for canonical source hash",
            "feature_builder": "ml.features.feature_engineering.build_training_rows with the approved awarded fixture removed",
            "python_runtime": sys.version.split()[0],
            "target_timezone": "UTC",
        },
    }


def run(
    output_path: Path = DEFAULT_OUTPUT, *, session_factory=None,
    v2_dataset_path: Path = V2_DATASET, v2_manifest_path: Path = V2_MANIFEST,
    v1_prediction_path: Path = V1_TEST_PREDICTIONS,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    matches = load_eligible_matches(session_factory)
    rows = build_comparator_b_rows(matches)
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    _write_dataset(rows, output_path)
    manifest = build_manifest(matches, rows, output_path, v2_manifest, v2_dataset_path, v1_prediction_path)
    _write_json(output_path.with_name("manifest.json"), manifest)
    logger.info("Built comparator B dataset rows=%d features=%d dataset_sha256=%s", len(rows), len(manifest["feature_columns"]), manifest["dataset_sha256"])
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean expanded-history Poisson comparator B dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

