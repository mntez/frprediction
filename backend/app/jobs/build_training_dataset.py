from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

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
from ml.features.feature_engineering import (  # noqa: E402
    COMPLETED_STATUSES,
    FEATURE_COLUMNS,
    CompletedMatch,
    build_training_rows,
    TARGET_COLUMNS,
)

logger = logging.getLogger(__name__)


def load_completed_matches(session_factory=None) -> list[CompletedMatch]:
    home_team = aliased(TeamRecord)
    away_team = aliased(TeamRecord)
    statement = (
        select(
            FixtureRecord.provider_id,
            FixtureRecord.kickoff_at,
            CompetitionRecord.code,
            SeasonRecord.provider_id,
            SeasonRecord.start_date,
            home_team.provider_id,
            home_team.name,
            away_team.provider_id,
            away_team.name,
            MatchResultRecord.home_score,
            MatchResultRecord.away_score,
        )
        .join(MatchResultRecord, MatchResultRecord.fixture_id == FixtureRecord.id)
        .join(CompetitionRecord, CompetitionRecord.id == FixtureRecord.competition_id)
        .join(SeasonRecord, SeasonRecord.id == FixtureRecord.season_id)
        .join(home_team, home_team.id == FixtureRecord.home_team_id)
        .join(away_team, away_team.id == FixtureRecord.away_team_id)
        .where(FixtureRecord.status.in_(COMPLETED_STATUSES))
        .order_by(FixtureRecord.kickoff_at, FixtureRecord.provider_id)
    )
    factory = session_factory or get_session_factory()
    with factory() as session:
        rows = session.execute(statement).all()

    return [
        CompletedMatch(
            provider_match_id=row[0],
            kickoff_at=row[1],
            competition=row[2],
            season_provider_id=row[3],
            season_start_date=row[4],
            home_team_provider_id=row[5],
            home_team_name=row[6],
            away_team_provider_id=row[7],
            away_team_name=row[8],
            home_goals=row[9],
            away_goals=row[10],
        )
        for row in rows
    ]


def write_dataset(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FEATURE_COLUMNS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-aware processed dataset from completed PostgreSQL matches"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "ml" / "data" / "processed" / "training_dataset.csv",
        help="CSV output path (defaults to ml/data/processed/training_dataset.csv)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    matches = load_completed_matches()
    rows = build_training_rows(matches)
    output_path = parse_args().output.resolve()
    write_dataset(rows, output_path)
    logger.info(
        "Processed dataset created rows=%d features=%d output=%s",
        len(rows),
        len(FEATURE_COLUMNS) - len(TARGET_COLUMNS),
        output_path,
    )


if __name__ == "__main__":
    main()
