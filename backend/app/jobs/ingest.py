import argparse
import asyncio
import logging
from datetime import date

from app.db.session import get_session_factory
from app.services.football_data import FootballDataClient
from app.services.ingestion import FootballDataIngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import football-data.org matches into PostgreSQL")
    parser.add_argument("--competition", help="Competition code, for example PL")
    parser.add_argument("--season", type=int, help="Season start year, for example 2025")
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Discover and import every season exposed for the selected competition",
    )
    parser.add_argument("--date-from", type=date.fromisoformat)
    parser.add_argument("--date-to", type=date.fromisoformat)
    parser.add_argument("--status", help="Provider match status, for example FINISHED")
    args = parser.parse_args()
    if args.all_seasons and not args.competition:
        parser.error("--all-seasons requires --competition to keep the import scope explicit")
    if args.all_seasons and args.season is not None:
        parser.error("--all-seasons cannot be combined with --season")
    return args


async def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    async with FootballDataClient() as client:
        service = FootballDataIngestionService(client, get_session_factory())
        filters = {
            "season": args.season,
            "date_from": args.date_from,
            "date_to": args.date_to,
            "status": args.status,
        }
        if args.all_seasons:
            summary = await service.sync_all_available_seasons(
                args.competition,
                date_from=args.date_from,
                date_to=args.date_to,
                status=args.status,
            )
        elif args.competition:
            summary = await service.sync_competition(args.competition, **filters)
        else:
            summary = await service.sync_initial_leagues(**filters)
    logging.getLogger(__name__).info("Ingestion totals: %s", summary)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
