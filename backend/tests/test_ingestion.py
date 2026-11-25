import asyncio
from datetime import date, datetime, timezone

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import (
    CompetitionRecord,
    FixtureRecord,
    MatchResultRecord,
    SeasonRecord,
    TeamRecord,
    utc_now,
)
from app.models import (
    INITIAL_COMPETITIONS,
    Competition,
    Match,
    ResponseSet,
    Score,
    Season,
    Team,
)
from app.services.ingestion import FootballDataIngestionService


@pytest.fixture
def db_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'football.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


def provider_fixtures():
    competition = Competition(
        id=2021,
        name="Premier League",
        code="PL",
        type="LEAGUE",
    )
    season = Season(
        id=2285,
        startDate=date(2025, 8, 1),
        endDate=date(2026, 5, 31),
        currentMatchday=1,
    )
    home = Team(id=57, name="Arsenal", shortName="Arsenal", tla="ARS")
    away = Team(id=61, name="Chelsea", shortName="Chelsea", tla="CHE")
    match = Match(
        id=555001,
        utcDate=datetime(2025, 8, 17, 15, 30, tzinfo=timezone.utc),
        status="FINISHED",
        matchday=1,
        homeTeam=home,
        awayTeam=away,
        score=Score(winner="HOME_TEAM", fullTime={"home": 2, "away": 1}),
        competition=competition,
        season=season,
    )
    return competition, season, home, away, match


class FakeFootballDataClient:
    def __init__(self):
        self.competition, self.season, self.home, self.away, self.match = provider_fixtures()
        self.match_filter_calls = []
        self.team_seasons = []

    async def get_competitions(self):
        return ResponseSet(competitions=[self.competition])

    async def get_teams(self, competition_id: str, *, season: int | None = None):
        assert competition_id == "PL"
        self.team_seasons.append(season)
        return ResponseSet(teams=[self.home, self.away], season=self.season)

    async def get_competition_matches(self, competition_id: str, **kwargs):
        assert competition_id == "PL"
        self.match_filter_calls.append(kwargs)
        return ResponseSet(
            competition=self.competition,
            season=self.season,
            matches=[self.match],
        )


def test_database_models_preserve_provider_ids_and_relationships(db_factory):
    competition, season, home, away, match = provider_fixtures()
    with db_factory.begin() as session:
        comp = CompetitionRecord(
            provider_id=competition.id,
            code=competition.code,
            name=competition.name,
            source="football-data.org",
            retrieved_at=utc_now(),
        )
        session.add(comp)
        session.flush()
        season_row = SeasonRecord(
            provider_id=season.id,
            competition_id=comp.id,
            start_date=season.startDate,
            end_date=season.endDate,
            source="football-data.org",
            retrieved_at=utc_now(),
        )
        home_row = TeamRecord(
            provider_id=home.id,
            name=home.name,
            source="football-data.org",
            retrieved_at=utc_now(),
        )
        away_row = TeamRecord(
            provider_id=away.id,
            name=away.name,
            source="football-data.org",
            retrieved_at=utc_now(),
        )
        session.add_all([season_row, home_row, away_row])
        session.flush()
        fixture = FixtureRecord(
            provider_id=match.id,
            competition_id=comp.id,
            season_id=season_row.id,
            kickoff_at=match.utcDate,
            status=match.status,
            home_team_id=home_row.id,
            away_team_id=away_row.id,
            source="football-data.org",
            retrieved_at=utc_now(),
        )
        session.add(fixture)
        session.flush()
        session.add(
            MatchResultRecord(
                fixture_id=fixture.id,
                home_score=2,
                away_score=1,
                source="football-data.org",
                retrieved_at=utc_now(),
            )
        )

    with db_factory() as session:
        stored = session.scalar(
            select(FixtureRecord).where(FixtureRecord.provider_id == match.id)
        )
        assert stored is not None
        assert stored.competition.provider_id == competition.id
        assert stored.season.provider_id == season.id
        assert stored.home_team.provider_id == home.id
        assert stored.away_team.provider_id == away.id
        assert stored.result.home_score == 2
        assert stored.result.away_score == 1


def test_provider_id_unique_constraint_prevents_duplicate_competitions(db_factory):
    with db_factory.begin() as session:
        session.add(
            CompetitionRecord(
                provider_id=2021,
                name="Premier League",
                source="football-data.org",
                retrieved_at=utc_now(),
            )
        )
    with pytest.raises(IntegrityError):
        with db_factory.begin() as session:
            session.add(
                CompetitionRecord(
                    provider_id=2021,
                    name="Duplicate",
                    source="football-data.org",
                    retrieved_at=utc_now(),
                )
            )


def test_reingesting_provider_response_updates_without_duplicates(db_factory):
    async def run():
        client = FakeFootballDataClient()
        service = FootballDataIngestionService(client, db_factory)
        date_from = date(2025, 8, 1)
        date_to = date(2026, 6, 1)
        first = await service.sync_competition(
            "PL", season=2025, date_from=date_from, date_to=date_to
        )
        second = await service.sync_competition("PL", season=2025)
        assert first.fixtures == second.fixtures == 1
        assert first.results == second.results == 1
        assert client.team_seasons == [2025, 2025]
        assert client.match_filter_calls[0]["date_from"] == date_from
        assert client.match_filter_calls[0]["date_to"] == date_to
        assert client.match_filter_calls[0]["season"] == 2025

    asyncio.run(run())
    with db_factory() as session:
        assert session.scalar(select(func.count()).select_from(CompetitionRecord)) == 1
        assert session.scalar(select(func.count()).select_from(SeasonRecord)) == 1
        assert session.scalar(select(func.count()).select_from(TeamRecord)) == 2
        assert session.scalar(select(func.count()).select_from(FixtureRecord)) == 1
        assert session.scalar(select(func.count()).select_from(MatchResultRecord)) == 1


def test_historical_season_discovery_and_repeat_import_are_idempotent(db_factory):
    class HistoricalClient(FakeFootballDataClient):
        async def get_competition(self, competition_id: str):
            assert competition_id == "PL"
            return Competition(
                id=2021,
                name="Premier League",
                code="PL",
                seasons=[
                    Season(id=2285, startDate=date(2025, 8, 1), endDate=date(2026, 5, 31))
                ],
            )

    async def run():
        client = HistoricalClient()
        service = FootballDataIngestionService(client, db_factory)
        first = await service.sync_all_available_seasons(
            "PL", date_from=date(2025, 8, 15), date_to=date(2025, 8, 23)
        )
        second = await service.sync_all_available_seasons(
            "PL", date_from=date(2025, 8, 15), date_to=date(2025, 8, 23)
        )
        assert first.fixtures == second.fixtures == 1
        assert first.records_inserted > 0
        assert second.records_inserted == 0
        assert second.records_updated > 0
        assert client.team_seasons == [2025, 2025]
        assert client.match_filter_calls[0]["date_from"] == date(2025, 8, 15)

    asyncio.run(run())
    with db_factory() as session:
        assert session.scalar(select(func.count()).select_from(CompetitionRecord)) == 1
        assert session.scalar(select(func.count()).select_from(SeasonRecord)) == 1
        assert session.scalar(select(func.count()).select_from(TeamRecord)) == 2
        assert session.scalar(select(func.count()).select_from(FixtureRecord)) == 1
        assert session.scalar(select(func.count()).select_from(MatchResultRecord)) == 1


def test_live_fixture_does_not_create_missing_result(db_factory):
    async def run():
        client = FakeFootballDataClient()
        client.match.status = "IN_PLAY"
        client.match.score = None
        service = FootballDataIngestionService(client, db_factory)
        summary = await service.sync_competition("PL")
        assert summary.fixtures == 1
        assert summary.results == 0

    asyncio.run(run())
    with db_factory() as session:
        assert session.scalar(select(func.count()).select_from(FixtureRecord)) == 1
        assert session.scalar(select(func.count()).select_from(MatchResultRecord)) == 0


def test_initial_league_sync_persists_all_six_competitions_and_seasons(db_factory):
    class SixLeagueClient:
        competitions = [
            Competition(
                id=2020 + index,
                code=config.code.value,
                name=config.name,
                currentSeason=Season(
                    id=3000 + index,
                    startDate=date(2025, 8, 1),
                    endDate=date(2026, 5, 31),
                ),
            )
            for index, config in enumerate(INITIAL_COMPETITIONS)
        ]

        async def get_competitions(self):
            return ResponseSet(competitions=self.competitions)

        async def get_teams(self, competition_id: str, *, season: int | None = None):
            return ResponseSet(season=next(
                item.currentSeason for item in self.competitions if item.code == competition_id
            ))

        async def get_competition_matches(self, competition_id: str, **kwargs):
            competition = next(
                item for item in self.competitions if item.code == competition_id
            )
            return ResponseSet(competition=competition, matches=[])

    async def run():
        service = FootballDataIngestionService(SixLeagueClient(), db_factory)
        summary = await service.sync_initial_leagues(season=2025)
        assert summary.competitions == 6
        assert summary.seasons == 6
        assert summary.fixtures == 0

    asyncio.run(run())
    with db_factory() as session:
        assert session.scalar(select(func.count()).select_from(CompetitionRecord)) == 6
        assert session.scalar(select(func.count()).select_from(SeasonRecord)) == 6


def test_alembic_migration_matches_models_and_can_be_reverted(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    config = Config("alembic.ini")
    config.attributes["database_url"] = database_url
    engine = create_engine(database_url)
    try:
        command.upgrade(config, "head")
        tables = set(inspect(engine).get_table_names())
        assert {
            "competitions",
            "seasons",
            "teams",
            "fixtures",
            "match_results",
            "alembic_version",
        } <= tables
        with engine.connect() as connection:
            differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        assert differences == []

        command.downgrade(config, "base")
        remaining_tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
        assert remaining_tables == set()
    finally:
        engine.dispose()
