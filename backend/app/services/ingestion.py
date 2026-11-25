import logging
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

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

logger = logging.getLogger(__name__)
SOURCE = "football-data.org"


class FootballDataProvider(Protocol):
    async def get_competitions(self) -> ResponseSet: ...

    async def get_competition_matches(self, competition_id: str, **kwargs) -> ResponseSet: ...

    async def get_competition(self, competition_id: str) -> Competition: ...

    async def get_teams(self, competition_id: str, *, season: int | None = None) -> ResponseSet: ...


class IngestionError(Exception):
    """Provider data could not be safely normalized or synchronized."""


@dataclass(frozen=True)
class IngestionSummary:
    competitions: int = 0
    seasons: int = 0
    teams: int = 0
    fixtures: int = 0
    results: int = 0
    records_fetched: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    errors: int = 0


class FootballDataIngestionService:
    def __init__(
        self,
        client: FootballDataProvider,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.client = client
        self.session_factory = session_factory

    async def sync_initial_leagues(
        self,
        *,
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> IngestionSummary:
        self._validate_dates(date_from, date_to)
        response = await self.client.get_competitions()
        configured_codes = {item.code.value for item in INITIAL_COMPETITIONS}
        competitions = [
            competition
            for competition in response.competitions
            if competition.code in configured_codes
        ]
        available_codes = {competition.code for competition in competitions}
        missing_codes = sorted(configured_codes - available_codes)
        if missing_codes:
            raise IngestionError(
                "football-data.org did not return configured competitions: "
                + ", ".join(missing_codes)
                + ". Check the account plan and competition access."
            )

        with self.session_factory.begin() as session:
            for competition in competitions:
                self._upsert_competition(session, competition)
            session.flush()

        total = IngestionSummary()
        for competition in competitions:
            item = await self._sync_competition(
                competition,
                season=season,
                date_from=date_from,
                date_to=date_to,
                status=status,
            )
            total = self._add_summaries(total, item)
        logger.info(
            "Initial league sync complete: competitions=%d seasons=%d teams=%d fixtures=%d results=%d",
            total.competitions,
            total.seasons,
            total.teams,
            total.fixtures,
            total.results,
        )
        return total

    async def sync_competition(
        self,
        competition_code: str,
        *,
        season: int | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> IngestionSummary:
        self._validate_dates(date_from, date_to)

        response = await self.client.get_competitions()
        competition = next(
            (item for item in response.competitions if item.code == competition_code),
            None,
        )
        if competition is None:
            raise IngestionError(
                f"Competition {competition_code} is not available to this account."
            )
        return await self._sync_competition(
            competition,
            season=season,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )

    async def sync_all_available_seasons(
        self,
        competition_code: str,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> IngestionSummary:
        """Import all seasons advertised by the provider for one competition."""
        self._validate_dates(date_from, date_to)
        get_competition = getattr(self.client, "get_competition", None)
        if get_competition is None:
            raise IngestionError("Provider client cannot discover available seasons.")
        try:
            detail = await get_competition(competition_code)
        except Exception:
            logger.exception("Season discovery failed for competition=%s", competition_code)
            raise
        years = sorted({season.startDate.year for season in detail.seasons if season.startDate})
        if not years and detail.currentSeason and detail.currentSeason.startDate:
            years = [detail.currentSeason.startDate.year]
        if not years:
            raise IngestionError(
                f"No provider seasons with start dates are available for {competition_code}."
            )
        if date_from and date_to:
            years = [
                year for year in years
                if year <= date_to.year and year + 1 >= date_from.year
            ]
        logger.info(
            "Discovered historical seasons competition=%s count=%d years=%s",
            competition_code, len(years), years,
        )
        total = IngestionSummary()
        for year in years:
            item = await self.sync_competition(
                competition_code,
                season=year,
                date_from=date_from,
                date_to=date_to,
                status=status,
            )
            total = self._add_summaries(total, item)
        return total

    @staticmethod
    def _add_summaries(left: IngestionSummary, right: IngestionSummary) -> IngestionSummary:
        return IngestionSummary(**{
            field: getattr(left, field) + getattr(right, field)
            for field in IngestionSummary.__dataclass_fields__
        })

    @staticmethod
    def _validate_dates(date_from: date | None, date_to: date | None) -> None:
        if (date_from is None) != (date_to is None):
            raise ValueError("date_from and date_to must be provided together")
        if date_from and date_to and date_from > date_to:
            raise ValueError("date_from must be before or equal to date_to")

    async def _sync_competition(
        self,
        competition: Competition,
        *,
        season: int | None,
        date_from: date | None,
        date_to: date | None,
        status: str | None,
    ) -> IngestionSummary:
        code = competition.code
        if not code:
            raise IngestionError(f"Competition {competition.id} has no provider code.")

        logger.info(
            "Fetching football-data.org records competition=%s season=%s date_from=%s date_to=%s",
            code,
            season,
            date_from,
            date_to,
        )
        try:
            teams_response = await self.client.get_teams(code, season=season)
            matches_response = await self.client.get_competition_matches(
                code,
                season=season,
                date_from=date_from,
                date_to=date_to,
                status=status,
            )
        except Exception:
            logger.exception(
                "Provider fetch failed competition=%s season=%s date_from=%s date_to=%s",
                code, season, date_from, date_to,
            )
            raise

        team_payloads = {team.id: team for team in teams_response.teams}
        for match in matches_response.matches:
            team_payloads.setdefault(match.homeTeam.id, match.homeTeam)
            team_payloads.setdefault(match.awayTeam.id, match.awayTeam)

        seasons_seen: set[int | tuple[date | None, date | None]] = set()
        results_count = 0
        inserted_count = 0
        updated_count = 0
        classified: set[tuple[str, object]] = set()

        def track(record: object, kind: str, provider_key: object) -> None:
            nonlocal inserted_count, updated_count
            marker = (kind, provider_key)
            if marker in classified:
                return
            classified.add(marker)
            if record in session.new:
                inserted_count += 1
            else:
                updated_count += 1

        try:
            with self.session_factory.begin() as session:
                competition_record = self._upsert_competition(session, competition)
                track(competition_record, "competition", competition.id)
                for team in team_payloads.values():
                    team_record = self._upsert_team(session, team)
                    track(team_record, "team", team.id)
                session.flush()

                for season_payload in (teams_response.season, matches_response.season):
                    if season_payload is not None:
                        season_record = self._upsert_season(session, competition_record, season_payload)
                        track(season_record, "season", season_payload.id or (season_payload.startDate, season_payload.endDate))
                        seasons_seen.add(
                            season_payload.id
                            or (season_payload.startDate, season_payload.endDate)
                        )

                for match in matches_response.matches:
                    match_competition = match.competition or matches_response.competition or competition
                    if match_competition.id != competition.id:
                        raise IngestionError(
                            f"Match {match.id} belongs to unexpected provider competition {match_competition.id}."
                        )
                    season_payload = match.season or matches_response.season or teams_response.season
                    if season_payload is None:
                        raise IngestionError(
                            f"Match {match.id} has no provider season; refusing to invent one."
                        )
                    season_record = self._upsert_season(
                        session, competition_record, season_payload
                    )
                    track(season_record, "season", season_payload.id or (season_payload.startDate, season_payload.endDate))
                    season_key = season_payload.id or (
                        season_payload.startDate,
                        season_payload.endDate,
                    )
                    seasons_seen.add(season_key)
                    fixture = self._upsert_fixture(
                        session,
                        match,
                        competition_record,
                        season_record,
                        team_payloads[match.homeTeam.id],
                        team_payloads[match.awayTeam.id],
                    )
                    track(fixture, "fixture", match.id)
                    session.flush()
                    if self._upsert_result(session, fixture, match.score):
                        results_count += 1
                        result_record = session.get(MatchResultRecord, fixture.id)
                        track(result_record, "result", match.id)
        except SQLAlchemyError:
            logger.exception(
                "Database upsert failed competition=%s season=%s date_from=%s date_to=%s",
                code, season, date_from, date_to,
            )
            raise

        fetched = 1 + len(seasons_seen) + len(team_payloads) + len(matches_response.matches) + results_count
        summary = IngestionSummary(
            competitions=1,
            seasons=len(seasons_seen),
            teams=len(team_payloads),
            fixtures=len(matches_response.matches),
            results=results_count,
            records_fetched=fetched,
            records_inserted=inserted_count,
            records_updated=updated_count,
        )
        logger.info(
            "Competition sync complete code=%s season=%s date_from=%s date_to=%s fetched=%d inserted=%d updated=%d skipped=%d errors=%d",
            code,
            season,
            date_from,
            date_to,
            summary.records_fetched,
            summary.records_inserted,
            summary.records_updated,
            summary.records_skipped,
            summary.errors,
        )
        return summary

    @staticmethod
    def _upsert_competition(session: Session, item: Competition) -> CompetitionRecord:
        record = session.scalar(
            select(CompetitionRecord).where(CompetitionRecord.provider_id == item.id)
        )
        if record is None:
            record = CompetitionRecord(provider_id=item.id, name=item.name)
            session.add(record)
        record.code = item.code
        record.name = item.name
        record.type = item.type
        record.emblem = item.emblem
        if item.area:
            record.area_provider_id = item.area.id
            record.area_name = item.area.name
            record.area_code = item.area.code
        record.source = SOURCE
        record.retrieved_at = utc_now()
        return record

    @staticmethod
    def _upsert_season(
        session: Session, competition: CompetitionRecord, item: Season
    ) -> SeasonRecord:
        if item.id is not None:
            record = session.scalar(
                select(SeasonRecord).where(SeasonRecord.provider_id == item.id)
            )
        elif item.startDate is not None:
            record = session.scalar(
                select(SeasonRecord).where(
                    SeasonRecord.competition_id == competition.id,
                    SeasonRecord.start_date == item.startDate,
                )
            )
        else:
            raise IngestionError("Provider season has no stable ID or start date.")
        if record is None:
            record = SeasonRecord(competition_id=competition.id)
            session.add(record)
        elif record.competition_id != competition.id:
            raise IngestionError(
                f"Provider season {item.id} is already linked to another competition."
            )
        record.provider_id = item.id
        record.competition_id = competition.id
        record.start_date = item.startDate
        record.end_date = item.endDate
        record.current_matchday = item.currentMatchday
        record.source = SOURCE
        record.retrieved_at = utc_now()
        return record

    @staticmethod
    def _upsert_team(session: Session, item: Team) -> TeamRecord:
        record = session.scalar(select(TeamRecord).where(TeamRecord.provider_id == item.id))
        if record is None:
            record = TeamRecord(provider_id=item.id, name=item.name)
            session.add(record)
        record.name = item.name
        record.short_name = item.shortName
        record.tla = item.tla
        record.crest = item.crest
        record.venue = item.venue
        if item.area:
            record.area_provider_id = item.area.id
            record.area_name = item.area.name
        record.source = SOURCE
        record.retrieved_at = utc_now()
        return record

    @staticmethod
    def _upsert_fixture(
        session: Session,
        item: Match,
        competition: CompetitionRecord,
        season: SeasonRecord,
        home_team: Team,
        away_team: Team,
    ) -> FixtureRecord:
        home_team_id = session.scalar(
            select(TeamRecord.id).where(TeamRecord.provider_id == home_team.id)
        )
        away_team_id = session.scalar(
            select(TeamRecord.id).where(TeamRecord.provider_id == away_team.id)
        )
        if home_team_id is None or away_team_id is None:
            raise IngestionError(f"Match {item.id} references a team that was not stored.")
        record = session.scalar(select(FixtureRecord).where(FixtureRecord.provider_id == item.id))
        if record is None:
            record = FixtureRecord(provider_id=item.id)
            session.add(record)
        record.competition_id = competition.id
        record.season_id = season.id
        record.kickoff_at = item.utcDate
        record.status = item.status
        record.matchday = item.matchday
        record.stage = item.stage
        record.group = item.group
        record.home_team_id = home_team_id
        record.away_team_id = away_team_id
        record.source = SOURCE
        record.retrieved_at = utc_now()
        return record

    @staticmethod
    def _upsert_result(
        session: Session, fixture: FixtureRecord, score: Score | None
    ) -> bool:
        if score is None or score.fullTime is None:
            return False
        home_score = score.fullTime.get("home")
        away_score = score.fullTime.get("away")
        if home_score is None or away_score is None:
            return False
        record = session.get(MatchResultRecord, fixture.id)
        if record is None:
            record = MatchResultRecord(fixture_id=fixture.id)
            session.add(record)
        record.home_score = home_score
        record.away_score = away_score
        record.winner = score.winner
        record.source = SOURCE
        record.retrieved_at = utc_now()
        return True
