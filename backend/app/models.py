from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderModel(BaseModel):
    """Provider response model that tolerates fields added by football-data.org."""

    model_config = ConfigDict(extra="allow")


class CompetitionCode(StrEnum):
    PREMIER_LEAGUE = "PL"
    LA_LIGA = "PD"
    SERIE_A = "SA"
    BUNDESLIGA = "BL1"
    LIGUE_1 = "FL1"
    EREDIVISIE = "DED"


class ConfiguredCompetition(BaseModel):
    code: CompetitionCode
    name: str


INITIAL_COMPETITIONS = [
    ConfiguredCompetition(code=CompetitionCode.PREMIER_LEAGUE, name="Premier League"),
    ConfiguredCompetition(code=CompetitionCode.LA_LIGA, name="La Liga / Primera Division"),
    ConfiguredCompetition(code=CompetitionCode.SERIE_A, name="Serie A"),
    ConfiguredCompetition(code=CompetitionCode.BUNDESLIGA, name="Bundesliga"),
    ConfiguredCompetition(code=CompetitionCode.LIGUE_1, name="Ligue 1"),
    ConfiguredCompetition(code=CompetitionCode.EREDIVISIE, name="Eredivisie"),
]


class Area(ProviderModel):
    id: int | None = None
    name: str
    code: str | None = None
    flag: str | None = None


class Season(ProviderModel):
    id: int | None = None
    startDate: date | None = None
    endDate: date | None = None
    currentMatchday: int | None = None
    winner: dict[str, Any] | None = None


class Competition(ProviderModel):
    id: int
    name: str
    code: str | None = None
    type: str | None = None
    emblem: str | None = None
    area: Area | None = None
    currentSeason: Season | None = None
    seasons: list[Season] = Field(default_factory=list)
    numberOfAvailableSeasons: int | None = None
    lastUpdated: datetime | None = None


class Team(ProviderModel):
    id: int
    name: str
    shortName: str | None = None
    tla: str | None = None
    crest: str | None = None
    address: str | None = None
    website: str | None = None
    founded: int | None = None
    clubColors: str | None = None
    venue: str | None = None
    area: Area | None = None


class Score(ProviderModel):
    winner: str | None = None
    duration: str | None = None
    fullTime: dict[str, int | None] | None = None
    halfTime: dict[str, int | None] | None = None
    regularTime: dict[str, int | None] | None = None
    extraTime: dict[str, int | None] | None = None
    penalties: dict[str, int | None] | None = None


class Match(ProviderModel):
    id: int
    utcDate: datetime
    status: str
    matchday: int | None = None
    stage: str | None = None
    group: str | None = None
    lastUpdated: datetime | None = None
    homeTeam: Team
    awayTeam: Team
    score: Score | None = None
    competition: Competition | None = None
    season: Season | None = None


class Standing(ProviderModel):
    position: int | None = None
    team: Team
    playedGames: int | None = None
    won: int | None = None
    draw: int | None = None
    lost: int | None = None
    points: int | None = None
    goalsFor: int | None = None
    goalsAgainst: int | None = None
    goalDifference: int | None = None
    form: str | None = None


class StandingTable(ProviderModel):
    stage: str | None = None
    type: str | None = None
    group: str | None = None
    table: list[Standing] = Field(default_factory=list)


class ResponseSet(ProviderModel):
    count: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    competitions: list[Competition] = Field(default_factory=list)
    matches: list[Match] = Field(default_factory=list)
    teams: list[Team] = Field(default_factory=list)
    standings: list[StandingTable] = Field(default_factory=list)
    competition: Competition | None = None
    season: Season | None = None
    resultSet: dict[str, Any] | None = None
