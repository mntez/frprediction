from datetime import date, datetime, timezone

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CompetitionRecord(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(Integer, unique=True)
    code: Mapped[str | None] = mapped_column(String(12), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    type: Mapped[str | None] = mapped_column(String(32))
    emblem: Mapped[str | None] = mapped_column(String(500))
    area_provider_id: Mapped[int | None] = mapped_column(Integer)
    area_name: Mapped[str | None] = mapped_column(String(120))
    area_code: Mapped[str | None] = mapped_column(String(12))
    source: Mapped[str] = mapped_column(String(40), default="football-data.org")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)

    seasons: Mapped[list["SeasonRecord"]] = relationship(back_populates="competition")


class SeasonRecord(Base):
    __tablename__ = "seasons"
    __table_args__ = (
        UniqueConstraint("competition_id", "start_date", name="uq_season_competition_start"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    current_matchday: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(40), default="football-data.org")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)

    competition: Mapped[CompetitionRecord] = relationship(back_populates="seasons")


class TeamRecord(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(Integer, unique=True)
    name: Mapped[str] = mapped_column(String(160))
    short_name: Mapped[str | None] = mapped_column(String(120))
    tla: Mapped[str | None] = mapped_column(String(8))
    crest: Mapped[str | None] = mapped_column(String(500))
    venue: Mapped[str | None] = mapped_column(String(160))
    area_provider_id: Mapped[int | None] = mapped_column(Integer)
    area_name: Mapped[str | None] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(40), default="football-data.org")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)


class FixtureRecord(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(Integer, unique=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), index=True)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    matchday: Mapped[int | None] = mapped_column(Integer)
    stage: Mapped[str | None] = mapped_column(String(48))
    group: Mapped[str | None] = mapped_column(String(48))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    source: Mapped[str] = mapped_column(String(40), default="football-data.org")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)

    competition: Mapped[CompetitionRecord] = relationship()
    season: Mapped[SeasonRecord] = relationship()
    home_team: Mapped[TeamRecord] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[TeamRecord] = relationship(foreign_keys=[away_team_id])
    result: Mapped["MatchResultRecord | None"] = relationship(
        back_populates="fixture", cascade="all, delete-orphan", uselist=False
    )


class MatchResultRecord(Base):
    __tablename__ = "match_results"

    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), primary_key=True)
    home_score: Mapped[int] = mapped_column(Integer)
    away_score: Mapped[int] = mapped_column(Integer)
    winner: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(40), default="football-data.org")
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    provider_metadata: Mapped[dict | None] = mapped_column(JSON)

    fixture: Mapped[FixtureRecord] = relationship(back_populates="result")
