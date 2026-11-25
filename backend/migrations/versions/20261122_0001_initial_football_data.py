"""Create football-data persistence tables.

Revision ID: 20261122_0001
Revises:
Create Date: 2026-11-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20261122_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=12), nullable=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=True),
        sa.Column("emblem", sa.String(length=500), nullable=True),
        sa.Column("area_provider_id", sa.Integer(), nullable=True),
        sa.Column("area_name", sa.String(length=120), nullable=True),
        sa.Column("area_code", sa.String(length=12), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_competitions"),
        sa.UniqueConstraint("provider_id", name="uq_competitions_provider_id"),
        sa.UniqueConstraint("code", name="uq_competitions_code"),
    )
    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=True),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("current_matchday", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"], name="fk_seasons_competition_id_competitions"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_seasons"),
        sa.UniqueConstraint("provider_id", name="uq_seasons_provider_id"),
        sa.UniqueConstraint(
            "competition_id", "start_date", name="uq_season_competition_start"
        ),
    )
    op.create_index("ix_seasons_competition_id", "seasons", ["competition_id"])

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("short_name", sa.String(length=120), nullable=True),
        sa.Column("tla", sa.String(length=8), nullable=True),
        sa.Column("crest", sa.String(length=500), nullable=True),
        sa.Column("venue", sa.String(length=160), nullable=True),
        sa.Column("area_provider_id", sa.Integer(), nullable=True),
        sa.Column("area_name", sa.String(length=120), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_teams"),
        sa.UniqueConstraint("provider_id", name="uq_teams_provider_id"),
    )
    op.create_table(
        "fixtures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider_id", sa.Integer(), nullable=False),
        sa.Column("competition_id", sa.Integer(), nullable=False),
        sa.Column("season_id", sa.Integer(), nullable=False),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("matchday", sa.Integer(), nullable=True),
        sa.Column("stage", sa.String(length=48), nullable=True),
        sa.Column("group", sa.String(length=48), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=False),
        sa.Column("away_team_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["competition_id"], ["competitions.id"], name="fk_fixtures_competition_id_competitions"
        ),
        sa.ForeignKeyConstraint(
            ["season_id"], ["seasons.id"], name="fk_fixtures_season_id_seasons"
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"], ["teams.id"], name="fk_fixtures_home_team_id_teams"
        ),
        sa.ForeignKeyConstraint(
            ["away_team_id"], ["teams.id"], name="fk_fixtures_away_team_id_teams"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_fixtures"),
        sa.UniqueConstraint("provider_id", name="uq_fixtures_provider_id"),
    )
    for column in (
        "competition_id",
        "season_id",
        "kickoff_at",
        "status",
        "home_team_id",
        "away_team_id",
    ):
        op.create_index(f"ix_fixtures_{column}", "fixtures", [column])

    op.create_table(
        "match_results",
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("home_score", sa.Integer(), nullable=False),
        sa.Column("away_score", sa.Integer(), nullable=False),
        sa.Column("winner", sa.String(length=16), nullable=True),
        sa.Column("source", sa.String(length=40), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("provider_metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(
            ["fixture_id"], ["fixtures.id"], name="fk_match_results_fixture_id_fixtures"
        ),
        sa.PrimaryKeyConstraint("fixture_id", name="pk_match_results"),
    )


def downgrade() -> None:
    op.drop_table("match_results")
    for column in (
        "away_team_id",
        "home_team_id",
        "status",
        "kickoff_at",
        "season_id",
        "competition_id",
    ):
        op.drop_index(f"ix_fixtures_{column}", table_name="fixtures")
    op.drop_table("fixtures")
    op.drop_table("teams")
    op.drop_index("ix_seasons_competition_id", table_name="seasons")
    op.drop_table("seasons")
    op.drop_table("competitions")
