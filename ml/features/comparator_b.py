"""Versioned data shaping for the original six-input Poisson comparator."""

from __future__ import annotations

from datetime import date
from typing import Any

from ml.baselines.poisson import TEAM_FEATURES
from ml.features.feature_engineering import (
    FEATURE_COLUMNS as LEGACY_COLUMNS,
    TARGET_COLUMNS,
    CompletedMatch,
    build_training_rows,
)

MATCH_INFORMATION_COLUMNS = (
    "provider_match_id", "competition", "season_provider_id", "season_start_year",
    "match_date", "home_team_provider_id", "home_team", "away_team_provider_id", "away_team",
)
COMPARATOR_B_COLUMNS = (*MATCH_INFORMATION_COLUMNS, *TEAM_FEATURES, *TARGET_COLUMNS)
EXCLUDED_FIXTURE_IDS = frozenset({542704})
CUTOFF_DATE = date(2026, 10, 31)

FEATURE_DEFINITIONS = {
    "home_home_matches_played": "Count of prior home-venue appearances by the home club, across competitions, before the target UTC date.",
    "home_home_goals_scored_avg": "Mean goals scored by the home club in those prior home-venue appearances.",
    "home_home_goals_conceded_avg": "Mean goals conceded by the home club in those prior home-venue appearances.",
    "away_away_matches_played": "Count of prior away-venue appearances by the away club, across competitions, before the target UTC date.",
    "away_away_goals_scored_avg": "Mean goals scored by the away club in those prior away-venue appearances.",
    "away_away_goals_conceded_avg": "Mean goals conceded by the away club in those prior away-venue appearances.",
}


def build_comparator_b_rows(matches: list[CompletedMatch]) -> list[dict[str, Any]]:
    """Reproduce the v1 six inputs, excluding the specifically ineligible award.

    The caller supplies FINISHED source matches only. The original feature builder
    is reused so team history remains global by provider team ID, matching v1.
    Its same-UTC-date batching prevents own-result and same-day leakage.
    """
    eligible = [m for m in matches if m.provider_match_id not in EXCLUDED_FIXTURE_IDS]
    rows = build_training_rows(eligible)
    rows.sort(key=lambda row: (str(row["match_date"]), int(row["provider_match_id"])))
    return [{column: row.get(column) for column in COMPARATOR_B_COLUMNS} for row in rows]

