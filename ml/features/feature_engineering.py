from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any


ROLLING_WINDOW = 5
MIN_HEAD_TO_HEAD_MATCHES = 3
COMPLETED_STATUSES = frozenset({"FINISHED", "AWARDED"})
TARGET_COLUMNS = [
    "home_goals", "away_goals", "home_win", "draw", "away_win", "total_goals",
    "over_1_5", "over_2_5", "over_3_5", "btts",
]


@dataclass(frozen=True)
class CompletedMatch:
    provider_match_id: int
    kickoff_at: datetime
    competition: str | None
    season_provider_id: int | None
    season_start_date: date | None
    home_team_provider_id: int
    home_team_name: str
    away_team_provider_id: int
    away_team_name: str
    home_goals: int
    away_goals: int

    @property
    def kickoff_utc(self) -> datetime:
        value = self.kickoff_at
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Appearance:
    goals_for: int
    goals_against: int
    outcome: str
    venue: str


def _appearance(match: CompletedMatch, venue: str) -> Appearance:
    if venue == "home":
        scored, conceded = match.home_goals, match.away_goals
    else:
        scored, conceded = match.away_goals, match.home_goals
    outcome = "W" if scored > conceded else "D" if scored == conceded else "L"
    return Appearance(scored, conceded, outcome, venue)


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _team_features(history: list[Appearance], *, venue: str, prefix: str) -> dict[str, Any]:
    outcomes = Counter(item.outcome for item in history)
    rolling = history[-ROLLING_WINDOW:]
    venue_history = [item for item in history if item.venue == venue]
    wins = sum(item.outcome == "W" for item in venue_history)
    return {
        f"{prefix}_previous_matches_played": len(history),
        f"{prefix}_previous_wins": outcomes["W"],
        f"{prefix}_previous_draws": outcomes["D"],
        f"{prefix}_previous_losses": outcomes["L"],
        f"{prefix}_previous_goals_scored": sum(item.goals_for for item in history),
        f"{prefix}_previous_goals_conceded": sum(item.goals_against for item in history),
        f"{prefix}_rolling_avg_goals_scored": _mean([item.goals_for for item in rolling]),
        f"{prefix}_rolling_avg_goals_conceded": _mean([item.goals_against for item in rolling]),
        f"{prefix}_recent_form": "".join(item.outcome for item in rolling) or None,
        f"{prefix}_{venue}_matches_played": len(venue_history),
        f"{prefix}_{venue}_win_rate": wins / len(venue_history) if venue_history else None,
        f"{prefix}_{venue}_goals_scored_avg": _mean([item.goals_for for item in venue_history]),
        f"{prefix}_{venue}_goals_conceded_avg": _mean([item.goals_against for item in venue_history]),
    }


def _head_to_head_features(
    history: list[CompletedMatch], target_home_id: int
) -> dict[str, int | None]:
    if len(history) < MIN_HEAD_TO_HEAD_MATCHES:
        return {
            "h2h_previous_matches": None,
            "h2h_home_team_wins": None,
            "h2h_draws": None,
            "h2h_home_team_losses": None,
        }
    outcomes = []
    for match in history:
        if match.home_team_provider_id == target_home_id:
            home_goals, away_goals = match.home_goals, match.away_goals
        else:
            home_goals, away_goals = match.away_goals, match.home_goals
        outcomes.append("W" if home_goals > away_goals else "D" if home_goals == away_goals else "L")
    counts = Counter(outcomes)
    return {
        "h2h_previous_matches": len(history),
        "h2h_home_team_wins": counts["W"],
        "h2h_draws": counts["D"],
        "h2h_home_team_losses": counts["L"],
    }


FEATURE_COLUMNS = [
    "provider_match_id", "competition", "season_provider_id", "season_start_year",
    "match_date", "home_team_provider_id", "home_team", "away_team_provider_id", "away_team",
    "home_previous_matches_played", "home_previous_wins", "home_previous_draws",
    "home_previous_losses", "home_previous_goals_scored", "home_previous_goals_conceded",
    "home_rolling_avg_goals_scored", "home_rolling_avg_goals_conceded", "home_recent_form",
    "home_home_matches_played", "home_home_win_rate", "home_home_goals_scored_avg",
    "home_home_goals_conceded_avg",
    "away_previous_matches_played", "away_previous_wins", "away_previous_draws",
    "away_previous_losses", "away_previous_goals_scored", "away_previous_goals_conceded",
    "away_rolling_avg_goals_scored", "away_rolling_avg_goals_conceded", "away_recent_form",
    "away_away_matches_played", "away_away_win_rate", "away_away_goals_scored_avg",
    "away_away_goals_conceded_avg",
    "h2h_previous_matches", "h2h_home_team_wins", "h2h_draws", "h2h_home_team_losses",
    *TARGET_COLUMNS,
]


def build_training_rows(matches: list[CompletedMatch]) -> list[dict[str, Any]]:
    """Build chronological rows; only results from earlier UTC dates enter features."""
    ids = [match.provider_match_id for match in matches]
    if len(ids) != len(set(ids)):
        raise ValueError("Completed matches must have unique provider match IDs")
    ordered = sorted(
        matches,
        key=lambda match: (match.kickoff_utc, match.provider_match_id),
    )
    rows: list[dict[str, Any]] = []
    team_history: dict[int, list[Appearance]] = defaultdict(list)
    h2h_history: dict[frozenset[int], list[CompletedMatch]] = defaultdict(list)

    index = 0
    while index < len(ordered):
        match_date = ordered[index].kickoff_utc.date()
        end = index
        while end < len(ordered) and ordered[end].kickoff_utc.date() == match_date:
            end += 1
        day_matches = ordered[index:end]

        # All fixtures on a UTC date are featurized before any result from that date
        # is added to history. This avoids same-day and simultaneous-match leakage.
        for match in day_matches:
            home_id = match.home_team_provider_id
            away_id = match.away_team_provider_id
            row: dict[str, Any] = {
                "provider_match_id": match.provider_match_id,
                "competition": match.competition,
                "season_provider_id": match.season_provider_id,
                "season_start_year": match.season_start_date.year if match.season_start_date else None,
                "match_date": match_date.isoformat(),
                "home_team_provider_id": home_id,
                "home_team": match.home_team_name,
                "away_team_provider_id": away_id,
                "away_team": match.away_team_name,
            }
            row.update(_team_features(team_history[home_id], venue="home", prefix="home"))
            row.update(_team_features(team_history[away_id], venue="away", prefix="away"))
            row.update(_head_to_head_features(h2h_history[frozenset((home_id, away_id))], home_id))

            total_goals = match.home_goals + match.away_goals
            row.update({
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "home_win": int(match.home_goals > match.away_goals),
                "draw": int(match.home_goals == match.away_goals),
                "away_win": int(match.home_goals < match.away_goals),
                "total_goals": total_goals,
                "over_1_5": int(total_goals > 1.5),
                "over_2_5": int(total_goals > 2.5),
                "over_3_5": int(total_goals > 3.5),
                "btts": int(match.home_goals > 0 and match.away_goals > 0),
            })
            rows.append(row)

        for match in day_matches:
            team_history[match.home_team_provider_id].append(_appearance(match, "home"))
            team_history[match.away_team_provider_id].append(_appearance(match, "away"))
            h2h_history[frozenset((match.home_team_provider_id, match.away_team_provider_id))].append(match)
        index = end

    return rows
