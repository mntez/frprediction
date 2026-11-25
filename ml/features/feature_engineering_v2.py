from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import log
from typing import Any, Iterable


SCHEMA_VERSION = "v2"
ELIGIBLE_STATUS = "FINISHED"
MIN_BASELINE_MATCHES = 20
MIN_VENUE_MATCHES = 5
STRENGTH_PRIOR = 5.0
ELO_INITIAL_RATING = 1500.0
ELO_SCALE = 400.0
ELO_K_FACTOR = 10.0
ELO_HOME_ADVANTAGE = 60.0
ELO_OFFSEASON_REGRESSION = 0.75
STRENGTH_HALF_LIFE_DAYS = 365.0
RECENT_WINDOW = 5

MATCH_INFORMATION_COLUMNS = [
    "provider_match_id", "competition", "season_provider_id", "season_start_year",
    "match_date", "home_team_provider_id", "home_team", "away_team_provider_id",
    "away_team",
]
FEATURE_COLUMNS = [
    "home_attack_strength", "home_defense_strength",
    "away_attack_strength", "away_defense_strength",
    "home_form_points_5", "away_form_points_5",
    "home_goals_for_5", "home_goals_against_5",
    "away_goals_for_5", "away_goals_against_5",
    "elo_difference",
    "home_history_matches", "away_history_matches",
]
TARGET_COLUMNS = [
    "home_goals", "away_goals", "home_win", "draw", "away_win", "total_goals",
    "over_1_5", "over_2_5", "over_3_5", "btts",
]
DATASET_COLUMNS = MATCH_INFORMATION_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


@dataclass(frozen=True)
class SourceMatch:
    provider_match_id: int
    kickoff_at: datetime
    status: str
    competition: str
    season_provider_id: int | None
    season_start_date: date | None
    home_team_provider_id: int
    home_team_name: str
    away_team_provider_id: int
    away_team_name: str
    home_goals: int
    away_goals: int
    winner: str | None = None

    @property
    def kickoff_utc(self) -> datetime:
        if self.kickoff_at.tzinfo is None:
            return self.kickoff_at.replace(tzinfo=timezone.utc)
        return self.kickoff_at.astimezone(timezone.utc)

    @property
    def match_date_utc(self) -> date:
        return self.kickoff_utc.date()

    @property
    def season_start_year(self) -> int:
        return self.season_start_date.year if self.season_start_date else self.kickoff_utc.year


@dataclass(frozen=True)
class _Appearance:
    match_date: date
    goals_for: int
    goals_against: int
    points: int
    venue: str
    attack_ratio: float | None
    concession_ratio: float | None


@dataclass
class _CompetitionGoals:
    home_matches: int = 0
    away_matches: int = 0
    home_goals: int = 0
    away_goals: int = 0

    def baselines(self) -> tuple[float | None, float | None]:
        home = (
            self.home_goals / self.home_matches
            if self.home_matches >= MIN_BASELINE_MATCHES and self.home_goals > 0 else None
        )
        away = (
            self.away_goals / self.away_matches
            if self.away_matches >= MIN_BASELINE_MATCHES and self.away_goals > 0 else None
        )
        return home, away

    def add(self, match: SourceMatch) -> None:
        self.home_matches += 1
        self.away_matches += 1
        self.home_goals += match.home_goals
        self.away_goals += match.away_goals


@dataclass
class _EloRating:
    rating: float
    season_start_year: int


def _strengths(
    history: list[_Appearance], *, target_date: date, baseline_ready: bool
) -> tuple[float | None, float | None]:
    if not baseline_ready:
        return None, None
    weighted_attack: list[tuple[float, float]] = []
    weighted_concession: list[tuple[float, float]] = []
    for item in history:
        if item.match_date >= target_date:
            continue
        age_days = max(0, (target_date - item.match_date).days)
        weight = 2 ** (-age_days / STRENGTH_HALF_LIFE_DAYS)
        if item.attack_ratio is not None:
            weighted_attack.append((weight, item.attack_ratio))
        if item.concession_ratio is not None:
            weighted_concession.append((weight, item.concession_ratio))

    if not weighted_attack:
        attack = 1.0
    else:
        attack = (STRENGTH_PRIOR + sum(w * value for w, value in weighted_attack)) / (
            STRENGTH_PRIOR + sum(w for w, _ in weighted_attack)
        )
    if not weighted_concession:
        defence = 1.0
    else:
        defence = (STRENGTH_PRIOR + sum(w for w, _ in weighted_concession)) / (
            STRENGTH_PRIOR + sum(w * value for w, value in weighted_concession)
        )
    return attack, defence


def _recent(
    history: list[_Appearance], *, target_date: date
) -> tuple[float | None, float | None, float | None]:
    prior = [item for item in history if item.match_date < target_date]
    recent = prior[-RECENT_WINDOW:]
    if not recent:
        return None, None, None
    count = len(recent)
    points = sum(item.points for item in recent) / count
    goals_for = sum(item.goals_for for item in recent) / count
    goals_against = sum(item.goals_against for item in recent) / count
    return points, goals_for, goals_against


def _season_rating(
    ratings: dict[tuple[str, int], _EloRating], key: tuple[str, int], year: int
) -> float:
    entry = ratings.get(key)
    if entry is None:
        ratings[key] = _EloRating(ELO_INITIAL_RATING, year)
        return ELO_INITIAL_RATING
    if year > entry.season_start_year:
        transitions = year - entry.season_start_year
        entry.rating = ELO_INITIAL_RATING + (
            entry.rating - ELO_INITIAL_RATING
        ) * (ELO_OFFSEASON_REGRESSION ** transitions)
        entry.season_start_year = year
    return entry.rating


def _elo_expected(home_rating: float, away_rating: float) -> float:
    exponent = -(
        home_rating - away_rating + ELO_HOME_ADVANTAGE
    ) / ELO_SCALE
    return 1.0 / (1.0 + 10**exponent)


def _targets(match: SourceMatch) -> dict[str, int]:
    total = match.home_goals + match.away_goals
    return {
        "home_goals": match.home_goals,
        "away_goals": match.away_goals,
        "home_win": int(match.home_goals > match.away_goals),
        "draw": int(match.home_goals == match.away_goals),
        "away_win": int(match.home_goals < match.away_goals),
        "total_goals": total,
        "over_1_5": int(total > 1.5),
        "over_2_5": int(total > 2.5),
        "over_3_5": int(total > 3.5),
        "btts": int(match.home_goals > 0 and match.away_goals > 0),
    }


def build_training_rows_v2(source_matches: Iterable[SourceMatch]) -> list[dict[str, Any]]:
    """Build v2 rows from FINISHED matches using only earlier UTC-date state."""
    all_matches = list(source_matches)
    all_ids = [match.provider_match_id for match in all_matches]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Source match provider IDs must be unique")
    matches = sorted(
        (match for match in all_matches if match.status == ELIGIBLE_STATUS),
        key=lambda item: (item.kickoff_utc, item.provider_match_id),
    )
    if any(match.home_goals < 0 or match.away_goals < 0 for match in matches):
        raise ValueError("Finished match scores cannot be negative")

    rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, int], list[_Appearance]] = defaultdict(list)
    competition_goals: dict[str, _CompetitionGoals] = defaultdict(_CompetitionGoals)
    elo: dict[tuple[str, int], _EloRating] = {}

    cursor = 0
    while cursor < len(matches):
        day = matches[cursor].match_date_utc
        end = cursor
        while end < len(matches) and matches[end].match_date_utc == day:
            end += 1
        day_matches = matches[cursor:end]
        date_baselines = {
            competition: state.baselines()
            for competition, state in competition_goals.items()
        }
        day_ratings: dict[int, tuple[float, float]] = {}

        # Snapshot all pre-match states for the date before any same-date result is applied.
        for match in day_matches:
            competition = match.competition
            home_key = (competition, match.home_team_provider_id)
            away_key = (competition, match.away_team_provider_id)
            home_history = histories[home_key]
            away_history = histories[away_key]
            home_baseline, away_baseline = date_baselines.get(competition, (None, None))
            baseline_ready = home_baseline is not None and away_baseline is not None
            home_attack, home_defense = _strengths(
                home_history, target_date=day, baseline_ready=baseline_ready
            )
            away_attack, away_defense = _strengths(
                away_history, target_date=day, baseline_ready=baseline_ready
            )
            home_form, home_for, home_against = _recent(home_history, target_date=day)
            away_form, away_for, away_against = _recent(away_history, target_date=day)

            home_rating = _season_rating(elo, home_key, match.season_start_year)
            away_rating = _season_rating(elo, away_key, match.season_start_year)
            day_ratings[match.provider_match_id] = (home_rating, away_rating)
            rows.append({
                "provider_match_id": match.provider_match_id,
                "competition": competition,
                "season_provider_id": match.season_provider_id,
                "season_start_year": match.season_start_year,
                "match_date": day.isoformat(),
                "home_team_provider_id": match.home_team_provider_id,
                "home_team": match.home_team_name,
                "away_team_provider_id": match.away_team_provider_id,
                "away_team": match.away_team_name,
                "home_attack_strength": home_attack,
                "home_defense_strength": home_defense,
                "away_attack_strength": away_attack,
                "away_defense_strength": away_defense,
                "home_form_points_5": home_form,
                "away_form_points_5": away_form,
                "home_goals_for_5": home_for,
                "home_goals_against_5": home_against,
                "away_goals_for_5": away_for,
                "away_goals_against_5": away_against,
                "elo_difference": home_rating - away_rating,
                "home_history_matches": len(home_history),
                "away_history_matches": len(away_history),
                **_targets(match),
            })

        # Results update form, strengths, competition means and Elo only after the UTC day.
        elo_deltas: dict[tuple[str, int], float] = defaultdict(float)
        for match in day_matches:
            competition = match.competition
            home_key = (competition, match.home_team_provider_id)
            away_key = (competition, match.away_team_provider_id)
            home_rating, away_rating = day_ratings[match.provider_match_id]
            expected = _elo_expected(home_rating, away_rating)
            actual = 1.0 if match.home_goals > match.away_goals else (
                0.5 if match.home_goals == match.away_goals else 0.0
            )
            delta = ELO_K_FACTOR * (actual - expected)
            elo_deltas[home_key] += delta
            elo_deltas[away_key] -= delta

            home_baseline, away_baseline = date_baselines.get(competition, (None, None))
            home_attack_ratio = match.home_goals / home_baseline if home_baseline else None
            home_concession_ratio = match.away_goals / away_baseline if away_baseline else None
            away_attack_ratio = match.away_goals / away_baseline if away_baseline else None
            away_concession_ratio = match.home_goals / home_baseline if home_baseline else None
            home_points = 3 if actual == 1.0 else 1 if actual == 0.5 else 0
            away_points = 3 if actual == 0.0 else 1 if actual == 0.5 else 0
            histories[home_key].append(_Appearance(
                day, match.home_goals, match.away_goals, home_points, "home",
                home_attack_ratio, home_concession_ratio,
            ))
            histories[away_key].append(_Appearance(
                day, match.away_goals, match.home_goals, away_points, "away",
                away_attack_ratio, away_concession_ratio,
            ))
            competition_goals[competition].add(match)

        for key, delta in elo_deltas.items():
            elo[key].rating += delta
        cursor = end

    rows.sort(key=lambda row: (row["match_date"], row["provider_match_id"]))
    return rows


def split_fixture_ids(
    rows: list[dict[str, Any]], cutoff_date: date
) -> tuple[list[int], list[int]]:
    ordered = sorted(rows, key=lambda row: (str(row["match_date"]), int(row["provider_match_id"])))
    training_ids = [int(row["provider_match_id"]) for row in ordered if date.fromisoformat(str(row["match_date"])) < cutoff_date]
    test_ids = [int(row["provider_match_id"]) for row in ordered if date.fromisoformat(str(row["match_date"])) >= cutoff_date]
    if not training_ids or not test_ids:
        raise ValueError("Cutoff must leave non-empty training and test fixture sets")
    if max(date.fromisoformat(str(row["match_date"])) for row in ordered if int(row["provider_match_id"]) in set(training_ids)) >= cutoff_date:
        raise ValueError("Training fixture date is not strictly before cutoff")
    return training_ids, test_ids


def assert_comparator_ids_match(
    v2_rows: list[dict[str, Any]],
    expanded_history_rows: list[dict[str, Any]],
    cutoff_date: date,
) -> tuple[list[int], list[int]]:
    v2_by_id = {int(row["provider_match_id"]): row for row in v2_rows}
    legacy_eligible = [
        row for row in expanded_history_rows
        if int(row["provider_match_id"]) in v2_by_id
    ]
    if {int(row["provider_match_id"]) for row in legacy_eligible} != set(v2_by_id):
        raise ValueError("Expanded-history data does not contain every eligible v2 fixture")
    v2_train, v2_test = split_fixture_ids(v2_rows, cutoff_date)
    legacy_train, legacy_test = split_fixture_ids(legacy_eligible, cutoff_date)
    if v2_train != legacy_train or v2_test != legacy_test:
        raise ValueError("Expanded-history and v2 comparator fixture ID splits differ")
    return v2_train, v2_test


def venue_residuals(
    history: list[_Appearance], *, venue: str, target_date: date,
    overall_attack: float | None, overall_defence: float | None,
) -> tuple[float | None, float | None]:
    """Optional group-C ablation helper; not included in the v2 core dataset."""
    if overall_attack is None or overall_defence is None:
        return None, None
    venue_history = [
        item for item in history
        if item.match_date < target_date and item.venue == venue
        and item.attack_ratio is not None and item.concession_ratio is not None
    ]
    if len(venue_history) < MIN_VENUE_MATCHES:
        return None, None
    weighted_attack = []
    weighted_concession = []
    for item in venue_history:
        age_days = max(0, (target_date - item.match_date).days)
        weight = 2 ** (-age_days / STRENGTH_HALF_LIFE_DAYS)
        weighted_attack.append((weight, item.attack_ratio))
        weighted_concession.append((weight, item.concession_ratio))
    venue_attack = (STRENGTH_PRIOR + sum(w * value for w, value in weighted_attack)) / (
        STRENGTH_PRIOR + sum(w for w, _ in weighted_attack)
    )
    venue_defence = (STRENGTH_PRIOR + sum(w for w, _ in weighted_concession)) / (
        STRENGTH_PRIOR + sum(w * value for w, value in weighted_concession)
    )
    return log(venue_attack) - log(overall_attack), log(venue_defence) - log(overall_defence)
