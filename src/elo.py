from __future__ import annotations
from src.config import CONTINENTAL_TOURNAMENTS, DEFAULT_TOURNAMENT


def tournament_importance(tournament: str) -> float:
    tournament_name = str(tournament)
    lowered = tournament_name.lower()
    if tournament_name == DEFAULT_TOURNAMENT:
        return 5.0
    if "qualification" in lowered or "qualifier" in lowered:
        return 3.5
    if tournament_name in CONTINENTAL_TOURNAMENTS:
        return 4.0
    if "friendly" in lowered:
        return 1.0
    return 2.0


def tournament_flags(tournament: str) -> dict[str, int]:
    tournament_name = str(tournament)
    lowered = tournament_name.lower()
    return {
        "is_world_cup": int(tournament_name == DEFAULT_TOURNAMENT),
        "is_world_cup_qualifier": int(
            "fifa world cup qualification" in lowered
            or "world cup qualification" in lowered
        ),
        "is_continental_championship": int(tournament_name in CONTINENTAL_TOURNAMENTS),
        "is_friendly": int("friendly" in lowered),
    }


def elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def elo_k(tournament: str) -> float:
    return 12.0 + 8.0 * tournament_importance(tournament)


def goal_margin_multiplier(goal_diff: int) -> float:
    margin = abs(goal_diff)
    if margin <= 1:
        return 1.0
    return min(1.0 + 0.25 * (margin - 1), 1.75)


def points_for(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def result_targets(home_score: int, away_score: int) -> tuple[int, int, float]:
    if home_score > away_score:
        return 2, 0, 1.0
    if home_score == away_score:
        return 1, 1, 0.5
    return 0, 2, 0.0
