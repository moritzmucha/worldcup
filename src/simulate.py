from __future__ import annotations
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from src.config import (
    DEFAULT_NEUTRAL,
    DEFAULT_TOURNAMENT,
    DRAW_SCORELINES,
    TOURNAMENT_SIMULATION_CSV,
    TOURNAMENT_SIMULATION_JSON,
    WIN_SCORELINES,
)
from src.predict import load_model_payload, matchup_probabilities


def load_tournament_spec(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if "groups" not in spec or "knockout" not in spec:
        raise ValueError("Tournament spec must contain groups and knockout sections.")
    return spec


def _sample_scoreline(outcome: int, rng: np.random.Generator) -> tuple[int, int]:
    if outcome == 1:
        return DRAW_SCORELINES[int(rng.integers(0, len(DRAW_SCORELINES)))]
    if outcome == 2:
        return WIN_SCORELINES[int(rng.integers(0, len(WIN_SCORELINES)))]
    goals_for, goals_against = WIN_SCORELINES[int(rng.integers(0, len(WIN_SCORELINES)))]
    return goals_against, goals_for


def _simulate_fixture(
    payload: dict[str, object],
    team_a: str,
    team_b: str,
    tournament: str,
    rng: np.random.Generator,
) -> tuple[int, int, np.ndarray]:
    probabilities = matchup_probabilities(
        payload, team_a, team_b, tournament=tournament, neutral=DEFAULT_NEUTRAL
    )
    outcome = int(rng.choice([0, 1, 2], p=probabilities))
    goals_a, goals_b = _sample_scoreline(outcome, rng)
    return goals_a, goals_b, probabilities


def _simulate_group(
    payload: dict[str, object],
    teams: list[str],
    tournament: str,
    rng: np.random.Generator,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    table = {
        team: {"points": 0.0, "gd": 0.0, "gf": 0.0, "ga": 0.0, "wins": 0.0}
        for team in teams
    }
    for team_a, team_b in combinations(teams, 2):
        goals_a, goals_b, _ = _simulate_fixture(
            payload, team_a, team_b, tournament, rng
        )
        table[team_a]["gf"] += goals_a
        table[team_a]["ga"] += goals_b
        table[team_a]["gd"] += goals_a - goals_b
        table[team_b]["gf"] += goals_b
        table[team_b]["ga"] += goals_a
        table[team_b]["gd"] += goals_b - goals_a

        if goals_a > goals_b:
            table[team_a]["points"] += 3
            table[team_a]["wins"] += 1
        elif goals_a < goals_b:
            table[team_b]["points"] += 3
            table[team_b]["wins"] += 1
        else:
            table[team_a]["points"] += 1
            table[team_b]["points"] += 1

    tiebreak_noise = {team: float(rng.random()) for team in teams}
    ranking = sorted(
        teams,
        key=lambda team: (
            table[team]["points"],
            table[team]["gd"],
            table[team]["gf"],
            table[team]["wins"],
            tiebreak_noise[team],
        ),
        reverse=True,
    )
    return ranking, table


def _resolve_reference(
    reference: str,
    group_rankings: dict[str, list[str]],
    knockout_winners: dict[str, str],
) -> str:
    if reference in knockout_winners:
        return knockout_winners[reference]
    group_name = reference[0]
    placement = int(reference[1:]) - 1
    return group_rankings[group_name][placement]


def _simulate_knockout_match(
    payload: dict[str, object],
    team_a: str,
    team_b: str,
    tournament: str,
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    _, _, probabilities = _simulate_fixture(payload, team_a, team_b, tournament, rng)
    decisive_mass = probabilities[0] + probabilities[2]
    if decisive_mass <= 0:
        penalty_win_probability = 0.5
    else:
        penalty_win_probability = float(probabilities[2] / decisive_mass)

    outcome = int(rng.choice([0, 1, 2], p=probabilities))
    if outcome == 2:
        return team_a, probabilities
    if outcome == 0:
        return team_b, probabilities
    return (team_a if rng.random() < penalty_win_probability else team_b), probabilities


def simulate_tournament(
    payload: dict[str, object],
    spec: dict[str, object],
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    tournament_name = str(spec.get("tournament", DEFAULT_TOURNAMENT))
    groups: dict[str, list[str]] = {
        name: list(teams) for name, teams in spec["groups"].items()
    }
    knockout_rounds: list[dict[str, object]] = list(spec["knockout"])

    teams = sorted({team for group in groups.values() for team in group})
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for team in teams:
        counts[team]["reach_group_stage"] = float(iterations)

    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        group_rankings: dict[str, list[str]] = {}
        for group_name, group_teams in groups.items():
            ranking, table = _simulate_group(payload, group_teams, tournament_name, rng)
            group_rankings[group_name] = ranking
            for team, stats in table.items():
                counts[team]["mean_group_points_total"] += stats["points"]
                counts[team]["mean_group_goal_diff_total"] += stats["gd"]

        knockout_winners: dict[str, str] = {}
        for round_index, round_spec in enumerate(knockout_rounds):
            round_name = str(round_spec["round"])
            matches = list(round_spec["matches"])
            round_teams = []
            for match in matches:
                team_a = _resolve_reference(
                    match["home"], group_rankings, knockout_winners
                )
                team_b = _resolve_reference(
                    match["away"], group_rankings, knockout_winners
                )
                round_teams.extend([team_a, team_b])
                winner, _ = _simulate_knockout_match(
                    payload, team_a, team_b, tournament_name, rng
                )
                knockout_winners[match["slot"]] = winner
            for team in round_teams:
                counts[team][f"reach_{round_name}"] += 1
            if round_index == 0:
                for team in round_teams:
                    counts[team]["advance_from_groups"] += 1

        champion = knockout_winners[list(knockout_rounds[-1]["matches"])[0]["slot"]]
        counts[champion]["champion"] += 1

    rows = []
    probability_columns = sorted(
        {
            column
            for team_counts in counts.values()
            for column in team_counts
            if not column.endswith("_total")
        }
    )
    for team in teams:
        row = {"team": team}
        for column in probability_columns:
            row[column] = counts[team][column] / iterations
        row["mean_group_points"] = counts[team]["mean_group_points_total"] / iterations
        row["mean_group_goal_diff"] = (
            counts[team]["mean_group_goal_diff_total"] / iterations
        )
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["champion", "advance_from_groups"], ascending=False)
        .reset_index(drop=True)
    )


def simulate_tournament_command(args) -> None:
    payload = load_model_payload(args.model_dir)
    spec = load_tournament_spec(args.spec)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probabilities = simulate_tournament(
        payload,
        spec,
        iterations=args.iterations,
        seed=args.seed,
    )
    probabilities.to_csv(output_dir / TOURNAMENT_SIMULATION_CSV, index=False)
    with (output_dir / TOURNAMENT_SIMULATION_JSON).open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(probabilities.to_dict(orient="records"), handle, indent=2)
    print(probabilities.to_string(index=False))
