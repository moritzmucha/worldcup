from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import DEFAULT_NEUTRAL, MODEL_FILENAME
from src.features import future_feature_row


def load_model_payload(model_dir: str | Path) -> dict[str, object]:
    return joblib.load(Path(model_dir) / MODEL_FILENAME)


def matchup_probabilities(
    payload: dict[str, object],
    team_a: str,
    team_b: str,
    *,
    tournament: str,
    neutral: int,
) -> np.ndarray:
    model = payload["model"]
    latest = payload["latest"]
    features = payload["features"]
    row = pd.DataFrame(
        [
            future_feature_row(
                team_a, team_b, latest, tournament=tournament, neutral=neutral
            )
        ]
    )
    return model.predict_proba(row[features])[0]


def predict_pair_command(args) -> None:
    payload = load_model_payload(args.model_dir)
    probabilities = matchup_probabilities(
        payload,
        args.team_a,
        args.team_b,
        tournament=args.tournament,
        neutral=int(args.neutral),
    )
    result = {
        "team_a": args.team_a,
        "team_b": args.team_b,
        "p_team_a_loss": float(probabilities[0]),
        "p_draw": float(probabilities[1]),
        "p_team_a_win": float(probabilities[2]),
        "expected_points_team_a": float(3 * probabilities[2] + probabilities[1]),
    }
    print(json.dumps(result, indent=2))


def rank_teams_command(args) -> None:
    payload = load_model_payload(args.model_dir)
    rows = []
    for team in args.teams:
        expected_points = []
        win_probabilities = []
        for opponent in args.teams:
            if opponent == team:
                continue
            probabilities = matchup_probabilities(
                payload,
                team,
                opponent,
                tournament=args.tournament,
                neutral=DEFAULT_NEUTRAL,
            )
            expected_points.append(3 * probabilities[2] + probabilities[1])
            win_probabilities.append(probabilities[2])
        rows.append(
            {
                "team": team,
                "avg_expected_points_vs_field": float(np.mean(expected_points)),
                "avg_win_probability_vs_field": float(np.mean(win_probabilities)),
                "latest_internal_elo": float(payload["latest"][team]["elo"]),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        "avg_expected_points_vs_field", ascending=False
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(args.output, index=False)
    print(ranking.to_string(index=False))
