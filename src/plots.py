from __future__ import annotations
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from src.config import DEFAULT_MAX_GOALS, DEFAULT_NEUTRAL, DEFAULT_TOURNAMENT
from src.features import future_feature_row
from src.predict import load_model_payload


def _resolve_payload(
    payload_or_model_dir: dict[str, object] | str | Path,
) -> dict[str, object]:
    if isinstance(payload_or_model_dir, dict):
        return payload_or_model_dir

    path = Path(payload_or_model_dir)
    if path.is_file():
        return joblib.load(path)
    return load_model_payload(path)


def _poisson_bins(lam: float, max_goals: int) -> np.ndarray:
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")

    lam = max(float(lam), 1e-9)
    probabilities = np.array(
        [
            math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))
            for k in range(max_goals)
        ],
        dtype=float,
    )
    tail = max(0.0, 1.0 - float(probabilities.sum()))
    return np.append(probabilities, tail)


def score_probability_matrix(
    payload_or_model_dir: dict[str, object] | str | Path,
    team_a: str,
    team_b: str,
    *,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral: int = DEFAULT_NEUTRAL,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> pd.DataFrame:
    payload = _resolve_payload(payload_or_model_dir)
    goal_model = payload["goal_model"]
    features = payload["features"]
    latest = payload["latest"]
    row_a = pd.DataFrame(
        [
            future_feature_row(
                team_a, team_b, latest, tournament=tournament, neutral=neutral
            )
        ]
    )
    row_b = pd.DataFrame(
        [
            future_feature_row(
                team_b, team_a, latest, tournament=tournament, neutral=neutral
            )
        ]
    )

    lambda_a = float(np.clip(goal_model.predict(row_a[features])[0], 0.0, None))
    lambda_b = float(np.clip(goal_model.predict(row_b[features])[0], 0.0, None))
    probabilities_a = _poisson_bins(lambda_a, max_goals)
    probabilities_b = _poisson_bins(lambda_b, max_goals)

    labels = [str(goal) for goal in range(max_goals)] + [f"{max_goals}+"]
    return pd.DataFrame(
        np.outer(probabilities_b, probabilities_a),
        index=pd.Index(labels, name=f"{team_b} goals"),
        columns=pd.Index(labels, name=f"{team_a} goals"),
    )


def expected_goals(
    payload_or_model_dir: dict[str, object] | str | Path,
    team_a: str,
    team_b: str,
    *,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral: int = DEFAULT_NEUTRAL,
) -> tuple[float, float]:
    payload = _resolve_payload(payload_or_model_dir)
    goal_model = payload["goal_model"]
    features = payload["features"]
    latest = payload["latest"]
    row_a = pd.DataFrame(
        [
            future_feature_row(
                team_a, team_b, latest, tournament=tournament, neutral=neutral
            )
        ]
    )
    row_b = pd.DataFrame(
        [
            future_feature_row(
                team_b, team_a, latest, tournament=tournament, neutral=neutral
            )
        ]
    )
    goals_a = float(np.clip(goal_model.predict(row_a[features])[0], 0.0, None))
    goals_b = float(np.clip(goal_model.predict(row_b[features])[0], 0.0, None))
    return goals_a, goals_b


def plot_goal_probability_heatmap(
    payload_or_model_dir: dict[str, object] | str | Path,
    team_a: str,
    team_b: str,
    *,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral: int = DEFAULT_NEUTRAL,
    max_goals: int = DEFAULT_MAX_GOALS,
    ax: Any = None,
    cmap: str = "YlOrRd",
):
    import matplotlib.pyplot as plt

    matrix = score_probability_matrix(
        payload_or_model_dir,
        team_a,
        team_b,
        tournament=tournament,
        neutral=neutral,
        max_goals=max_goals,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    image = ax.imshow(matrix.to_numpy(), cmap=cmap, origin="upper")
    ax.set_xticks(np.arange(matrix.shape[1]), labels=list(matrix.columns))
    ax.set_yticks(np.arange(matrix.shape[0]), labels=list(matrix.index))
    ax.set_xlabel(matrix.columns.name)
    ax.set_ylabel(matrix.index.name)
    ax.set_title(f"Goal probability heatmap: {team_a} vs {team_b}")
    fig.colorbar(image, ax=ax, label="Probability")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{matrix.iat[row_index, col_index]:.1%}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig, ax, matrix
