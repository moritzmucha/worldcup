from __future__ import annotations
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from src.plots import plot_goal_probability_heatmap, score_probability_matrix


class FakeGoalModel:
    def __init__(self) -> None:
        self.predictions = [1.4, 0.9]

    def predict(self, _frame):
        return np.array([self.predictions.pop(0)])


def _fake_payload() -> dict[str, object]:
    latest = {
        "Alpha": {"elo": 1600.0, "gf10": 1.5, "ga10": 0.8, "gd10": 0.7, "pts10": 2.1},
        "Beta": {"elo": 1500.0, "gf10": 1.1, "ga10": 1.2, "gd10": -0.1, "pts10": 1.4},
    }
    return {"goal_model": FakeGoalModel(), "features": ["elo_diff"], "latest": latest}


def test_score_probability_matrix_has_labeled_tail_bins_and_sums_to_one() -> None:
    matrix = score_probability_matrix(_fake_payload(), "Alpha", "Beta", max_goals=3)

    assert matrix.shape == (4, 4)
    assert matrix.columns.name == "Alpha goals"
    assert matrix.index.name == "Beta goals"
    assert list(matrix.columns) == ["0", "1", "2", "3+"]
    assert list(matrix.index) == ["0", "1", "2", "3+"]
    assert matrix.to_numpy().sum() == np.float64(1.0)


def test_plot_goal_probability_heatmap_returns_figure_axes_and_matrix() -> None:
    fig, ax, matrix = plot_goal_probability_heatmap(
        _fake_payload(), "Alpha", "Beta", max_goals=3
    )

    assert fig is ax.figure
    assert matrix.shape == (4, 4)
    assert ax.get_xlabel() == "Alpha goals"
    assert ax.get_ylabel() == "Beta goals"
    plt.close(fig)
