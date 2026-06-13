from __future__ import annotations
import numpy as np
from app import country_options, default_matchup, prediction_summary
from src.config import DEFAULT_APP_TEAM_A, DEFAULT_APP_TEAM_B


class FakeOutcomeModel:
    def predict_proba(self, _frame):
        return np.array([[0.25, 0.20, 0.55]])


def _fake_payload() -> dict[str, object]:
    latest = {
        DEFAULT_APP_TEAM_A: {
            "elo": 1600.0,
            "gf10": 1.5,
            "ga10": 0.8,
            "gd10": 0.7,
            "pts10": 2.1,
        },
        DEFAULT_APP_TEAM_B: {
            "elo": 1580.0,
            "gf10": 1.4,
            "ga10": 0.9,
            "gd10": 0.5,
            "pts10": 2.0,
        },
    }
    return {
        "model": FakeOutcomeModel(),
        "goal_model": object(),
        "features": ["elo_diff"],
        "latest": latest,
    }


def test_streamlit_app_helpers_work_with_fake_payload() -> None:
    payload = _fake_payload()

    assert country_options(payload) == sorted([DEFAULT_APP_TEAM_A, DEFAULT_APP_TEAM_B])
    assert default_matchup(country_options(payload)) == (
        DEFAULT_APP_TEAM_A,
        DEFAULT_APP_TEAM_B,
    )

    summary = prediction_summary(payload, DEFAULT_APP_TEAM_A, DEFAULT_APP_TEAM_B)

    assert summary["team_a_win"] == 0.55
    assert summary["draw"] == 0.20
    assert summary["team_b_win"] == 0.25
