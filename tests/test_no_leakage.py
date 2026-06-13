from __future__ import annotations
import pandas as pd
from src.features import build_feature_history
from src.train import chronological_split


def _sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2021-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            },
            {
                "date": "2021-02-01",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            },
            {
                "date": "2021-03-01",
                "home_team": "Gamma",
                "away_team": "Alpha",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": False,
            },
        ]
    ).assign(date=lambda frame: pd.to_datetime(frame["date"]))


def test_first_match_features_do_not_use_same_match_result() -> None:
    features, _ = build_feature_history(_sample_results(), rolling_n=10)
    first_home_row = features[
        (features["match_id"] == 0) & (features["team_a"] == "Alpha")
    ].iloc[0]
    first_away_row = features[
        (features["match_id"] == 0) & (features["team_a"] == "Beta")
    ].iloc[0]

    assert first_home_row["gf10_a"] == 1.2
    assert first_home_row["ga10_a"] == 1.2
    assert first_home_row["pts10_a"] == 1.0
    assert first_home_row["matches_played_a"] == 0
    assert first_away_row["gf10_a"] == 1.2
    assert first_away_row["ga10_a"] == 1.2
    assert first_away_row["pts10_a"] == 1.0
    assert first_away_row["matches_played_a"] == 0


def test_rolling_stats_only_use_earlier_matches() -> None:
    features, _ = build_feature_history(_sample_results(), rolling_n=10)

    second_alpha_row = features[
        (features["match_id"] == 1) & (features["team_a"] == "Alpha")
    ].iloc[0]
    third_alpha_row = features[
        (features["match_id"] == 2) & (features["team_a"] == "Alpha")
    ].iloc[0]

    assert second_alpha_row["matches_played_a"] == 1
    assert second_alpha_row["gf10_a"] == 3.0
    assert second_alpha_row["ga10_a"] == 0.0
    assert second_alpha_row["pts10_a"] == 3.0

    assert third_alpha_row["matches_played_a"] == 2
    assert third_alpha_row["gf10_a"] == 2.0
    assert third_alpha_row["ga10_a"] == 0.5
    assert third_alpha_row["pts10_a"] == 2.0


def test_chronological_split_has_no_date_overlap() -> None:
    features, _ = build_feature_history(_sample_results(), rolling_n=10)
    train_df, test_df = chronological_split(features, "2021-03-01")

    assert train_df["date"].max() < pd.Timestamp("2021-03-01")
    assert test_df["date"].min() >= pd.Timestamp("2021-03-01")


def test_goal_targets_are_from_team_perspective_for_neutral_matches() -> None:
    results = pd.DataFrame(
        [
            {
                "date": "2021-01-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 4,
                "away_score": 2,
                "tournament": "Friendly",
                "city": "X",
                "country": "Y",
                "neutral": True,
            }
        ]
    ).assign(date=lambda frame: pd.to_datetime(frame["date"]))

    features, _ = build_feature_history(results, rolling_n=10)
    alpha_row = features[features["team_a"] == "Alpha"].iloc[0]
    beta_row = features[features["team_a"] == "Beta"].iloc[0]

    assert alpha_row["goals_for"] == 4
    assert alpha_row["goals_against"] == 2
    assert alpha_row["home_advantage_diff"] == 0
    assert beta_row["goals_for"] == 2
    assert beta_row["goals_against"] == 4
    assert beta_row["home_advantage_diff"] == 0
