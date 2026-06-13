from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from src.config import (
    DAYS_PER_YEAR,
    DEFAULT_HALF_LIFE_YEARS,
    GAIN_IMPORTANCE_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    RAW_RESULTS_FILENAME,
    RESULTS_SOURCE,
    SHAP_IMPORTANCE_FILENAME,
    SHAP_SAMPLE_SIZE,
    TRAINING_FEATURES_FILENAME,
    XGB_COLSAMPLE_BYTREE,
    XGB_MIN_CHILD_WEIGHT,
    XGB_RANDOM_STATE,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)
from src.data import (
    load_optional_fifa_rankings,
    load_optional_market_values,
    load_optional_xg,
    load_results,
)
from src.features import build_feature_history, get_feature_columns


def recency_weights(
    dates: pd.Series, half_life_years: float = DEFAULT_HALF_LIFE_YEARS
) -> np.ndarray:
    newest_date = dates.max()
    age_years = (newest_date - dates).dt.days / DAYS_PER_YEAR
    return np.power(0.5, age_years / half_life_years).to_numpy()


def chronological_split(
    feature_frame: pd.DataFrame, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(test_start)
    train_df = feature_frame[feature_frame["date"] < cutoff].copy()
    test_df = feature_frame[feature_frame["date"] >= cutoff].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            "Empty train or test split. Adjust --min-date or --test-start."
        )
    return train_df, test_df


def _gain_importance(model: Any, features: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    importance = pd.DataFrame(
        {"feature": features, "gain": [gain.get(feature, 0.0) for feature in features]}
    )
    gain_sum = float(importance["gain"].sum())
    importance["gain_share"] = 0.0 if gain_sum == 0 else importance["gain"] / gain_sum
    return importance.sort_values("gain", ascending=False).reset_index(drop=True)


def _shap_importance(
    model: Any, X: pd.DataFrame
) -> tuple[pd.DataFrame | None, str | None]:
    try:
        import shap  # type: ignore
    except ImportError:
        return None, "shap_not_installed"

    try:
        sample = (
            X
            if len(X) <= SHAP_SAMPLE_SIZE
            else X.sample(SHAP_SAMPLE_SIZE, random_state=XGB_RANDOM_STATE)
        )
        explainer = shap.TreeExplainer(model)
        explanation = explainer(sample)
        values = np.asarray(explanation.values)

        if values.ndim == 3:
            mean_abs = np.abs(values).mean(axis=(0, 2))
        elif values.ndim == 2:
            mean_abs = np.abs(values).mean(axis=0)
        else:
            return None, f"unexpected_shap_shape_{values.shape}"

        importance = pd.DataFrame(
            {"feature": list(sample.columns), "mean_abs_shap": mean_abs}
        )
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(
            drop=True
        ), None
    except (
        Exception
    ) as exc:  # pragma: no cover - defensive path for optional dependency
        return None, str(exc)


def train_model(args) -> dict[str, object]:
    from xgboost import XGBClassifier, XGBRegressor

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = data_dir / RAW_RESULTS_FILENAME
    if args.refresh or not raw_path.exists():
        raw_results = load_results(args.results_url)
        raw_results.to_csv(raw_path, index=False)
    else:
        raw_results = load_results(str(raw_path))

    if args.min_date:
        raw_results = raw_results[
            raw_results["date"] >= pd.Timestamp(args.min_date)
        ].copy()

    fifa_rankings = load_optional_fifa_rankings(args.fifa_rankings_path)
    market_values = load_optional_market_values(args.market_values_path)
    xg_matches = load_optional_xg(args.xg_path)

    feature_frame, latest = build_feature_history(
        raw_results,
        fifa_rankings=fifa_rankings,
        market_values=market_values,
        xg_matches=xg_matches,
        start_elo=args.start_elo,
        rolling_n=args.rolling_n,
        xg_rolling_n=args.xg_rolling_n,
    )
    feature_columns = get_feature_columns(feature_frame)
    train_df, test_df = chronological_split(feature_frame, args.test_start)

    X_train = train_df[feature_columns]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df["target"].astype(int)
    y_train_goals = train_df["goals_for"].astype(float)
    y_test_goals = test_df["goals_for"].astype(float)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        reg_lambda=XGB_REG_LAMBDA,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        eval_metric="mlogloss",
        random_state=XGB_RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=recency_weights(train_df["date"], args.half_life_years),
    )

    goal_model = XGBRegressor(
        objective="count:poisson",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        reg_lambda=XGB_REG_LAMBDA,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        eval_metric="poisson-nloglik",
        random_state=XGB_RANDOM_STATE,
    )
    goal_model.fit(
        X_train,
        y_train_goals,
        sample_weight=recency_weights(train_df["date"], args.half_life_years),
    )

    probabilities = model.predict_proba(X_test)
    predictions = np.argmax(probabilities, axis=1)
    goal_predictions = np.clip(goal_model.predict(X_test), 0.0, None)
    metrics = {
        "dataset_source": RESULTS_SOURCE,
        "trained_through": str(raw_results["date"].max().date()),
        "test_start": args.test_start,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_features": int(len(feature_columns)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1, 2])),
        "goal_mae": float(mean_absolute_error(y_test_goals, goal_predictions)),
        "goal_rmse": float(np.sqrt(mean_squared_error(y_test_goals, goal_predictions))),
    }

    gain_importance = _gain_importance(model, feature_columns)
    shap_importance, shap_status = _shap_importance(model, X_test)
    metrics["shap_status"] = "ok" if shap_importance is not None else shap_status

    payload = {
        "model": model,
        "goal_model": goal_model,
        "features": feature_columns,
        "latest": latest,
        "trained_through": metrics["trained_through"],
        "config": {
            "start_elo": args.start_elo,
            "rolling_n": args.rolling_n,
            "xg_rolling_n": args.xg_rolling_n,
        },
    }

    joblib.dump(payload, model_dir / MODEL_FILENAME)
    feature_frame.to_parquet(out_dir / TRAINING_FEATURES_FILENAME, index=False)
    gain_importance.to_csv(out_dir / GAIN_IMPORTANCE_FILENAME, index=False)
    if shap_importance is not None:
        shap_importance.to_csv(out_dir / SHAP_IMPORTANCE_FILENAME, index=False)
    with (out_dir / METRICS_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "metrics": metrics,
        "gain_importance": gain_importance,
        "shap_importance": shap_importance,
        "feature_columns": feature_columns,
        "feature_frame": feature_frame,
    }


def train_command(args) -> None:
    artifacts = train_model(args)
    print(json.dumps(artifacts["metrics"], indent=2))
    print("\nFeature importance by gain:")
    print(artifacts["gain_importance"].to_string(index=False))
    if artifacts["shap_importance"] is not None:
        print("\nFeature importance by SHAP:")
        print(artifacts["shap_importance"].to_string(index=False))
