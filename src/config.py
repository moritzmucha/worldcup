from __future__ import annotations

RESULTS_SOURCE = "martj42/international_results/results.csv"
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "models"
DEFAULT_OUTPUT_DIR = "outputs"
MODEL_FILENAME = "xgb_worldcup.joblib"

DEFAULT_TOURNAMENT = "FIFA World Cup"
DEFAULT_MIN_DATE = "1990-01-01"
DEFAULT_TEST_START = "2022-01-01"
DEFAULT_NEUTRAL = 1

DEFAULT_START_ELO = 1500.0
DEFAULT_ROLLING_N = 10
DEFAULT_XG_ROLLING_N = 10
DEFAULT_N_ESTIMATORS = 500
DEFAULT_MAX_DEPTH = 3
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_HALF_LIFE_YEARS = 8.0

DEFAULT_SIMULATION_ITERATIONS = 5000
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_GOALS = 6
DEFAULT_MIN_GOALS_BUCKET = 3
DEFAULT_MAX_GOALS_BUCKET = 10
DAYS_PER_YEAR = 365.25

RAW_RESULTS_FILENAME = "results.csv"
TRAINING_FEATURES_FILENAME = "training_features.parquet"
GAIN_IMPORTANCE_FILENAME = "feature_importance_gain.csv"
SHAP_IMPORTANCE_FILENAME = "feature_importance_shap.csv"
METRICS_FILENAME = "metrics.json"
TOURNAMENT_SIMULATION_CSV = "tournament_simulation_probabilities.csv"
TOURNAMENT_SIMULATION_JSON = "tournament_simulation_probabilities.json"

XGB_RANDOM_STATE = 42
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_REG_LAMBDA = 2.0
XGB_MIN_CHILD_WEIGHT = 3
SHAP_SAMPLE_SIZE = 1000

APP_PAGE_TITLE = "World Cup Matchup Predictor"
APP_DESCRIPTION = (
    "Choose two countries to see win probabilities and a modeled scoreline heatmap."
)
DEFAULT_APP_TEAM_A = "Mexico"
DEFAULT_APP_TEAM_B = "South Africa"

CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Championship",
    "CONCACAF Gold Cup",
    "Oceania Nations Cup",
}

BASE_FEATURES = [
    "elo_diff",
    "elo_a",
    "elo_b",
    "gf10_diff",
    "ga10_diff",
    "gd10_diff",
    "pts10_diff",
    "home_advantage_diff",
    "neutral",
    "tournament_importance",
    "is_world_cup",
    "is_world_cup_qualifier",
    "is_continental_championship",
    "is_friendly",
]

OPTIONAL_FIFA_FEATURES = [
    "fifa_rank_a",
    "fifa_rank_b",
    "fifa_rank_diff",
    "fifa_points_a",
    "fifa_points_b",
    "fifa_points_diff",
]

OPTIONAL_MARKET_VALUE_FEATURES = [
    "squad_value_a",
    "squad_value_b",
    "squad_value_log_diff",
    "squad_value_ratio",
]

OPTIONAL_XG_FEATURES = [
    "xg_for10_a",
    "xg_for10_b",
    "xg_for10_diff",
    "xg_against10_a",
    "xg_against10_b",
    "xg_against10_diff",
    "xg_diff10_a",
    "xg_diff10_b",
    "xg_diff10_diff",
]

ALL_FEATURES = (
    BASE_FEATURES
    + OPTIONAL_FIFA_FEATURES
    + OPTIONAL_MARKET_VALUE_FEATURES
    + OPTIONAL_XG_FEATURES
)

DRAW_SCORELINES = [(0, 0), (1, 1), (2, 2)]
WIN_SCORELINES = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 1)]
