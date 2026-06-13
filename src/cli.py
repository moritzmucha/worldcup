from __future__ import annotations
import argparse
from src.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_HALF_LIFE_YEARS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_DATE,
    DEFAULT_MODEL_DIR,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_NEUTRAL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_ROLLING_N,
    DEFAULT_SIMULATION_ITERATIONS,
    DEFAULT_START_ELO,
    DEFAULT_TEST_START,
    DEFAULT_TOURNAMENT,
    DEFAULT_XG_ROLLING_N,
    RESULTS_URL,
)
from src.predict import predict_pair_command, rank_teams_command
from src.simulate import simulate_tournament_command
from src.train import train_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="International soccer World Cup prediction workflow with Elo-style features and XGBoost."
    )
    subcommands = parser.add_subparsers(dest="cmd", required=True)

    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("--results-url", default=RESULTS_URL)
    train_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    train_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--refresh", action="store_true")
    train_parser.add_argument("--min-date", default=DEFAULT_MIN_DATE)
    train_parser.add_argument("--test-start", default=DEFAULT_TEST_START)
    train_parser.add_argument("--start-elo", type=float, default=DEFAULT_START_ELO)
    train_parser.add_argument("--rolling-n", type=int, default=DEFAULT_ROLLING_N)
    train_parser.add_argument("--xg-rolling-n", type=int, default=DEFAULT_XG_ROLLING_N)
    train_parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    train_parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    train_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE
    )
    train_parser.add_argument(
        "--half-life-years", type=float, default=DEFAULT_HALF_LIFE_YEARS
    )
    train_parser.add_argument("--fifa-rankings-path")
    train_parser.add_argument("--market-values-path")
    train_parser.add_argument("--xg-path")
    train_parser.set_defaults(func=train_command)

    pair_parser = subcommands.add_parser("predict-pair")
    pair_parser.add_argument("team_a")
    pair_parser.add_argument("team_b")
    pair_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pair_parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT)
    pair_parser.add_argument("--neutral", type=int, default=DEFAULT_NEUTRAL)
    pair_parser.set_defaults(func=predict_pair_command)

    rank_parser = subcommands.add_parser("rank-teams")
    rank_parser.add_argument("teams", nargs="+")
    rank_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    rank_parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT)
    rank_parser.add_argument("--output")
    rank_parser.set_defaults(func=rank_teams_command)

    simulate_parser = subcommands.add_parser("simulate-tournament")
    simulate_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    simulate_parser.add_argument("--spec", required=True)
    simulate_parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    simulate_parser.add_argument(
        "--iterations", type=int, default=DEFAULT_SIMULATION_ITERATIONS
    )
    simulate_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    simulate_parser.set_defaults(func=simulate_tournament_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
