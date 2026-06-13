# World Cup Matchup Predictor

International soccer prediction tools built around pre-match Elo/form features, XGBoost, and a Streamlit browser app.

The main app lets users pick two countries and see:

- win/draw probabilities;
- expected goals for each country;
- a scoreline probability heatmap;
- a score probability table.

## Run the App

```bash
uv sync
uv run streamlit run app.py
```

The app loads `models/xgb_worldcup.joblib`.

## CLI

Train or refresh the model:

```bash
uv run worldcup train --refresh
```

The `--refresh` flag downloads the current upstream `results.csv` before building features and fitting the model. Without it, training reuses the local `data/results.csv` cache.

Predict one matchup:

```bash
uv run worldcup predict-pair "Mexico" "South Africa"
```

Rank a set of teams:

```bash
uv run worldcup rank-teams "Mexico" "South Africa" "Argentina" "France"
```

Simulate a tournament:

```bash
uv run worldcup simulate-tournament --spec examples/world_cup_2022.json
```

## Data Source

The base match-results dataset is `results.csv` from [`martj42/international_results`](https://github.com/martj42/international_results).

That upstream dataset is licensed under **CC0-1.0**. The repository describes the data as men's full international football results from 1872 onward. This project uses that data to train the included model artifacts and generate features.

Optional loaders exist for FIFA ranking history, squad market value, and xG data, but those sources are not bundled here. Check the license and terms for any optional data source before publishing derived data or model artifacts trained with it.
