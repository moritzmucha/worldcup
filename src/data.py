from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.config import RESULTS_URL


def _read_table(path_or_url: str) -> pd.DataFrame:
    suffix = Path(path_or_url).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path_or_url)
    return pd.read_csv(path_or_url)


def _coerce_team_column(
    df: pd.DataFrame, candidates: list[str], target: str
) -> pd.DataFrame:
    for column in candidates:
        if column in df.columns:
            if column != target:
                df = df.rename(columns={column: target})
            return df
    raise ValueError(f"Missing required team column. Expected one of: {candidates}")


def _coerce_date_column(df: pd.DataFrame, required: bool = True) -> pd.DataFrame:
    if "date" not in df.columns:
        if required:
            raise ValueError("Missing required date column.")
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if required:
        df = df.dropna(subset=["date"])
    return df


def load_results(url_or_path: str = RESULTS_URL) -> pd.DataFrame:
    df = _read_table(url_or_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(
        subset=["date", "home_team", "away_team", "home_score", "away_score"]
    )
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = (
        df["neutral"]
        .astype(str)
        .str.upper()
        .map({"TRUE": 1, "FALSE": 0})
        .fillna(0)
        .astype(int)
    )
    return df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def load_optional_fifa_rankings(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None

    df = _read_table(path).copy()
    df = _coerce_date_column(df, required=True)
    df = _coerce_team_column(df, ["country", "team", "national_team"], "country")

    if "rank" not in df.columns:
        for candidate in ["fifa_rank", "ranking"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "rank"})
                break
    if "points" not in df.columns:
        for candidate in ["total_points", "fifa_points", "ranking_points"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "points"})
                break

    if "rank" not in df.columns:
        raise ValueError("FIFA rankings file must include a rank column.")

    if "points" not in df.columns:
        df["points"] = pd.NA

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df = df.dropna(subset=["country", "rank"])
    df = df.rename(columns={"rank": "fifa_rank", "points": "fifa_points"})
    return df[["date", "country", "fifa_rank", "fifa_points"]].sort_values(
        ["country", "date"]
    )


def load_optional_market_values(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None

    df = _read_table(path).copy()
    df = _coerce_team_column(df, ["country", "team", "national_team"], "country")
    df = _coerce_date_column(df, required=False)

    value_column = None
    for candidate in [
        "market_value",
        "squad_value",
        "squad_value_eur",
        "transfermarkt_value_eur",
    ]:
        if candidate in df.columns:
            value_column = candidate
            break
    if value_column is None:
        raise ValueError("Market value file must include a squad value column.")

    df = df.rename(columns={value_column: "squad_value"})
    df["squad_value"] = pd.to_numeric(df["squad_value"], errors="coerce")
    df = df.dropna(subset=["country", "squad_value"])

    columns = ["country", "squad_value"]
    if "date" in df.columns:
        columns.insert(0, "date")
    return df[columns].sort_values(columns[:2])


def load_optional_xg(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None

    df = _read_table(path).copy()
    df = _coerce_date_column(df, required=True)

    wide_required = {"home_team", "away_team", "home_xg", "away_xg"}
    if wide_required.issubset(df.columns):
        home = df[["date", "home_team", "away_team", "home_xg", "away_xg"]].rename(
            columns={
                "home_team": "team",
                "away_team": "opponent",
                "home_xg": "xg",
                "away_xg": "xga",
            }
        )
        away = df[["date", "home_team", "away_team", "home_xg", "away_xg"]].rename(
            columns={
                "away_team": "team",
                "home_team": "opponent",
                "away_xg": "xg",
                "home_xg": "xga",
            }
        )
        out = pd.concat([home, away], ignore_index=True)
    else:
        df = _coerce_team_column(df, ["team", "country", "national_team"], "team")
        df = _coerce_team_column(
            df, ["opponent", "opponent_team", "against"], "opponent"
        )
        if "xg" not in df.columns:
            raise ValueError("xG file must include xg or home_xg/away_xg columns.")
        if "xga" not in df.columns:
            for candidate in ["xg_against", "opponent_xg", "against_xg"]:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: "xga"})
                    break
        if "xga" not in df.columns:
            df["xga"] = pd.NA
        out = df[["date", "team", "opponent", "xg", "xga"]].copy()

    out["xg"] = pd.to_numeric(out["xg"], errors="coerce")
    out["xga"] = pd.to_numeric(out["xga"], errors="coerce")
    out = out.dropna(subset=["date", "team", "opponent", "xg"])
    return out.sort_values(["date", "team", "opponent"]).reset_index(drop=True)
