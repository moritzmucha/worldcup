from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Deque
from src.config import ALL_FEATURES
from src.config import DEFAULT_NEUTRAL, DEFAULT_TOURNAMENT
from src.elo import (
    elo_expected,
    elo_k,
    goal_margin_multiplier,
    points_for,
    result_targets,
    tournament_flags,
    tournament_importance,
)


@dataclass
class TeamSnapshot:
    elo: float
    gf10: float
    ga10: float
    gd10: float
    pts10: float
    matches_played: int
    fifa_rank: float | None = None
    fifa_points: float | None = None
    squad_value: float | None = None
    xg_for10: float | None = None
    xg_against10: float | None = None
    xg_diff10: float | None = None


class TeamHistoryLookup:
    def __init__(
        self, frame: pd.DataFrame | None, team_col: str, value_cols: list[str]
    ):
        self.by_team: dict[str, tuple[np.ndarray | None, list[dict[str, float]]]] = {}
        if frame is None or frame.empty:
            return

        if "date" in frame.columns:
            frame = frame.sort_values([team_col, "date"])
        else:
            frame = frame.sort_values([team_col])

        for team, group in frame.groupby(team_col, sort=False):
            records = group[value_cols].to_dict("records")
            dates = (
                group["date"].view("int64").to_numpy()
                if "date" in group.columns
                else None
            )
            self.by_team[str(team)] = (dates, records)

    def lookup(self, team: str, match_date: pd.Timestamp) -> dict[str, float]:
        payload = self.by_team.get(team)
        if payload is None:
            return {}

        dates, records = payload
        if dates is None:
            return dict(records[-1])

        target = pd.Timestamp(match_date).value
        index = int(np.searchsorted(dates, target, side="right") - 1)
        if index < 0:
            return {}
        return dict(records[index])


def mean_or_default(items: list[float], default: float) -> float:
    if not items:
        return default
    return float(np.mean(items))


def _safe_difference(left: float | None, right: float | None) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return np.nan
    return float(left - right)


def _safe_inverted_rank_diff(rank_a: float | None, rank_b: float | None) -> float:
    if rank_a is None or rank_b is None or pd.isna(rank_a) or pd.isna(rank_b):
        return np.nan
    return float(rank_b - rank_a)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if (
        numerator is None
        or denominator is None
        or pd.isna(numerator)
        or pd.isna(denominator)
        or denominator <= 0
    ):
        return np.nan
    return float(numerator / denominator)


def _safe_log_diff(value_a: float | None, value_b: float | None) -> float:
    if value_a is None or value_b is None or pd.isna(value_a) or pd.isna(value_b):
        return np.nan
    return float(np.log1p(value_a) - np.log1p(value_b))


def _rolling_snapshot(
    team: str,
    elo: dict[str, float],
    goal_hist: dict[str, Deque[tuple[float, float, float]]],
    xg_hist: dict[str, Deque[tuple[float, float]]] | None,
    fifa_lookup: TeamHistoryLookup | None,
    market_lookup: TeamHistoryLookup | None,
    match_date: pd.Timestamp,
) -> TeamSnapshot:
    matches = list(goal_hist[team])
    goals_for = mean_or_default([row[0] for row in matches], 1.2)
    goals_against = mean_or_default([row[1] for row in matches], 1.2)
    points = mean_or_default([row[2] for row in matches], 1.0)

    xg_for = None
    xg_against = None
    xg_diff = None
    if xg_hist is not None:
        xg_matches = list(xg_hist[team])
        xg_for = mean_or_default([row[0] for row in xg_matches], 1.2)
        xg_against = mean_or_default([row[1] for row in xg_matches], 1.2)
        xg_diff = xg_for - xg_against

    fifa_data = fifa_lookup.lookup(team, match_date) if fifa_lookup else {}
    market_data = market_lookup.lookup(team, match_date) if market_lookup else {}

    return TeamSnapshot(
        elo=float(elo[team]),
        gf10=goals_for,
        ga10=goals_against,
        gd10=goals_for - goals_against,
        pts10=points,
        matches_played=len(matches),
        fifa_rank=fifa_data.get("fifa_rank"),
        fifa_points=fifa_data.get("fifa_points"),
        squad_value=market_data.get("squad_value"),
        xg_for10=xg_for,
        xg_against10=xg_against,
        xg_diff10=xg_diff,
    )


def _xg_event_map(
    xg_matches: pd.DataFrame | None,
) -> dict[tuple[pd.Timestamp, str, str], Deque[tuple[float, float]]]:
    events: dict[tuple[pd.Timestamp, str, str], Deque[tuple[float, float]]] = (
        defaultdict(deque)
    )
    if xg_matches is None or xg_matches.empty:
        return events

    for row in xg_matches.itertuples(index=False):
        events[(row.date, row.team, row.opponent)].append(
            (float(row.xg), float(row.xga) if not pd.isna(row.xga) else 1.2)
        )
    return events


def _perspective_row(
    *,
    match_id: int,
    match_date: pd.Timestamp,
    tournament: str,
    neutral: int,
    team_a: str,
    team_b: str,
    target: int,
    goals_for: int,
    goals_against: int,
    snapshot_a: TeamSnapshot,
    snapshot_b: TeamSnapshot,
    home_advantage_diff: int,
) -> dict[str, object]:
    row = {
        "match_id": match_id,
        "date": match_date,
        "team_a": team_a,
        "team_b": team_b,
        "target": target,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "neutral": neutral,
        "tournament": tournament,
        "tournament_importance": tournament_importance(tournament),
        "elo_a": snapshot_a.elo,
        "elo_b": snapshot_b.elo,
        "elo_diff": snapshot_a.elo - snapshot_b.elo,
        "gf10_a": snapshot_a.gf10,
        "gf10_b": snapshot_b.gf10,
        "gf10_diff": snapshot_a.gf10 - snapshot_b.gf10,
        "ga10_a": snapshot_a.ga10,
        "ga10_b": snapshot_b.ga10,
        "ga10_diff": snapshot_a.ga10 - snapshot_b.ga10,
        "gd10_a": snapshot_a.gd10,
        "gd10_b": snapshot_b.gd10,
        "gd10_diff": snapshot_a.gd10 - snapshot_b.gd10,
        "pts10_a": snapshot_a.pts10,
        "pts10_b": snapshot_b.pts10,
        "pts10_diff": snapshot_a.pts10 - snapshot_b.pts10,
        "matches_played_a": snapshot_a.matches_played,
        "matches_played_b": snapshot_b.matches_played,
        "home_advantage_diff": home_advantage_diff,
        "source_home_score": pd.NA,
        "source_away_score": pd.NA,
        **tournament_flags(tournament),
    }

    row.update(
        {
            "fifa_rank_a": snapshot_a.fifa_rank,
            "fifa_rank_b": snapshot_b.fifa_rank,
            "fifa_rank_diff": _safe_inverted_rank_diff(
                snapshot_a.fifa_rank, snapshot_b.fifa_rank
            ),
            "fifa_points_a": snapshot_a.fifa_points,
            "fifa_points_b": snapshot_b.fifa_points,
            "fifa_points_diff": _safe_difference(
                snapshot_a.fifa_points, snapshot_b.fifa_points
            ),
            "squad_value_a": snapshot_a.squad_value,
            "squad_value_b": snapshot_b.squad_value,
            "squad_value_log_diff": _safe_log_diff(
                snapshot_a.squad_value, snapshot_b.squad_value
            ),
            "squad_value_ratio": _safe_ratio(
                snapshot_a.squad_value, snapshot_b.squad_value
            ),
            "xg_for10_a": snapshot_a.xg_for10,
            "xg_for10_b": snapshot_b.xg_for10,
            "xg_for10_diff": _safe_difference(snapshot_a.xg_for10, snapshot_b.xg_for10),
            "xg_against10_a": snapshot_a.xg_against10,
            "xg_against10_b": snapshot_b.xg_against10,
            "xg_against10_diff": _safe_difference(
                snapshot_a.xg_against10, snapshot_b.xg_against10
            ),
            "xg_diff10_a": snapshot_a.xg_diff10,
            "xg_diff10_b": snapshot_b.xg_diff10,
            "xg_diff10_diff": _safe_difference(
                snapshot_a.xg_diff10, snapshot_b.xg_diff10
            ),
        }
    )
    return row


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        feature
        for feature in ALL_FEATURES
        if feature in df.columns and not df[feature].isna().all()
    ]


def build_feature_history(
    df: pd.DataFrame,
    *,
    fifa_rankings: pd.DataFrame | None = None,
    market_values: pd.DataFrame | None = None,
    xg_matches: pd.DataFrame | None = None,
    start_elo: float = 1500.0,
    rolling_n: int = 10,
    xg_rolling_n: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | None]]]:
    ordered = df.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    elo = defaultdict(lambda: start_elo)
    goal_hist: dict[str, Deque[tuple[float, float, float]]] = defaultdict(
        lambda: deque(maxlen=rolling_n)
    )
    xg_hist: dict[str, Deque[tuple[float, float]]] | None = None
    if xg_matches is not None:
        xg_hist = defaultdict(lambda: deque(maxlen=xg_rolling_n))
    fifa_lookup = (
        TeamHistoryLookup(fifa_rankings, "country", ["fifa_rank", "fifa_points"])
        if fifa_rankings is not None
        else None
    )
    market_lookup = (
        TeamHistoryLookup(market_values, "country", ["squad_value"])
        if market_values is not None
        else None
    )
    xg_events = _xg_event_map(xg_matches)

    rows: list[dict[str, object]] = []
    for match_id, row in enumerate(ordered.itertuples(index=False)):
        home_team = row.home_team
        away_team = row.away_team
        home_score = int(row.home_score)
        away_score = int(row.away_score)
        match_date = row.date
        tournament = str(row.tournament)
        neutral = int(row.neutral)

        home_pre = _rolling_snapshot(
            home_team, elo, goal_hist, xg_hist, fifa_lookup, market_lookup, match_date
        )
        away_pre = _rolling_snapshot(
            away_team, elo, goal_hist, xg_hist, fifa_lookup, market_lookup, match_date
        )
        home_target, away_target, home_result = result_targets(home_score, away_score)

        home_row = _perspective_row(
            match_id=match_id,
            match_date=match_date,
            tournament=tournament,
            neutral=neutral,
            team_a=home_team,
            team_b=away_team,
            target=home_target,
            goals_for=home_score,
            goals_against=away_score,
            snapshot_a=home_pre,
            snapshot_b=away_pre,
            home_advantage_diff=0 if neutral else 1,
        )
        away_row = _perspective_row(
            match_id=match_id,
            match_date=match_date,
            tournament=tournament,
            neutral=neutral,
            team_a=away_team,
            team_b=home_team,
            target=away_target,
            goals_for=away_score,
            goals_against=home_score,
            snapshot_a=away_pre,
            snapshot_b=home_pre,
            home_advantage_diff=0 if neutral else -1,
        )

        home_row["source_home_score"] = home_score
        home_row["source_away_score"] = away_score
        away_row["source_home_score"] = home_score
        away_row["source_away_score"] = away_score
        rows.extend([home_row, away_row])

        expected_home = elo_expected(elo[home_team], elo[away_team])
        rating_delta = (
            elo_k(tournament)
            * goal_margin_multiplier(home_score - away_score)
            * (home_result - expected_home)
        )
        elo[home_team] += rating_delta
        elo[away_team] -= rating_delta

        goal_hist[home_team].append(
            (home_score, away_score, points_for(home_score, away_score))
        )
        goal_hist[away_team].append(
            (away_score, home_score, points_for(away_score, home_score))
        )

        if xg_hist is not None:
            home_event = xg_events.get((match_date, home_team, away_team))
            if home_event:
                xg_hist[home_team].append(home_event.popleft())
            away_event = xg_events.get((match_date, away_team, home_team))
            if away_event:
                xg_hist[away_team].append(away_event.popleft())

    feature_frame = pd.DataFrame(rows)
    teams = sorted(set(ordered["home_team"]).union(ordered["away_team"]))
    latest_date = ordered["date"].max()
    latest = {}
    for team in teams:
        snapshot = _rolling_snapshot(
            team,
            elo,
            goal_hist,
            xg_hist,
            fifa_lookup,
            market_lookup,
            latest_date + pd.Timedelta(days=1),
        )
        latest[team] = asdict(snapshot)
    return feature_frame, latest


def future_feature_row(
    team_a: str,
    team_b: str,
    latest: dict[str, dict[str, float | None]],
    *,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral: int = DEFAULT_NEUTRAL,
) -> dict[str, float | int | None]:
    if team_a not in latest:
        raise ValueError(f"Unknown team: {team_a}")
    if team_b not in latest:
        raise ValueError(f"Unknown team: {team_b}")

    snapshot_a = latest[team_a]
    snapshot_b = latest[team_b]
    row = {
        "elo_diff": float(snapshot_a["elo"] - snapshot_b["elo"]),
        "elo_a": float(snapshot_a["elo"]),
        "elo_b": float(snapshot_b["elo"]),
        "gf10_diff": float(snapshot_a["gf10"] - snapshot_b["gf10"]),
        "ga10_diff": float(snapshot_a["ga10"] - snapshot_b["ga10"]),
        "gd10_diff": float(snapshot_a["gd10"] - snapshot_b["gd10"]),
        "pts10_diff": float(snapshot_a["pts10"] - snapshot_b["pts10"]),
        "home_advantage_diff": 0 if neutral else 1,
        "neutral": int(neutral),
        "tournament_importance": tournament_importance(tournament),
        **tournament_flags(tournament),
    }
    row.update(
        {
            "fifa_rank_a": snapshot_a.get("fifa_rank"),
            "fifa_rank_b": snapshot_b.get("fifa_rank"),
            "fifa_rank_diff": _safe_inverted_rank_diff(
                snapshot_a.get("fifa_rank"), snapshot_b.get("fifa_rank")
            ),
            "fifa_points_a": snapshot_a.get("fifa_points"),
            "fifa_points_b": snapshot_b.get("fifa_points"),
            "fifa_points_diff": _safe_difference(
                snapshot_a.get("fifa_points"), snapshot_b.get("fifa_points")
            ),
            "squad_value_a": snapshot_a.get("squad_value"),
            "squad_value_b": snapshot_b.get("squad_value"),
            "squad_value_log_diff": _safe_log_diff(
                snapshot_a.get("squad_value"), snapshot_b.get("squad_value")
            ),
            "squad_value_ratio": _safe_ratio(
                snapshot_a.get("squad_value"), snapshot_b.get("squad_value")
            ),
            "xg_for10_a": snapshot_a.get("xg_for10"),
            "xg_for10_b": snapshot_b.get("xg_for10"),
            "xg_for10_diff": _safe_difference(
                snapshot_a.get("xg_for10"), snapshot_b.get("xg_for10")
            ),
            "xg_against10_a": snapshot_a.get("xg_against10"),
            "xg_against10_b": snapshot_b.get("xg_against10"),
            "xg_against10_diff": _safe_difference(
                snapshot_a.get("xg_against10"), snapshot_b.get("xg_against10")
            ),
            "xg_diff10_a": snapshot_a.get("xg_diff10"),
            "xg_diff10_b": snapshot_b.get("xg_diff10"),
            "xg_diff10_diff": _safe_difference(
                snapshot_a.get("xg_diff10"), snapshot_b.get("xg_diff10")
            ),
        }
    )
    return row
