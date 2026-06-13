from __future__ import annotations
import matplotlib.pyplot as plt
import streamlit as st
from src.config import (
    APP_DESCRIPTION,
    APP_PAGE_TITLE,
    DEFAULT_APP_TEAM_A,
    DEFAULT_APP_TEAM_B,
    DEFAULT_MAX_GOALS_BUCKET,
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_GOALS_BUCKET,
    DEFAULT_MODEL_DIR,
    DEFAULT_NEUTRAL,
    DEFAULT_TOURNAMENT,
)
from src.plots import expected_goals, plot_goal_probability_heatmap
from src.predict import load_model_payload, matchup_probabilities


@st.cache_resource
def cached_payload(model_dir: str = DEFAULT_MODEL_DIR) -> dict[str, object]:
    return load_model_payload(model_dir)


def country_options(payload: dict[str, object]) -> list[str]:
    return sorted(payload["latest"].keys())


def default_matchup(countries: list[str]) -> tuple[str, str]:
    team_a = DEFAULT_APP_TEAM_A if DEFAULT_APP_TEAM_A in countries else countries[0]
    fallback_index = min(1, len(countries) - 1)
    team_b = (
        DEFAULT_APP_TEAM_B
        if DEFAULT_APP_TEAM_B in countries
        else countries[fallback_index]
    )
    if team_a == team_b and len(countries) > 1:
        team_b = countries[1] if countries[0] == team_a else countries[0]
    return team_a, team_b


def prediction_summary(
    payload: dict[str, object],
    team_a: str,
    team_b: str,
    *,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral: int = DEFAULT_NEUTRAL,
) -> dict[str, float]:
    probabilities = matchup_probabilities(
        payload,
        team_a,
        team_b,
        tournament=tournament,
        neutral=neutral,
    )
    return {
        "team_a_win": float(probabilities[2]),
        "draw": float(probabilities[1]),
        "team_b_win": float(probabilities[0]),
    }


def main() -> None:
    st.set_page_config(page_title=APP_PAGE_TITLE, page_icon=":soccer:", layout="wide")
    st.title(APP_PAGE_TITLE)
    st.write(APP_DESCRIPTION)

    try:
        payload = cached_payload()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    countries = country_options(payload)
    default_a, default_b = default_matchup(countries)

    controls = st.columns([1, 1, 1])
    with controls[0]:
        team_a = st.selectbox("Country A", countries, index=countries.index(default_a))
    with controls[1]:
        team_b = st.selectbox("Country B", countries, index=countries.index(default_b))
    with controls[2]:
        max_goals = st.slider(
            "Max goals bucket",
            min_value=DEFAULT_MIN_GOALS_BUCKET,
            max_value=DEFAULT_MAX_GOALS_BUCKET,
            value=DEFAULT_MAX_GOALS,
            step=1,
        )

    if team_a == team_b:
        st.warning("Choose two different countries.")
        st.stop()

    summary = prediction_summary(payload, team_a, team_b)
    team_a_goals, team_b_goals = expected_goals(
        payload,
        team_a,
        team_b,
        tournament=DEFAULT_TOURNAMENT,
        neutral=DEFAULT_NEUTRAL,
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric(f"{team_a} win", f"{summary['team_a_win']:.1%}")
    metric_columns[1].metric("Draw", f"{summary['draw']:.1%}")
    metric_columns[2].metric(f"{team_b} win", f"{summary['team_b_win']:.1%}")
    metric_columns[3].metric(f"{team_a} E[goals]", f"{team_a_goals:.2f}")
    metric_columns[4].metric(f"{team_b} E[goals]", f"{team_b_goals:.2f}")

    fig, _ax, matrix = plot_goal_probability_heatmap(
        payload,
        team_a,
        team_b,
        tournament=DEFAULT_TOURNAMENT,
        neutral=DEFAULT_NEUTRAL,
        max_goals=max_goals,
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.subheader("Score Probability Matrix")
    st.dataframe((matrix * 100).round(2), use_container_width=True)

    trained_through = payload.get("trained_through")
    if trained_through:
        st.caption(f"Model trained through {trained_through}.")


if __name__ == "__main__":
    main()
