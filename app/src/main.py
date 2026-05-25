# app/src/main.py

from pathlib import Path
import sys

import re

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agenda_parser import apply_filters  # noqa: E402
from src.recommender import (  # noqa: E402
    build_best_schedule,
    build_recommendations,
    rank_sessions_by_topic,
)

from src.visualizer import (  # noqa: E402
    create_personal_schedule_timeline,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/agenda.csv")

TOP_N_RECOMMENDATIONS = 15
MIN_SEMANTIC_SCORE = 0.4

INCLUDE_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced",
]

EXCLUDE_TYPES = [
    # "Paid Training"
]

DEFAULT_INTEREST_TOPICS = [
    "geospatial spatial sql",
    "deploying lakebase databricks apps in production",
    "secure data sharing with unity catalog"
]


def parse_topics_input(raw_input: str) -> list[str]:

    if not raw_input:
        return []

    return [
        topic.strip()
        for topic in re.split(r"[\n,;]+", raw_input)
        if topic.strip()
    ]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:

    df = pd.read_csv(DATA_PATH)

    df["starts_pst"] = pd.to_datetime(
        df["starts_pst"]
    )

    df["ends_pst"] = pd.to_datetime(
        df["ends_pst"]
    )

    # Filter paid training
    df = df[
        df["type"] != "Paid Training"
    ]

    return df.reset_index(drop=True)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Databricks DAIS Agenda Planner",
    layout="wide",
)

st.title(
    "Databricks DAIS Agenda Planner"
)

st.markdown(
    """
Generate recommendations and a personal schedule
using topic-based semantic ranking.
"""
)

st.sidebar.header("Recommendation Controls")

top_n_recommendations = st.sidebar.slider(
    "Top recommended sessions",
    min_value=1,
    max_value=100,
    value=TOP_N_RECOMMENDATIONS,
    step=1,
)

min_semantic_score = st.sidebar.slider(
    "Minimum semantic score",
    min_value=0.0,
    max_value=1.0,
    value=MIN_SEMANTIC_SCORE,
    step=0.05,
)

topics_input = st.text_area(
    (
        "Topics of interest "
        "(one per line, or separated by commas/semicolons)"
    ),
    value="\n".join(DEFAULT_INTEREST_TOPICS),
    height=120,
)

df = load_data()

topics = parse_topics_input(topics_input)

if topics:

    with st.spinner("Building recommendations..."):

        filtered_df = apply_filters(
            df=df,
            include_levels=INCLUDE_LEVELS,
            exclude_types=EXCLUDE_TYPES,
        )

        ranked_df = rank_sessions_by_topic(
            filtered_df,
            topics,
        )

        recommendations = build_recommendations(
            ranked_df,
            top_per_topic=top_n_recommendations,
            min_semantic_score=min_semantic_score,
        )

        best_per_slot = build_best_schedule(
            recommendations
        )

    st.caption(
        f"Filtered sessions: {len(filtered_df)}"
    )

    st.subheader("Personalized Schedule")

    personal_schedule_fig = create_personal_schedule_timeline(
        best_per_slot
    )

    st.plotly_chart(
        personal_schedule_fig,
        use_container_width=True,
    )

    st.subheader("Recommended Sessions")

    st.dataframe(
        best_per_slot.head(top_n_recommendations)[
            [
                "nid",
                "timeslot",
                "title",
                "description",
                "interest_topic",
                "semantic_score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Enter one or more topics of interest to generate recommendations."
    )
