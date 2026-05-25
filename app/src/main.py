# app/src/main.py

from pathlib import Path
import hashlib
import sys

import numpy as np
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
    compute_session_embeddings,
    compute_topic_embeddings,
    load_model,
    rank_sessions_by_topic,
)

from src.visualizer import (  # noqa: E402
    create_personal_schedule_timeline,
    create_recommendation_timeline,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/agenda.csv")

CACHE_DIR = REPO_ROOT / ".cache"
MODEL_CACHE_DIR = CACHE_DIR / "models"
EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"

TOP_PER_SLOT = 3
TOP_N_RECOMMENDATIONS = 15

INCLUDE_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced",
]

EXCLUDE_TYPES = []

DEFAULT_INTEREST_TOPICS = [
    "geospatial analytics",
    "deploying lakebase databricks apps",
    "unity catalog data sharing"
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
# MODEL + EMBEDDINGS
# ============================================================

@st.cache_resource
def get_model():

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return load_model(cache_folder=MODEL_CACHE_DIR)


@st.cache_resource
def get_embeddings(
    searchable_texts: tuple,
):

    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    texts_hash = hashlib.md5(
        "\n".join(searchable_texts).encode()
    ).hexdigest()

    cache_file = EMBEDDINGS_CACHE_DIR / f"{texts_hash}.npy"

    if cache_file.exists():
        return np.load(str(cache_file))

    model = get_model()

    embeddings = compute_session_embeddings(
        model=model,
        df=pd.DataFrame({
            "searchable_text": searchable_texts
        }),
    )

    np.save(str(cache_file), embeddings)

    return embeddings


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

        model = get_model()

        session_embeddings = get_embeddings(
            tuple(
                filtered_df["searchable_text"]
                .fillna("")
                .tolist()
            )
        )

        topics, topic_embeddings = compute_topic_embeddings(
            model,
            topics,
        )

        ranked_df = rank_sessions_by_topic(
            filtered_df,
            session_embeddings,
            topics,
            topic_embeddings,
        )

        recommendations = build_recommendations(
            ranked_df,
            top_per_slot=TOP_PER_SLOT,
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
        best_per_slot[
            [
                "nid",
                "title",
                "day",
                "timeslot",
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
