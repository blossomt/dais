# app/src/main.py

from pathlib import Path

import html
import re

import pandas as pd
import streamlit as st

from src.recommender import (
    build_best_schedule,
    build_recommendations,
    compute_interest_embedding,
    compute_session_embeddings,
    load_model,
    rank_sessions,
)

from src.visualizer import (
    create_personal_schedule_timeline,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/agenda.csv")

TOP_K_SEARCH = 3
TOP_K_SCHEDULE = 15


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

    return load_model()


@st.cache_resource
def get_embeddings(
    searchable_texts: tuple,
):

    model = get_model()

    embeddings = compute_session_embeddings(
        model=model,
        df=pd.DataFrame({
            "searchable_text": searchable_texts
        }),
    )

    return embeddings


# ============================================================
# SEARCH
# ============================================================

def run_semantic_search(
    df: pd.DataFrame,
    query: str,
    embeddings,
    top_k: int,
):

    model = get_model()

    interest_embedding = (
        compute_interest_embedding(
            model,
            query,
        )
    )

    ranked_df = rank_sessions(
        df=df,
        session_embeddings=embeddings,
        interest_embedding=interest_embedding,
    )

    ranked_df = ranked_df.rename(
        columns={
            "semantic_score": "score"
        }
    )

    return ranked_df.head(top_k)


def run_keyword_search(
    df: pd.DataFrame,
    query: str,
    top_k: int,
):

    query = query.lower().strip()

    results = df[
        df["searchable_text"]
        .fillna("")
        .str.lower()
        .str.contains(
            query,
            regex=False,
        )
    ].copy()

    results["score"] = 1.0

    return results.head(top_k)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def prepare_display_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    display_df = df.copy()

    display_df["time"] = (
        display_df["starts_pst"]
        .dt.strftime("%a %H:%M")
        + " - "
        + display_df["ends_pst"]
        .dt.strftime("%H:%M")
    )

    return display_df


# ============================================================
# HIGHLIGHTING
# ============================================================

def highlight_keywords(
    text: str,
    query: str,
) -> str:

    if not isinstance(text, str):
        return ""

    escaped_text = html.escape(text)

    keywords = [
        k.strip()
        for k in query.split()
        if k.strip()
    ]

    for keyword in keywords:

        pattern = re.compile(
            re.escape(keyword),
            re.IGNORECASE,
        )

        escaped_text = pattern.sub(
            lambda m: (
                f'<span style="background-color:#ffe066;'
                f'padding:0.1rem 0.2rem;'
                f'border-radius:0.2rem;">'
                f'{m.group(0)}</span>'
            ),
            escaped_text,
        )

    return escaped_text


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
Search conference sessions using:
- keyword matching
- semantic similarity search
"""
)

query = st.text_input(
    (
        "Search for topics, tools, "
        "or interests"
    ),
    placeholder=(
        "lakebase geospatial analytics "
        "pipeline development unity catalog"
    ),
)

use_semantic_search = st.checkbox(
    "Semantic search",
    value=True,
)

df = load_data()

embeddings = None

if use_semantic_search:

    embeddings = get_embeddings(
        tuple(
            df["searchable_text"]
            .fillna("")
            .tolist()
        )
    )


# ============================================================
# SEARCH RESULTS
# ============================================================

if query:

    with st.spinner(
        "Searching sessions..."
    ):

        if use_semantic_search:

            results = run_semantic_search(
                df=df,
                query=query,
                embeddings=embeddings,
                top_k=TOP_K_SEARCH,
            )

        else:

            results = run_keyword_search(
                df=df,
                query=query,
                top_k=TOP_K_SEARCH,
            )

    st.subheader(
        "Top Matching Sessions"
    )

    if len(results) == 0:

        st.warning(
            "No matching sessions found."
        )

    else:

        display_df = prepare_display_table(
            results
        )

        for _, row in display_df.iterrows():

            title = highlight_keywords(
                row["title"],
                query,
            )

            description = (
                highlight_keywords(
                    row["description"],
                    query,
                )
            )

            track = highlight_keywords(
                row["track"],
                query,
            )

            score = (
                f"{row['score']:.3f}"
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    f"""
<div style="
    font-size:1.1rem;
    font-weight:700;
    margin-bottom:0.5rem;
">
{title}
</div>
""",
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(
                    [3, 1]
                )

                with col1:

                    st.markdown(
                        f"""
<div style="
    color:#6b7280;
    font-size:0.9rem;
">
🕒 {row['time']}
</div>

<div style="margin-top:0.5rem;">
<b>Track:</b> {track}
</div>
""",
                        unsafe_allow_html=True,
                    )

                with col2:

                    st.metric(
                        "Relevance score",
                        score,
                    )

                with st.expander(
                    "Description"
                ):

                    st.markdown(
                        f"""
<div style="line-height:1.6;">
{description}
</div>
""",
                        unsafe_allow_html=True,
                    )


# ============================================================
# PERSONALIZED SCHEDULE
# ============================================================

if query:

    st.divider()

    st.header(
        "Recommended Personalized Schedule"
    )

    if use_semantic_search:

        ranked_sessions = (
            run_semantic_search(
                df=df,
                query=query,
                embeddings=embeddings,
                top_k=TOP_K_SCHEDULE,
            )
        )

    else:

        ranked_sessions = (
            run_keyword_search(
                df=df,
                query=query,
                top_k=TOP_K_SCHEDULE,
            )
        )

    recommendations = (
        build_recommendations(
            ranked_sessions.rename(
                columns={
                    "score":
                    "semantic_score"
                }
            ),
            top_per_slot=1,
        )
    )

    schedule_df = (
        build_best_schedule(
            recommendations
        )
    )

    if len(schedule_df) == 0:

        st.warning(
            "No sessions available "
            "for schedule generation."
        )

    else:

        timeline_fig = (
            create_personal_schedule_timeline(
                schedule_df
            )
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True,
        )

        schedule_display = (
            schedule_df[
                [
                    "day",
                    "starts_pst",
                    "ends_pst",
                    "title",
                    "track",
                    "semantic_score",
                ]
            ]
            .copy()
        )

        schedule_display[
            "start_time"
        ] = (
            schedule_display["starts_pst"]
            .dt.strftime("%H:%M")
        )

        schedule_display[
            "end_time"
        ] = (
            schedule_display["ends_pst"]
            .dt.strftime("%H:%M")
        )

        schedule_display = (
            schedule_display[
                [
                    "day",
                    "start_time",
                    "end_time",
                    "title",
                    "track",
                    "semantic_score",
                ]
            ]
        )

        st.dataframe(
            schedule_display,
            use_container_width=True,
            hide_index=True,
        )