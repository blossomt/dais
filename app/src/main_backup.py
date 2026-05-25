# app/src/main.py

from pathlib import Path

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import html
import re
import plotly.express as px

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

    # Ensure datetime parsing
    df["starts_pst"] = pd.to_datetime(df["starts_pst"])
    df["ends_pst"] = pd.to_datetime(df["ends_pst"])

    # Filter 
    df = df[df["type"] != "Paid Training"]

    return df


# ============================================================
# SEMANTIC MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def build_embeddings(texts: list[str]):
    model = load_model()

    embeddings = model.encode(
        texts,
        show_progress_bar=False,
    )

    return embeddings


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def exact_search(
    df: pd.DataFrame,
    query: str,
    top_k: int = TOP_K_SEARCH,
) -> pd.DataFrame:

    query = query.lower().strip()

    results = df[
        df["searchable_text"]
        .fillna("")
        .str.lower()
        .str.contains(query, regex=False)
    ].copy()

    results["score"] = 1.0

    return results.head(top_k)


def semantic_search(
    df: pd.DataFrame,
    query: str,
    embeddings,
    top_k: int = TOP_K_SEARCH,
) -> pd.DataFrame:

    model = load_model()

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    results = df.copy()

    results["score"] = similarities

    results = (
        results
        .sort_values("score", ascending=False)
        .head(top_k)
    )

    return results


# ============================================================
# DISPLAY HELPERS
# ============================================================

def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:

    display_df = df.copy()

    display_df["time"] = (
        display_df["starts_pst"]
        .dt.strftime("%a %H:%M")
        + " - "
        + display_df["ends_pst"]
        .dt.strftime("%H:%M")
    )

    columns = [
        "time",
        "title",
        "description",
        "track",
        "score",
    ]

    existing_columns = [
        c for c in columns
        if c in display_df.columns
    ]

    return display_df[existing_columns]

# ============================================================
# HIGHLIGHTING
# ============================================================

def highlight_keywords(text: str, query: str) -> str:

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
# PERSONALIZED SCHEDULE
# ============================================================

def get_ranked_sessions(
    df: pd.DataFrame,
    query: str,
    use_semantic_search: bool,
    embeddings=None,
) -> pd.DataFrame:

    if use_semantic_search:

        results = semantic_search(
            df=df,
            query=query,
            embeddings=embeddings,
            top_k=TOP_K_SCHEDULE,
        )

    else:

        results = exact_search(
            df=df,
            query=query,
            top_k=TOP_K_SCHEDULE,
        )

    return results


def build_personalized_schedule(
    ranked_df: pd.DataFrame,
) -> pd.DataFrame:

    # choose best session per overlapping timeslot
    schedule_df = (
        ranked_df
        .sort_values("score", ascending=False)
        .groupby(
            [
                "starts_pst",
            ]
        )
        .head(1)
        .copy()
    )

    return schedule_df.sort_values("starts_pst")


def create_schedule_timeline(
    schedule_df: pd.DataFrame,
):

    viz_df = schedule_df.copy()

    # anchor to same fake date so plotly can render time vertically
    anchor_date = "2026-01-01 "

    viz_df["viz_start"] = pd.to_datetime(
        anchor_date
        + viz_df["starts_pst"].dt.strftime("%H:%M:%S")
    )

    viz_df["viz_end"] = pd.to_datetime(
        anchor_date
        + viz_df["ends_pst"].dt.strftime("%H:%M:%S")
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig = px.timeline(
        viz_df,
        x_start="viz_start",
        x_end="viz_end",
        y="day",
        color="score",
        hover_data={
            "title": True,
            "type": True,
            "speakers": True,

            # hide unused fields
            "viz_start": True,
            "viz_end": True,
            "day": False,
            "level": False,
            "categories": False,
            "score": False,
        },
    )

    fig.update_traces(
        width=0.6,
    )

    fig.update_yaxes(
        autorange="reversed",
        tickformat="%H:%M",
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=day_order,
    )

    fig.update_layout(
        title="Recommended Personalized Schedule",
        height=800,
        bargap=0.25,
        margin=dict(
            l=80,
            r=80,
            t=80,
            b=80,
        ),
        hoverlabel=dict(
            align="left"
        )
    )

    return fig

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Databricks DAIS Agenda Planner",
    layout="wide",
)

st.title("Databricks DAIS Agenda Planner")

st.markdown(
    """
Search conference sessions using:
- keyword matching
- semantic similarity search
"""
)

query = st.text_input(
    "Search for topics, tools, or interests - multiple search terms works!",
    placeholder="scaling up pipeline development lakebase databricks apps geospatial analytics",
)

use_semantic_search = st.checkbox(
    "Semantic search",
    value=True,
)

df = load_data()

if use_semantic_search:
    embeddings = build_embeddings(
        df["searchable_text"]
        .fillna("")
        .tolist()
    )

if query:

    with st.spinner("Searching sessions..."):

        if use_semantic_search:
            results = semantic_search(
                df=df,
                query=query,
                embeddings=embeddings,
            )
        else:
            results = exact_search(
                df=df,
                query=query,
            )

    st.subheader("Top Matching Sessions")

    if len(results) == 0:
        st.warning("No matching sessions found.")

    else:

        display_df = prepare_display_table(results)

        for _, row in display_df.iterrows():

            title = highlight_keywords(
                row["title"],
                query,
            )

            description = highlight_keywords(
                row["description"],
                query,
            )

            track = highlight_keywords(
                row["track"],
                query,
            )

            score = (
                f"{row['score']:.3f}"
                if "score" in row
                else "-"
            )

            with st.container(border=True):

                st.markdown(
                    f"""
        <div style="
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        ">
        {title}
        </div>
        """,
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(
                        f"""
        <div style="
            color: #6b7280;
            font-size: 0.9rem;
        ">
        🕒 {row['time']}
        </div>

        <div style="
            margin-top: 0.5rem;
        ">
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

                with st.expander("Description"):
                    st.markdown(
                        f"""
        <div style="
            line-height: 1.6;
        ">
        {description}
        </div>
        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# PERSONALIZED SCHEDULE VIEW
# ============================================================

if query:

    st.divider()

    st.header("Recommended Personalized Schedule")

    ranked_sessions = get_ranked_sessions(
        df=df,
        query=query,
        use_semantic_search=use_semantic_search,
        embeddings=embeddings if use_semantic_search else None,
    )

    schedule_df = build_personalized_schedule(
        ranked_sessions,
    )

    if len(schedule_df) == 0:

        st.warning("No sessions available for schedule generation.")

    else:

        timeline_fig = create_schedule_timeline(
            schedule_df,
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True,
        )

        schedule_display = schedule_df[
            [
                "day",
                "starts_pst",
                "ends_pst",
                "title",
                "track",
                "score",
                "url",
            ]
        ].copy()

        schedule_display["start_time"] = (
            schedule_display["starts_pst"]
            .dt.strftime("%H:%M")
        )

        schedule_display["end_time"] = (
            schedule_display["ends_pst"]
            .dt.strftime("%H:%M")
        )

        schedule_display = schedule_display[
            [
                "day",
                "start_time",
                "end_time",
                "title",
                "track",
                "score",
            ]
        ]

        st.dataframe(
            schedule_display,
            use_container_width=True,
            hide_index=True,
        )