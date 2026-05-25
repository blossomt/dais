# src/visualizer.py

import pandas as pd
import plotly.express as px


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# ============================================================
# RECOMMENDATION TIMELINE
# ============================================================

def create_recommendation_timeline(
    recommendations: pd.DataFrame,
):

    fig = px.timeline(
        recommendations,
        x_start="starts_pst",
        x_end="ends_pst",
        y="track",
        color="semantic_score",
        hover_data=[
            "title",
            "level",
            "type",
            "speakers",
            "categories",
            "semantic_score",
        ],
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_layout(
        title=(
            "Recommended Databricks "
            "Conference Schedule"
        ),
        width=1800,
        height=600,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    return fig


# ============================================================
# PERSONAL SCHEDULE TIMELINE
# ============================================================

def create_personal_schedule_timeline(
    best_per_slot: pd.DataFrame,
):

    viz_df = best_per_slot.copy()

    anchor_date = "2026-01-01 "

    viz_df["viz_start"] = pd.to_datetime(
        anchor_date
        + viz_df["starts_pst"]
        .dt.strftime("%H:%M:%S")
    )

    viz_df["viz_end"] = pd.to_datetime(
        anchor_date
        + viz_df["ends_pst"]
        .dt.strftime("%H:%M:%S")
    )

    fig = px.timeline(
        viz_df,
        y="day",
        x_start="viz_start",
        x_end="viz_end",
        color="track",
        hover_data={
            "title": True,
            "type": True,
            "speakers": True,
            "viz_start": False,
            "viz_end": False,
            "day": False,
            "semantic_score": False,
        },
    )

    fig.update_traces(
        width=0.6,
    )

    fig.update_yaxes(
        autorange="reversed",
        tickformat="%H:%M",
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=DAY_ORDER,
    )

    fig.update_layout(
        title="Optimized Personal Schedule",
        height=900,
        bargap=0.25,
        margin=dict(
            l=80,
            r=80,
            t=80,
            b=80,
        ),
        plot_bgcolor="white",
    )

    return fig


# ============================================================
# SAVE HTML
# ============================================================

def save_plotly_html(
    fig,
    output_path: str,
):

    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
    )