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

    category_col = (
        "interest_topic"
        if "interest_topic" in recommendations.columns
        else "track"
    )

    hover_fields = [
        "title",
        "level",
        "type",
        "speakers",
        "categories",
        "semantic_score",
    ]

    if category_col not in hover_fields:
        hover_fields.append(category_col)

    fig = px.timeline(
        recommendations,
        x_start="starts_pst",
        x_end="ends_pst",
        y=category_col,
        color="semantic_score",
        color_continuous_scale=px.colors.sequential.Blues,
        hover_data=hover_fields,
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
        hoverlabel={
            "bgcolor": "white",
            "font_size": 12,
        },
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
        color="interest_topic",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={
            "title": True,
            "type": True,
            "speakers": True,
            "interest_topic": "interest_topic" in viz_df.columns,
            "viz_start": True,
            "viz_end": True,
            "day": False,
            "semantic_score": False,
        },
        opacity=0.7,
    )

    fig.update_traces(
        width=0.6,
        marker_line_color="white",
        marker_line_width=2,
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
        bargap=0.8,
        margin={
            "l": 80,
            "r": 80,
            "t": 80,
            "b": 80,
        },
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
