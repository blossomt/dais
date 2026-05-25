# main.py

from pathlib import Path

from src.agenda_parser import (
    apply_filters,
    parse_agenda,
)

from src.recommender import (
    build_best_schedule,
    build_recommendations,
    compute_session_embeddings,
    compute_topic_embeddings,
    load_model,
    rank_sessions_by_topic,
)

from src.visualizer import (
    create_personal_schedule_timeline,
    create_recommendation_timeline,
    save_plotly_html,
)


# ============================================================
# CONFIG
# ============================================================

AGENDA_PATH = "data/agenda.json"

OUTPUT_DIR = Path("output")

INTEREST_TOPICS = [
     "geospatial analytics",
     "industry use cases",
     "supply chain planning",
     "transport analytics",
     "urban planning",
     "data warehousing for analytics",
     "deploying data pipelines to production",
     "data lineage and observability",
     "developer experience",
]

INCLUDE_LEVELS = [
     "Beginner",
     "Intermediate",
     "Advanced",
]

EXCLUDE_TYPES = []

TOP_N_RECOMMENDATIONS = 15


# ============================================================
# LOAD
# ============================================================

print("Parsing agenda...")

df = parse_agenda(
    AGENDA_PATH,
)

print(f"Loaded {len(df)} sessions")

df.sort_values(
    "starts_pst"
).to_csv(
    "./data/agenda.csv",
    index=False,
)


# ============================================================
# FILTER
# ============================================================

filtered_df = apply_filters(
    df=df,
    include_levels=INCLUDE_LEVELS,
    exclude_types=EXCLUDE_TYPES,
)

print(
    f"After filtering: "
    f"{len(filtered_df)} sessions"
)


# ============================================================
# EMBEDDINGS
# ============================================================

print("Loading embedding model...")

model = load_model()

print("Computing session embeddings...")

session_embeddings = (
    compute_session_embeddings(
        model,
        filtered_df,
     )
)

print("Computing topic embeddings...")

topics, topic_embeddings = compute_topic_embeddings(
    model,
    INTEREST_TOPICS,
)

print(f"Matched against {len(topics)} topics: {', '.join(topics)}")


# ============================================================
# RANKING (per-topic)
# ============================================================

ranked_df = rank_sessions_by_topic(
    filtered_df,
    session_embeddings,
    topics,
    topic_embeddings,
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

recommendations = build_recommendations(
    ranked_df,
    top_per_slot=3,
)

# ============================================================
# BEST SCHEDULE
# ============================================================

best_per_slot = build_best_schedule(
    recommendations
)


# ============================================================
# EXPORT CSVS
# ============================================================

OUTPUT_DIR.mkdir(
    exist_ok=True,
)

recommendations[[
      "nid",
      "title",
      "day",
      "timeslot",
      "interest_topic",
      "semantic_score",
    ]].to_csv(
    OUTPUT_DIR / "recommended_sessions.csv",
    index=False,
)

best_per_slot[[
            "nid",
            "title",
            "day",
            "timeslot",
            "interest_topic",
            "semantic_score",
        ]].to_csv(
    OUTPUT_DIR / "best_schedule.csv",
    index=False,
)


# ============================================================
# VISUALIZATIONS
# ============================================================

recommendation_fig = (
    create_recommendation_timeline(
        recommendations
     )
)

personal_schedule_fig = (
    create_personal_schedule_timeline(
        best_per_slot
     )
)

save_plotly_html(
    recommendation_fig,
    OUTPUT_DIR
    / "recommendation_timeline.html",
)

save_plotly_html(
    personal_schedule_fig,
    OUTPUT_DIR
    / "personal_schedule_timeline.html",
)

print("\nSaved outputs to ./output/")
