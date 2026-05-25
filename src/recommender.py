# src/recommender.py

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# MODEL
# ============================================================

def load_model(cache_folder=None):

    kwargs = {}

    if cache_folder is not None:
        kwargs["cache_folder"] = str(cache_folder)

    return SentenceTransformer(MODEL_NAME, **kwargs)


# ============================================================
# EMBEDDINGS
# ============================================================

def compute_session_embeddings(
    model,
    df: pd.DataFrame,
):

    return model.encode(
        df["searchable_text"].tolist(),
        show_progress_bar=True,
     )


def _parse_interest_lines(
    interest_profile: str | list[str],
) -> list[str]:

    if isinstance(interest_profile, list):
        return [line.strip() for line in interest_profile if line.strip()]

    return [
        line.strip()
        for line in interest_profile.splitlines()
        if line.strip()
      ]


def compute_interest_embedding(
    model,
    interest_profile: str,
):

    interest_lines = _parse_interest_lines(interest_profile)

    line_embeddings = model.encode(
        interest_lines
      )

    return (
        line_embeddings
         .mean(axis=0)
         .reshape(1, -1)
      )


def compute_topic_embeddings(
    model,
    topics: list[str],
) -> tuple[list[str], np.ndarray]:

    topics = _parse_interest_lines(topics)

    return topics, model.encode(topics)


# ============================================================
# RANKING
# ============================================================

def rank_sessions(
    df: pd.DataFrame,
    session_embeddings,
    interest_embedding,
) -> pd.DataFrame:

    scores = cosine_similarity(
        interest_embedding,
        session_embeddings,
      )[0]

    ranked = df.copy()

    ranked["semantic_score"] = scores

    ranked = ranked.sort_values(
         "semantic_score",
        ascending=False,
      ).reset_index(drop=True)

    return ranked


def rank_sessions_by_topic(
    df: pd.DataFrame,
    session_embeddings,
    topics: list[str],
    topic_embeddings: np.ndarray,
) -> pd.DataFrame:

    ranked = df.copy()

    for idx, topic in enumerate(topics):

        scores = cosine_similarity(
            topic_embeddings[idx : idx + 1],
            session_embeddings,
          )[0]

        ranked[f"semantic_score_{topic}"] = scores

    score_cols = [f"semantic_score_{t}" for t in topics]

    ranked["interest_topic"] = ranked[score_cols].idxmax(axis=1).str.replace(
        "semantic_score_", ""
    )

    ranked["semantic_score"] = ranked[score_cols].max(axis=1)

    ranked = ranked.sort_values(
         "semantic_score",
        ascending=False,
      ).reset_index(drop=True)

    return ranked


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    ranked_df: pd.DataFrame,
    top_per_slot: int = 3,
    top_per_topic: int | None = 5,
) -> pd.DataFrame:

    candidates = ranked_df.copy()

    if (
        top_per_topic is not None
        and "interest_topic" in candidates.columns
        and "semantic_score" in candidates.columns
    ):

        candidates = (
            candidates
            .sort_values(
                "semantic_score",
                ascending=False,
            )
            .groupby("interest_topic")
            .head(top_per_topic)
            .copy()
        )

    recommendations = (
        candidates
         .sort_values(
             "semantic_score",
            ascending=False,
         )
         .groupby(["starts_pst"])
         .head(top_per_slot)
         .sort_values(
             ["starts_pst", "semantic_score"],
            ascending=[True, False],
         )
         .copy()
     )

    recommendations["timeslot"] = (
        recommendations["starts_pst"]
         .dt.strftime("%a %H:%M")
         + " → "
         + recommendations["ends_pst"]
         .dt.strftime("%H:%M")
      )

    return recommendations


def build_best_schedule(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:

    best_per_slot = (
        recommendations
         .sort_values(
             "semantic_score",
            ascending=False,
         )
         .groupby(
             [
                 "starts_pst",
             ]
         )
         .head(1)
         .sort_values("starts_pst")
         .copy()
      )

    return best_per_slot


# ============================================================
# CONFLICTS
# ============================================================

def detect_conflicts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    conflicts = []

    sorted_df = df.sort_values(
         "starts_pst"
      )

    rows = sorted_df.to_dict("records")

    for i in range(len(rows)):

        for j in range(i + 1, len(rows)):

            a = rows[i]
            b = rows[j]

            overlap = (
                a["starts_pst"] < b["ends_pst"]
                and b["starts_pst"] < a["ends_pst"]
             )

            if overlap:

                conflicts.append({
                     "session_a": a["title"],
                     "session_b": b["title"],
                     "start": max(
                        a["starts_pst"],
                        b["starts_pst"],
                     ),
                 })

    return pd.DataFrame(conflicts)
