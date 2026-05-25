# src/recommender.py

from collections import Counter
import math
import re

import pandas as pd


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "this", "these", "those", "you",
    "your", "we", "our", "they", "their", "or", "but", "not", "into",
    "about", "over", "under", "after", "before", "between", "during",
    "through", "across", "within", "without", "can", "could", "should",
    "would", "may", "might", "must", "do", "does", "did", "doing", "done",
    "what", "which", "who", "whom", "where", "when", "why", "how", "all",
    "any", "some", "more", "most", "many", "much", "such", "no", "nor",
    "too", "very", "than", "then", "there", "here", "also", "via", "per",
    "databricks", "data", "analytics",
    # Dataset-specific filler words that appear across many session blurbs
    "session", "sessions", "speaker", "speakers", "track", "tracks", "topic",
    "topics", "learn", "learns", "learned", "learning", "use", "uses", "used",
    "using", "explore", "explores", "exploring", "build", "builds", "building",
    "built", "overview", "introduction", "intro", "hands", "hand", "handson",
    "hands-on", "course", "courses", "workshop", "guide", "guides", "guided",
    "practical", "example", "examples", "new", "latest", "best", "practice",
    "practices", "provide", "provides", "provided", "including", "includes",
    "included", "create", "creates", "created", "help", "helps", "show",
    "shows", "shown", "understand", "understands", "understanding", "experience",
    "based", "focused", "focus", "focusing", "solution", "solutions", "platform",
    "platforms", "system", "systems", "tool", "tools", "way", "ways", "team",
    "teams", "business", "enterprise", "production", "real", "time", "step",
    "steps", "demo", "demos", "lab", "labs", "note", "notes", "prerequisite",
    "prerequisites", "summary",
}


def _tokenize_text(text: str) -> list[str]:

    if not isinstance(text, str):
        return []

    tokens = re.findall(r"[a-z0-9']+", text.lower())

    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def _build_idf_weights(texts: list[str]) -> dict[str, float]:

    document_frequency: Counter[str] = Counter()

    for text in texts:
        document_frequency.update(set(_tokenize_text(text)))

    total_documents = max(len(texts), 1)

    return {
        token: math.log((1 + total_documents) / (1 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _weighted_overlap_score(
    query_tokens: list[str],
    document_tokens: list[str],
    idf_weights: dict[str, float],
) -> float:

    if not query_tokens or not document_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    document_counts = Counter(document_tokens)

    query_weight = 0.0
    matched_weight = 0.0

    for token, query_count in query_counts.items():
        weight = idf_weights.get(token, 0.0)
        if weight <= 0:
            continue

        query_weight += query_count * weight
        matched_weight += min(query_count, document_counts.get(token, 0)) * weight

    if query_weight <= 0:
        return 0.0

    return matched_weight / query_weight


def _score_sessions_against_topic(
    df: pd.DataFrame,
    topic: str,
    idf_weights: dict[str, float],
) -> pd.Series:

    topic_tokens = _tokenize_text(topic)
    document_tokens = df["searchable_text"].map(_tokenize_text)

    return document_tokens.map(
        lambda tokens: _weighted_overlap_score(
            topic_tokens,
            tokens,
            idf_weights,
        )
    )


# ============================================================
# RANKING
# ============================================================

def rank_sessions_by_topic(
    df: pd.DataFrame,
    topics: list[str],
) -> pd.DataFrame:

    ranked = df.copy()
    idf_weights = _build_idf_weights(
        ranked["searchable_text"].tolist() + list(topics)
    )

    for topic in topics:

        scores = _score_sessions_against_topic(
            ranked,
            topic,
            idf_weights,
        )

        ranked[f"semantic_score_{topic}"] = scores

    score_cols = [f"semantic_score_{t}" for t in topics]

    ranked = ranked.loc[
        ranked[score_cols].max(axis=1) > 0
    ].copy()

    if ranked.empty:
        return df.head(0).copy()

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
    min_semantic_score: float = 0.4,
) -> pd.DataFrame:

    candidates = ranked_df.copy()

    # Keep only the highest-scoring topic assignment per session.
    if "semantic_score" in candidates.columns:
        session_group_cols = ["nid"]

        if "nid" not in candidates.columns:
            fallback_cols = ["title", "starts_pst", "ends_pst"]
            if all(col in candidates.columns for col in fallback_cols):
                session_group_cols = fallback_cols
            else:
                session_group_cols = []

        if session_group_cols:
            candidates = (
                candidates
                .sort_values("semantic_score", ascending=False)
                .groupby(session_group_cols, dropna=False)
                .head(1)
                .copy()
            )

    # Filter out low-confidence matches
    candidates = candidates.loc[
        candidates["semantic_score"] > min_semantic_score
    ].copy()

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

    if recommendations.empty:
        return recommendations.copy()

    ordered = recommendations.sort_values(
        ["day", "starts_pst", "ends_pst", "semantic_score"],
        ascending=[True, True, True, False],
    )

    selected_indices = []
    last_end_by_day: dict[str, pd.Timestamp] = {}

    for idx, row in ordered.iterrows():
        day = row["day"]
        start = row["starts_pst"]
        end = row["ends_pst"]
        score = row["semantic_score"]

        previous_end = last_end_by_day.get(day)

        # If this session overlaps the latest selected one for the day,
        # keep the session with the higher semantic score.
        if previous_end is not None and start < previous_end:
            last_selected_idx = selected_indices[-1] if selected_indices else None

            if last_selected_idx is None:
                continue

            last_selected = ordered.loc[last_selected_idx]
            if last_selected["day"] != day:
                continue

            if score > last_selected["semantic_score"]:
                selected_indices[-1] = idx
                last_end_by_day[day] = end

            continue

        selected_indices.append(idx)
        last_end_by_day[day] = end

    best_per_slot = ordered.loc[selected_indices].sort_values("starts_pst").copy()

    return best_per_slot
