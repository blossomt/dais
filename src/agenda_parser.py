# src/agenda_parser.py

import json
from pathlib import Path

import pandas as pd

from src.utils import (
    clean_html,
    flatten_categories,
)


# ============================================================
# PARSE AGENDA JSON
# ============================================================

def parse_agenda(
    path: str | Path,
) -> pd.DataFrame:

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    sessions = raw["pageProps"]["agenda"]["sessions"]

    rows = []

    for s in sessions:

        speakers = ", ".join([
            speaker.get("name", "")
            for speaker in s.get("speakers", [])
        ])

        companies = ", ".join(sorted(set([
            speaker.get("company", "")
            for speaker in s.get("speakers", [])
            if speaker.get("company")
        ])))

        category_blob = flatten_categories(
            s.get("categories", {})
        )

        body_clean = clean_html(
            s.get("body", "")
        )

        searchable_text = " ".join([
            s.get("title", ""),
            body_clean,
            category_blob,
            speakers,
            companies,
        ])

        rows.append({
            "nid": s.get("nid"),
            "title": s.get("title"),
            "day": s.get("day"),
            "starts_pst": pd.to_datetime(
                s.get("starts_pst")
            ),
            "ends_pst": pd.to_datetime(
                s.get("ends_pst")
            ),
            "duration": int(
                s.get("duration") or 0
            ),
            "type": ", ".join(
                s.get("categories", {})
                .get("type", [])
            ),
            "level": ", ".join(
                s.get("categories", {})
                .get("level", [])
            ),
            "track": ", ".join(
                s.get("categories", {})
                .get("track", [])
            ),
            "categories": category_blob,
            "speakers": speakers,
            "companies": companies,
            "description": body_clean,
            "searchable_text": searchable_text,
            "url": (
                "https://www.databricks.com/dataaisummit"
                f"{s.get('path')}"
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# FILTERS
# ============================================================

def apply_filters(
    df: pd.DataFrame,
    include_levels: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> pd.DataFrame:

    filtered = df.copy()

    if include_levels:

        filtered = filtered[
            filtered["level"].apply(
                lambda x: any(
                    level in x
                    for level in include_levels
                )
            )
        ]

    if exclude_types:

        filtered = filtered[
            ~filtered["type"].apply(
                lambda x: any(
                    t in x
                    for t in exclude_types
                )
            )
        ]

    return filtered.reset_index(drop=True)