# DAIS Conference Schedule Recommender

A machine learning-powered tool to personalize your Databricks conference schedule by recommending sessions based on your interests.

## Overview

This project parses the conference agenda and ranks sessions with a **rule-based NLP recommender**. It tokenizes session text and interest topics, removes stop words, weights rarer terms with IDF, and scores each session by weighted keyword overlap. The pipeline then filters and deconflicts results to produce a practical personal schedule.

## Features

- **Lexical Relevance Scoring**: Uses tokenization + stop-word removal + IDF-weighted overlap scoring
- **Best Topic Per Session**: Keeps each session only once using its highest-scoring topic match
- **Score-Based Deconflicting**: Resolves time overlaps by keeping the higher `semantic_score` session
- **Interactive Visualizations**: Generated Plotly timelines showing your recommended schedule
- **Flexible Filtering**: Filter sessions by level (Beginner/Intermediate/Advanced) and type
- **CSV Exports**: Export recommended sessions and your best schedule to CSV

## Installation

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd dais
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Recommendations (Batch Mode)

Run the main script to generate a personalized schedule:

```bash
python main.py
```

This will:
- Parse the agenda from `data/agenda.json`
- Compute lexical relevance for all sessions
- Rank sessions by your interest topics (defined in `main.py`)
- Generate visualizations and export results to `output/`

### Interactive Web App

Launch the Streamlit app for an interactive experience:

```bash
streamlit run app/src/main.py
```

The web app allows you to:
- Dynamically update your interest profile
- View and filter recommendations in real-time
- Download personalized schedules

## Configuration

Edit `main.py` to customize:

```python
INTEREST_TOPICS = [
    "geospatial spatial sql",
    "deploying lakebase databricks apps in production",
    "secure data sharing with unity catalog"
    # ... add your topics
]
```

## Recommendation Model

The recommender in `src/recommender.py` follows this flow:

1. **Tokenize + normalize text**
    - Session `searchable_text` and input topics are lowercased and tokenized.
    - Common English words and dataset-specific filler words are removed using `STOP_WORDS`.

2. **Compute IDF weights**
    - IDF is computed over session texts + topic texts:
    - `IDF(term) = log((1 + total_docs) / (1 + doc_frequency)) + 1.0`

3. **Score topic-session relevance**
    - Each topic is scored against each session using weighted overlap:
    - `semantic_score = Σ(matched_terms × IDF) / Σ(query_terms × IDF)`

4. **Assign best topic per session**
    - For each session, the pipeline keeps the single highest-scoring topic assignment.

5. **Filter + cap recommendations**
    - Applies `min_semantic_score` (default `0.4`).
    - Keeps top `top_per_topic` sessions per topic.

6. **Build final schedule with overlap handling**
    - Sessions are sorted chronologically by day.
    - When two sessions overlap, the one with higher `semantic_score` is kept.

### Runtime Controls (Streamlit)
- `Top recommended sessions` (`top_per_topic` / display cap)
- `Minimum semantic score` (`min_semantic_score`)

These controls are exposed in the app sidebar and passed into `build_recommendations()`.

## Tips for Better Recommendations

1. **Be specific with topics**: "Machine learning on supply chain data" > "machine learning"
2. **List related interests**: Include variations of your main interests
3. **Mix levels**: Beginners should include introductory content; experienced attendees can focus on Advanced
4. **Use the web app**: Interactively refine your preferences in real-time
