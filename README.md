# DAIS Conference Schedule Recommender

A machine learning-powered tool to personalize your Databricks conference schedule by recommending sessions based on your interests.

## Overview

This project uses **sentence transformers** and semantic similarity to analyze conference sessions and recommend the most relevant talks for your interests. It parses the conference agenda, embeds sessions based on title, description, and metadata, and ranks them against your interest topics to create a personalized schedule.

## Features

- **Semantic Session Matching**: Uses embedding models to understand session content beyond keyword matching
- **Topic-Based Recommendations**: Recommends the top sessions for each of your interest areas
- **Conflict Resolution**: Automatically selects the best session per time slot across all topics
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
- Compute embeddings for all sessions
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
    "geospatial analytics",
    "industry use cases",
    "supply chain planning",
    "transport analytics",
    "urban planning",
    # ... add your topics
]

INCLUDE_LEVELS = [
    "Beginner",
    "Intermediate",
    "Advanced",
]

TOP_N_RECOMMENDATIONS = 15  # Sessions to recommend per topic
```

## Tips for Better Recommendations

1. **Be specific with topics**: "Machine learning on supply chain data" > "machine learning"
2. **List related interests**: Include variations of your main interests
3. **Mix levels**: Beginners should include introductory content; experienced attendees can focus on Advanced
4. **Use the web app**: Interactively refine your preferences in real-time
