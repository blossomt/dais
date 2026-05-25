# src/utils.py

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    return soup.get_text(" ", strip=True)


def flatten_categories(categories: dict) -> str:

    if not categories:
        return ""

    values = []

    for _, v in categories.items():

        if isinstance(v, list):
            values.extend(v)

    return " | ".join(values)