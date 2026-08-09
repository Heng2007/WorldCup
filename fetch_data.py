"""Fetch current-cycle World Cup matches from football-data.org.

Run this only to refresh ``2026WorldCupLatest.csv``. The refreshed CSV is
committed to the repo, so the predictor itself runs without an API key.

Requires ``FOOTBALL_DATA_API_KEY`` in a local ``.env`` file — see ``.env.example``.
Register for a free key at https://www.football-data.org/client/register

Usage:
    python fetch_data.py
"""

import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.football-data.org/v4/competitions/WC/matches?season=2026"
OUTPUT_CSV = "2026WorldCupLatest.csv"


def fetch_matches():
    """Return the API's match list as a DataFrame.

    Raises:
        KeyError: if FOOTBALL_DATA_API_KEY is not set.
        requests.HTTPError: on any non-2xx response, so an auth failure surfaces
            here rather than as a confusing KeyError further down.
    """
    headers = {"X-Auth-Token": os.environ["FOOTBALL_DATA_API_KEY"]}
    response = requests.get(API_URL, headers=headers, timeout=10)
    response.raise_for_status()
    return pd.DataFrame(response.json()["matches"])


if __name__ == "__main__":
    matches = fetch_matches()
    matches.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(matches)} matches to {OUTPUT_CSV}")
