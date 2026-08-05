"""
Enrich app_details.csv with Google Play Store metadata (description, score,
installs, etc.) scraped via the `google-play-scraper` package.

Usage:
    pip install google-play-scraper --break-system-packages
    python3 scrape_play_descriptions.py

The script is resumable: it periodically writes progress to OUTPUT_CSV, and
if that file already exists on the next run, already-scraped package names
are skipped, so an interrupted run can just be restarted.
"""

import csv
import json
import random
import sys
import time

import pandas as pd
from google_play_scraper import app as gp_app
from google_play_scraper.exceptions import NotFoundError

INPUT_CSV = "../rico-data/data/play_store_metadata/app_details.csv"
OUTPUT_CSV = "../rico-data/data/play_store_metadata/app_details_enriched.csv"
PACKAGE_COLUMN = "App Package Name"

# Fields returned by google_play_scraper.app(); each becomes a "gp_<field>"
# column. Non-scalar fields (lists/dicts) are JSON-encoded for CSV safety.
GP_FIELDS = [
    "title", "description", "descriptionHTML", "summary", "installs",
    "minInstalls", "realInstalls", "score", "ratings", "reviews",
    "histogram", "price", "free", "currency", "sale", "saleTime",
    "originalPrice", "saleText", "offersIAP", "inAppProductPrice",
    "developer", "developerId", "developerEmail", "developerWebsite",
    "developerAddress", "privacyPolicy", "genre", "genreId", "categories",
    "icon", "headerImage", "screenshots", "video", "videoImage",
    "contentRating", "contentRatingDescription", "adSupported",
    "containsAds", "released", "lastUpdatedOn", "updated", "version",
    "comments",
]
GP_COLUMNS = [f"gp_{f}" for f in GP_FIELDS] + ["gp_status"]

MIN_DELAY = 1.5   # seconds, floor of the randomized delay between requests
MAX_DELAY = 3.0   # seconds, ceiling of the randomized delay between requests
MAX_RETRIES = 3   # retries for transient errors (e.g. throttling) per app
BACKOFF_BASE = 30  # seconds; retry delay grows as BACKOFF_BASE * 2**attempt
SAVE_EVERY = 20    # rows between progress checkpoints


def serialize(value):
    """CSV cells must be scalars; JSON-encode lists/dicts, pass through the rest."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def fetch_one(package_name):
    """
    Return a dict of gp_* columns for a single package name.
    Retries transient errors with backoff; NotFoundError is not retried
    since it means the app genuinely isn't on the store (or was removed).
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = gp_app(package_name)
            row = {f"gp_{f}": serialize(result.get(f)) for f in GP_FIELDS}
            row["gp_status"] = "ok"
            return row
        except NotFoundError:
            row = {f"gp_{f}": None for f in GP_FIELDS}
            row["gp_status"] = "not_found"
            return row
        except Exception as e:  # network hiccups, throttling (403/503), etc.
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (2 ** attempt)
                print(f"    error on {package_name} ({e!r}); "
                      f"retrying in {wait}s [{attempt + 1}/{MAX_RETRIES}]",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                row = {f"gp_{f}": None for f in GP_FIELDS}
                row["gp_status"] = f"error: {e!r}"
                return row


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)

    for col in GP_COLUMNS:
        if col not in df.columns:
            df[col] = None

    try:
        done = pd.read_csv(OUTPUT_CSV, dtype=str)
        scraped_mask = done.set_index(PACKAGE_COLUMN)["gp_status"].notna()
        already_done = set(scraped_mask[scraped_mask].index)
        df = df.set_index(PACKAGE_COLUMN)
        df.update(done.set_index(PACKAGE_COLUMN))
        df = df.reset_index()
        print(f"Resuming: {len(already_done)} apps already scraped, skipping those.")
    except FileNotFoundError:
        already_done = set()

    todo_idx = df.index[~df[PACKAGE_COLUMN].isin(already_done)]
    total = len(todo_idx)
    print(f"{total} apps left to scrape out of {len(df)} total.")

    for progress, idx in enumerate(todo_idx, start=1):
        package_name = df.at[idx, PACKAGE_COLUMN]
        print(f"[{progress}/{total}] {package_name}")

        row = fetch_one(package_name)
        for col, value in row.items():
            df.at[idx, col] = value

        if progress % SAVE_EVERY == 0 or progress == total:
            df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"    checkpoint saved ({progress}/{total})")

        if progress < total:
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Done. Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()