# src/viaf_linking.py
"""
VIAF linking for named entities extracted from Latin poetry.

This module takes the CSV produced by the NER + Index Nominum step
(entities_index_checked.csv), queries VIAF for each distinct (entity, lemma)
pair, stores the JSON responses for reproducibility, and enriches the CSV
with up to MAX_RESULTS VIAF candidates for both the raw entity and its lemma.

It also provides a simple function to compute summary statistics that can
be used in the article (e.g. percentages of rows with results, distribution
of candidates per row, etc.).
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

ENTITIES_DIR = BASE_DIR / "entities"

# Input CSV: output of the "entities + index nominum" step
INPUT_INDEX_CHECKED_CSV = ENTITIES_DIR / "entities_index_checked.csv"

# Output CSV: enriched with VIAF candidates
OUTPUT_VIAF_CSV = ENTITIES_DIR / "entities_index_checked_viaf.csv"

# Folder to store raw VIAF JSON responses (one file per query string)
VIAF_JSON_DIR = ENTITIES_DIR / "viaf_json"

# VIAF AutoSuggest endpoint (used in the article)
VIAF_AUTOSUGGEST_URL = "https://viaf.org/viaf/AutoSuggest"

# Maximum number of VIAF candidates to keep per entity / lemma
MAX_RESULTS = 5


# =========================
# Helper functions
# =========================

def sanitize_filename(name: str) -> str:
    """
    Sanitize a string so it can be safely used as a filename:
    replaces spaces with underscores and removes non alphanumeric chars.
    """
    return re.sub(r"[^A-Za-z0-9_]", "", name.replace(" ", "_"))


def ensure_directories() -> None:
    """Ensure that all required directories exist."""
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    VIAF_JSON_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# VIAF querying
# =========================

def get_viaf_details(
    name: str,
    max_results: int = MAX_RESULTS,
    sleep_interval: float = 1.0,
    error_sleep_min: float = 2.0,
    error_sleep_max: float = 60.0,
    max_retries: int = 3,
    session: requests.Session | None = None,
) -> List[Dict[str, str]]:
    """
    Query VIAF AutoSuggest for a given 'name' and return up to max_results matches.

    Each result is a dict:
        {"term": <label>, "link": <VIAF URI or empty string>}

    The full JSON response is stored in VIAF_JSON_DIR for reproducibility.

    On errors, the function retries up to max_retries times, waiting a random
    time between error_sleep_min and error_sleep_max seconds. A small
    sleep_interval is applied after successful requests to avoid overloading VIAF.
    """
    if not name:
        return []

    if session is None:
        session = requests.Session()

    params = {"query": name}
    headers = {"Accept": "application/json"}
    url = VIAF_AUTOSUGGEST_URL

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Save JSON response for this name
            json_path = VIAF_JSON_DIR / f"{sanitize_filename(name)}.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            results: List[Dict[str, str]] = []
            if "result" in data and data["result"]:
                for item in data["result"][:max_results]:
                    viaf_id = item.get("viafid", "")
                    results.append(
                        {
                            "term": item.get("term", ""),
                            "link": f"https://viaf.org/viaf/{viaf_id}" if viaf_id else "",
                        }
                    )
            # polite delay after success
            time.sleep(sleep_interval)
            return results

        except requests.RequestException as e:
            if attempt < max_retries:
                sleep_time = random.uniform(error_sleep_min, error_sleep_max)
                print(
                    f"[WARN] VIAF request error for '{name}': {e}. "
                    f"Retrying in {sleep_time:.1f} seconds ({attempt}/{max_retries})..."
                )
                time.sleep(sleep_time)
            else:
                print(
                    f"[ERROR] VIAF request failed for '{name}' after {max_retries} attempts. "
                    "Returning empty result list."
                )
                return []


def _results_to_wide_columns(
    prefix: str,
    results: List[Dict[str, str]],
    max_results: int = MAX_RESULTS,
) -> Dict[str, str]:
    """
    Convert a list of VIAF results into a 'wide' dict of columns:

    prefix_viaf_label_1, prefix_viaf_uri_1, ..., prefix_viaf_label_max, prefix_viaf_uri_max
    """
    cols: Dict[str, str] = {}
    for i in range(1, max_results + 1):
        if i <= len(results):
            cols[f"{prefix}_viaf_label_{i}"] = results[i - 1].get("term", "")
            cols[f"{prefix}_viaf_uri_{i}"] = results[i - 1].get("link", "")
        else:
            cols[f"{prefix}_viaf_label_{i}"] = ""
            cols[f"{prefix}_viaf_uri_{i}"] = ""
    return cols


# =========================
# Main enrichment pipeline
# =========================

def enrich_entities_with_viaf(
    input_csv: Path = INPUT_INDEX_CHECKED_CSV,
    output_csv: Path = OUTPUT_VIAF_CSV,
    max_rows: int | None = None,
    only_missing: bool = False,
) -> pd.DataFrame:
    """
    Enrich the entities CSV with VIAF candidates for both 'entity' and 'lemma'.

    Steps:
    1. Load the CSV produced by the NER + Index Nominum step.
    2. Normalise 'entity' and 'lemma' into 'entity_clean' and 'lemma_clean'.
    3. Build a key 'pair_key' = entity_clean || lemma_clean.
    4. For each *distinct* pair_key:
       - if only_missing=True and the first row of that group already has VIAF
         candidates (non-empty in *_viaf_label_1), reuse them and propagate;
       - otherwise, query VIAF once for entity and once for lemma.
    5. Expand results into wide columns for each row and save to output_csv.

    Returns the enriched DataFrame.
    """
    ensure_directories()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        print(f"[INFO] Processing only first {max_rows} rows (for testing).")

    # Normalised forms for grouping and deduplication
    df["entity_clean"] = df["entity"].astype(str).str.strip().str.lower()
    df["lemma_clean"] = df["lemma"].astype(str).str.strip().str.lower()
    df["pair_key"] = df["entity_clean"] + "||" + df["lemma_clean"]

    # Prepare VIAF columns if they do not exist
    viaf_cols: List[str] = []
    for prefix in ("entity", "lemma"):
        for i in range(1, MAX_RESULTS + 1):
            viaf_cols.append(f"{prefix}_viaf_label_{i}")
            viaf_cols.append(f"{prefix}_viaf_uri_{i}")

    for col in viaf_cols:
        if col not in df.columns:
            df[col] = ""

    # Group by pair_key so that we only query VIAF once per distinct (entity, lemma)
    session = requests.Session()
    pair_results: Dict[str, Dict[str, str]] = {}

    grouped = df.groupby("pair_key", sort=False)

    for pair_key, group in grouped:
        # Skip completely empty keys (unlikely, but safe)
        if not pair_key.strip("|"):
            continue

        # Check if we can reuse existing results (only_missing=True)
        existing_row = group.iloc[0]
        reuse = False
        if only_missing:
            has_entity_viaf = any(
                str(existing_row.get(f"entity_viaf_label_{i}", "")).strip()
                for i in range(1, MAX_RESULTS + 1)
            )
            has_lemma_viaf = any(
                str(existing_row.get(f"lemma_viaf_label_{i}", "")).strip()
                for i in range(1, MAX_RESULTS + 1)
            )
            reuse = has_entity_viaf or has_lemma_viaf

        if reuse:
            cols_dict = {
                col: existing_row[col]
                for col in viaf_cols
                if col in existing_row.index
            }
            pair_results[pair_key] = cols_dict
            print(f"[INFO] Reusing existing VIAF results for pair_key={pair_key!r}")
            continue

        # Otherwise, perform fresh queries for this pair
        raw_entity = str(group.iloc[0]["entity"]).strip()
        raw_lemma = str(group.iloc[0]["lemma"]).strip()

        print(f"[INFO] VIAF lookup for pair_key={pair_key!r} (entity='{raw_entity}', lemma='{raw_lemma}')")

        entity_results = get_viaf_details(raw_entity, max_results=MAX_RESULTS, session=session)
        lemma_results = get_viaf_details(raw_lemma, max_results=MAX_RESULTS, session=session)

        # If VIAF returns no results, we mark label_1 / uri_1 as "No results"
        entity_cols = _results_to_wide_columns("entity", entity_results)
        lemma_cols = _results_to_wide_columns("lemma", lemma_results)

        if not entity_results:
            entity_cols["entity_viaf_label_1"] = "No results"
            entity_cols["entity_viaf_uri_1"] = "No results"

        if not lemma_results:
            lemma_cols["lemma_viaf_label_1"] = "No results"
            lemma_cols["lemma_viaf_uri_1"] = "No results"

        cols_dict = {**entity_cols, **lemma_cols}
        pair_results[pair_key] = cols_dict

    # Apply results to all rows
    for pair_key, cols_dict in pair_results.items():
        mask = df["pair_key"] == pair_key
        for col, val in cols_dict.items():
            df.loc[mask, col] = val

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[OK] VIAF-enriched CSV saved at: {output_csv}")
    return df


# =========================
# Statistics and summaries
# =========================

def compute_viaf_summary(
    viaf_csv: Path = OUTPUT_VIAF_CSV,
    max_results: int = MAX_RESULTS,
) -> pd.DataFrame:
    """
    Compute summary statistics for VIAF results:

    For both 'entity' and 'lemma':
      - classify each row as:
        * 'error'        → all cells empty/nan (should not happen if pipeline ran)
        * 'no_results'   → first label cell == 'No results'
        * 'ok'           → at least one (label, uri) pair
      - count how many rows fall in each category
      - compute distribution of the number of valid (label, uri) pairs per row.

    Returns a DataFrame with one row per ('target', 'status' or 'pair_count'),
    and also prints a human-readable summary (similar to your notebook).
    """
    if not viaf_csv.exists():
        raise FileNotFoundError(f"VIAF CSV not found: {viaf_csv}")

    df = pd.read_csv(viaf_csv)
    total_rows = len(df)

    def _status_and_pairs(prefix: str) -> Tuple[pd.Series, pd.Series]:
        label_cols = [f"{prefix}_viaf_label_{i}" for i in range(1, max_results + 1)]
        uri_cols = [f"{prefix}_viaf_uri_{i}" for i in range(1, max_results + 1)]

        def is_empty(cell: object) -> bool:
            return str(cell).strip().lower() in ("", "nan")

        def is_no_results(cell: object) -> bool:
            return str(cell).strip().lower() == "no results"

        statuses: List[str] = []
        pair_counts: List[int] = []

        for _, row in df.iterrows():
            cells = [row[col] for col in label_cols + uri_cols]
            if all(is_empty(c) for c in cells):
                statuses.append("error")
                pair_counts.append(0)
                continue

            first_label = row[label_cols[0]]
            if is_no_results(first_label):
                statuses.append("no_results")
                pair_counts.append(0)
                continue

            count = 0
            for l_col, u_col in zip(label_cols, uri_cols):
                l_val = row[l_col]
                u_val = row[u_col]
                if (
                    not is_empty(l_val)
                    and not is_empty(u_val)
                    and not is_no_results(l_val)
                    and not is_no_results(u_val)
                ):
                    count += 1

            statuses.append("ok")
            pair_counts.append(count)

        return pd.Series(statuses, index=df.index), pd.Series(pair_counts, index=df.index)

    summary_rows = []

    for prefix in ("entity", "lemma"):
        status_series, pair_series = _status_and_pairs(prefix)

        # category counts
        status_counts = status_series.value_counts()
        print(f"\n=== {prefix.upper()} VIAF ===")
        print(f"Total rows: {total_rows}")
        for status in ["error", "no_results", "ok"]:
            count = int(status_counts.get(status, 0))
            pct = 100.0 * count / total_rows if total_rows else 0.0
            print(f"{status}: {count} rows = {pct:.2f}%")
            summary_rows.append(
                {
                    "target": prefix,
                    "metric": "status",
                    "category": status,
                    "count": count,
                    "percentage": pct,
                }
            )

        # distribution of valid pairs (only where status == "ok")
        ok_mask = status_series == "ok"
        distribution = pair_series[ok_mask].value_counts().sort_index()
        print("\nDistribution of valid VIAF pairs per row:")
        for k in range(1, max_results + 1):
            count = int(distribution.get(k, 0))
            pct = 100.0 * count / total_rows if total_rows else 0.0
            print(f"{k} results -> {count} rows = {pct:.2f}%")
            summary_rows.append(
                {
                    "target": prefix,
                    "metric": "valid_pairs",
                    "category": str(k),
                    "count": count,
                    "percentage": pct,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


# =========================
# Main entry point
# =========================

def main():
    # 1) Enrich CSV with VIAF results
    df_viaf = enrich_entities_with_viaf(
        input_csv=INPUT_INDEX_CHECKED_CSV,
        output_csv=OUTPUT_VIAF_CSV,
        max_rows=None,      # set to a small number for testing
        only_missing=False, # set True if you re-run on a partially populated CSV
    )

    # 2) Compute and display summary statistics
    compute_viaf_summary(OUTPUT_VIAF_CSV, max_results=MAX_RESULTS)


if __name__ == "__main__":
    main()
