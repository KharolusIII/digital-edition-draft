# src/pleiades_linking.py
"""
Pleiades enrichment for place-related entities (LOC, and potentially others).

This module:
- reads the entities CSV (already enriched with VIAF and index checks),
- loads Pleiades CSV data (names + geometries),
- matches entity and lemma strings against attested/romanized forms,
- adds Pleiades URIs and geometries to each row (up to MAX_RESULTS),
- computes simple statistics about Pleiades coverage.

All file paths are relative and can be adapted in the configuration section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

ENTITIES_DIR = BASE_DIR / "entities"
PLEIADES_DIR = BASE_DIR / "Pleiades_CSVs"

# Input: output of the VIAF pipeline
ENTITIES_VIAF_CSV = ENTITIES_DIR / "entities_index_checked_viaf.csv"

# Output: VIAF + Pleiades enrichment
ENTITIES_PLEIADES_CSV = ENTITIES_DIR / "entities_index_checked_viaf_pleiades.csv"

# Statistics outputs
PLEIADES_STATS_GLOBAL_CSV = ENTITIES_DIR / "pleiades_stats_global.csv"
PLEIADES_STATS_BY_TYPE_CSV = ENTITIES_DIR / "pleiades_stats_by_type.csv"

# Pleiades CSVs (downloaded from https://atlantides.org/downloads/pleiades/gis/)
PLEIADES_NAMES_CSV = PLEIADES_DIR / "names.csv"
PLEIADES_POINTS_CSV = PLEIADES_DIR / "location_points.csv"
PLEIADES_POLYGONS_CSV = PLEIADES_DIR / "location_polygons.csv"

# Maximum number of Pleiades matches to keep per term (entity / lemma)
MAX_RESULTS = 5

# For quick testing: limit number of rows from the entities CSV (0 means "all")
MAX_ROWS = 0


# =========================
# 1) Load Pleiades tables
# =========================

def load_pleiades_names(
    names_path: Path = PLEIADES_NAMES_CSV,
    points_path: Path = PLEIADES_POINTS_CSV,
    polygons_path: Path = PLEIADES_POLYGONS_CSV,
) -> pd.DataFrame:
    """
    Load Pleiades 'names' CSV and merge point/polygon geometries.

    Returns a DataFrame with columns:
    - place_id
    - attested_form
    - romanized_form_1, romanized_form_2, romanized_form_3
    - uri
    - geometry_point
    - geometry_polygon
    """
    if not names_path.exists():
        raise FileNotFoundError(f"Pleiades names CSV not found: {names_path}")

    df_names = pd.read_csv(
        names_path,
        usecols=[
            "place_id",
            "attested_form",
            "romanized_form_1",
            "romanized_form_2",
            "romanized_form_3",
            "uri",
        ],
    )

    # Merge point geometries
    if points_path.exists():
        df_points = (
            pd.read_csv(points_path, usecols=["place_id", "geometry_wkt"])
            .rename(columns={"geometry_wkt": "geometry_point"})
        )
        df_names = df_names.merge(df_points, on="place_id", how="left")
    else:
        df_names["geometry_point"] = None

    # Merge polygon geometries
    if polygons_path.exists():
        df_polygons = (
            pd.read_csv(polygons_path, usecols=["place_id", "geometry_wkt"])
            .rename(columns={"geometry_wkt": "geometry_polygon"})
        )
        df_names = df_names.merge(df_polygons, on="place_id", how="left")
    else:
        df_names["geometry_polygon"] = None

    # Normalize string columns used for matching
    for col in ["attested_form", "romanized_form_1", "romanized_form_2", "romanized_form_3"]:
        df_names[col] = df_names[col].astype(str).str.strip().str.lower()

    return df_names


# =========================
# 2) Build match dictionary
# =========================

def build_pleiades_match_dict(
    df_base: pd.DataFrame,
    df_names: pd.DataFrame,
    max_results: int = MAX_RESULTS,
) -> Dict[str, pd.DataFrame]:
    """
    For each unique term (entity or lemma) in df_base, find up to max_results
    matching rows in df_names where the term matches one of:
    - attested_form
    - romanized_form_1
    - romanized_form_2
    - romanized_form_3

    Returns a dict:
        term (lowercased string) -> DataFrame of matches (up to max_results rows).
    """
    # Collect unique terms from entity and lemma
    unique_terms = pd.Series(
        pd.concat([df_base["entity"].dropna(), df_base["lemma"].dropna()])
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )
    unique_terms = unique_terms[unique_terms != ""]  # drop empty strings

    match_dict: Dict[str, pd.DataFrame] = {}

    print(f"[INFO] Building Pleiades match dictionary for {len(unique_terms)} unique terms...")
    for term in unique_terms:
        mask = df_names[["attested_form", "romanized_form_1", "romanized_form_2", "romanized_form_3"]].eq(term).any(axis=1)
        matches = df_names.loc[mask].head(max_results).copy()
        match_dict[term] = matches.reset_index(drop=True)
        if not matches.empty:
            print(f"  Term '{term}' -> {len(matches)} match(es).")

    return match_dict


# =========================
# 3) Enrich entities with Pleiades
# =========================

def _ensure_pleiades_columns(df: pd.DataFrame, max_results: int) -> pd.DataFrame:
    """
    Ensure the DataFrame has empty Pleiades columns for entity and lemma.
    """
    for i in range(1, max_results + 1):
        for prefix in ("entity", "lemma"):
            for suffix in ("uri", "point", "polygon"):
                col_name = f"{prefix}_pleiades_{suffix}_{i}"
                if col_name not in df.columns:
                    df[col_name] = ""
    return df


def enrich_entities_with_pleiades(
    df_base: pd.DataFrame,
    df_names: pd.DataFrame,
    max_results: int = MAX_RESULTS,
) -> pd.DataFrame:
    """
    Enrich the entities DataFrame with Pleiades URIs and geometries.

    For each row and for both 'entity' and 'lemma':
    - look up the lowercased text in the match dictionary,
    - copy up to max_results matches into:
        entity_pleiades_uri_i, entity_pleiades_point_i, entity_pleiades_polygon_i
        lemma_pleiades_uri_i,  lemma_pleiades_point_i,  lemma_pleiades_polygon_i
    """
    df_base = _ensure_pleiades_columns(df_base, max_results)
    match_dict = build_pleiades_match_dict(df_base, df_names, max_results=max_results)

    print("[INFO] Enriching entities with Pleiades data...")
    for idx, row in df_base.iterrows():
        entity_term = str(row.get("entity", "")).strip().lower()
        lemma_term = str(row.get("lemma", "")).strip().lower()

        for prefix, term in (("entity", entity_term), ("lemma", lemma_term)):
            if not term:
                continue

            matches = match_dict.get(term)
            if matches is None or matches.empty:
                continue

            n_matches = min(max_results, len(matches))
            for i in range(n_matches):
                df_base.at[idx, f"{prefix}_pleiades_uri_{i+1}"] = matches.at[i, "uri"]
                df_base.at[idx, f"{prefix}_pleiades_point_{i+1}"] = matches.at[i, "geometry_point"]
                df_base.at[idx, f"{prefix}_pleiades_polygon_{i+1}"] = matches.at[i, "geometry_polygon"]

    return df_base


# =========================
# 4) Counts and statistics
# =========================

def _is_valid_cell(cell) -> bool:
    """
    Decide whether a cell with a Pleiades URI should be considered a valid result.
    """
    value = str(cell).strip().lower()
    return value not in ("", "nan", "no results")


def add_pleiades_counts(df: pd.DataFrame, max_results: int = MAX_RESULTS) -> pd.DataFrame:
    """
    Add two columns:
    - entity_pleiades_count: number of valid Pleiades URIs for entity
    - lemma_pleiades_count:  number of valid Pleiades URIs for lemma
    """
    def count_valid(row, prefix: str) -> int:
        cols = [f"{prefix}_pleiades_uri_{i}" for i in range(1, max_results + 1)]
        return sum(_is_valid_cell(row.get(col, "")) for col in cols)

    df["entity_pleiades_count"] = df.apply(lambda r: count_valid(r, "entity"), axis=1)
    df["lemma_pleiades_count"] = df.apply(lambda r: count_valid(r, "lemma"), axis=1)
    return df


def build_pleiades_global_stats(df: pd.DataFrame, max_results: int = MAX_RESULTS) -> pd.DataFrame:
    """
    Build a global statistics table (no breakdown by entity_type):

    For 'entity' and 'lemma' separately:
      - how many rows have 0, 1, 2, ..., max_results Pleiades matches
      - percentage of total rows for each count.
    """
    rows = []
    total = len(df) if len(df) > 0 else 1  # avoid division by zero

    for target, count_col in (("entity", "entity_pleiades_count"), ("lemma", "lemma_pleiades_count")):
        distribution = df[count_col].value_counts().sort_index()
        for n_results, n_rows in distribution.items():
            pct = (n_rows / total) * 100.0
            rows.append(
                {
                    "target": target,
                    "n_results": int(n_results),
                    "rows": int(n_rows),
                    "pct_rows": pct,
                }
            )

    stats_df = pd.DataFrame(rows)
    return stats_df


def build_pleiades_stats_by_type(
    df: pd.DataFrame,
    entity_type_col: str = "entity_type",
    max_results: int = MAX_RESULTS,
) -> pd.DataFrame:
    """
    Build a statistics table broken down by entity_type:

    For each target ('entity'/'lemma') and each entity_type:
      - distribution of 0, 1, 2, ..., max_results Pleiades matches
      - percentages over the total for that entity_type.
    """
    if entity_type_col not in df.columns:
        raise KeyError(f"Column '{entity_type_col}' not found in DataFrame.")

    rows = []

    for target, count_col in (("entity", "entity_pleiades_count"), ("lemma", "lemma_pleiades_count")):
        for etype, group in df.groupby(entity_type_col):
            total = len(group) if len(group) > 0 else 1
            distribution = group[count_col].value_counts().sort_index()
            for n_results, n_rows in distribution.items():
                pct = (n_rows / total) * 100.0
                rows.append(
                    {
                        "target": target,
                        "entity_type": etype,
                        "n_results": int(n_results),
                        "rows": int(n_rows),
                        "pct_rows": pct,
                    }
                )

    stats_df = pd.DataFrame(rows)
    return stats_df


# =========================
# Main entry point
# =========================

def main():
    # 1) Load entities CSV
    if not ENTITIES_VIAF_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {ENTITIES_VIAF_CSV}")

    df_base = pd.read_csv(ENTITIES_VIAF_CSV)

    if MAX_ROWS and MAX_ROWS > 0:
        df_base = df_base.head(MAX_ROWS)
        print(f"[INFO] Limiting to first {MAX_ROWS} rows for testing.")

    if df_base.empty:
        print("[WARN] No rows in VIAF CSV; skipping Pleiades enrichment.")
        return

    # 2) Load Pleiades names + geometries
    if not PLEIADES_NAMES_CSV.exists():
        print(f"[WARN] Pleiades names CSV not found: {PLEIADES_NAMES_CSV}. Skipping.")
        return
    df_names = load_pleiades_names()

    # 3) Enrich with Pleiades
    df_enriched = enrich_entities_with_pleiades(df_base, df_names, max_results=MAX_RESULTS)

    # 4) Add counts of Pleiades matches
    df_enriched = add_pleiades_counts(df_enriched, max_results=MAX_RESULTS)

    # 5) Save enriched CSV
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(ENTITIES_PLEIADES_CSV, index=False, encoding="utf-8")
    print(f"[OK] Pleiades-enriched CSV saved at: {ENTITIES_PLEIADES_CSV}")

    # 6) Global stats
    global_stats = build_pleiades_global_stats(df_enriched, max_results=MAX_RESULTS)
    global_stats.to_csv(PLEIADES_STATS_GLOBAL_CSV, index=False, encoding="utf-8")
    print(f"[OK] Global Pleiades stats saved at: {PLEIADES_STATS_GLOBAL_CSV}")

    # 7) Stats by entity_type (if column exists)
    if "entity_type" in df_enriched.columns:
        type_stats = build_pleiades_stats_by_type(df_enriched, entity_type_col="entity_type", max_results=MAX_RESULTS)
        type_stats.to_csv(PLEIADES_STATS_BY_TYPE_CSV, index=False, encoding="utf-8")
        print(f"[OK] Pleiades stats by entity_type saved at: {PLEIADES_STATS_BY_TYPE_CSV}")
    else:
        print("[WARN] 'entity_type' column not found; skipping stats by type.")

    # Short printed summary
    print("\n[SUMMARY] Global Pleiades results distribution:")
    print(global_stats)


if __name__ == "__main__":
    main()
