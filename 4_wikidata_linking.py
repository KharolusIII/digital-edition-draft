# wikidata_linking.py
"""
Wikidata linking pipeline for Latin named entities (persons and places).

This module:
- Reads an entities CSV file (with at least the columns: filename, entity, lemma, entity_type).
- For each unique (entity, lemma, entity_type) triple, queries Wikidata using wbsearchentities.
- For each candidate item, retrieves birth/death years, instance-of types, and English aliases.
- Applies a simple heuristic score based on:
    - label match with the search term,
    - compatibility of types with the expected category (PERSON / LOC / NORP),
    - temporal filter (entities up to a given DATE_THRESHOLD, e.g. 100 CE).
- Produces up to MAX_RESULTS candidates for:
    - entity (strict filter: type + date),
    - entity (lax filter: type only),
    - lemma (strict),
    - lemma (lax),
  separated into "with date" and "without date" lists.
- Stores raw JSON search responses in a folder for reproducibility.
- Writes a wide CSV with one row per original entity and many Wikidata columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import random
import re
import time

import pandas as pd
import requests


# Reuse HTTP connections across calls
SESSION = requests.Session()

# Cache search results in memory to avoid repeated Wikidata calls
# key = (name, max_results, date_threshold, expected_category, strict)
SEARCH_RESULT_CACHE: Dict[
    Tuple[str, int, int, str | None, bool],
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]],
] = {}

# Cache metadata per Wikidata entity id (birth/death years, types, aliases)
ENTITY_META_CACHE: Dict[str, Dict[str, Any]] = {}


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

# Input CSV with entities (output of the index/VIAF/Pleiades steps)
INPUT_CSV = BASE_DIR / "entities" / "entities_index_checked_viaf_pleiades.csv"

# Output CSV enriched with Wikidata candidates
OUTPUT_CSV = BASE_DIR / "entities" / "entities_enriched_wikidata.csv"

# Directory for raw JSON responses from Wikidata (wbsearchentities)
WIKIDATA_JSON_DIR = BASE_DIR / "entities" / "wikidata_json"

# Maximum number of candidates to keep per list (with_date / without_date)
MAX_RESULTS = 5

# Date threshold: only keep entities whose birth/death year is <= this value (e.g. 100 CE)
DATE_THRESHOLD = 100

# Optional: limit the number of rows to process (None → process all)
MAX_ROWS_TO_PROCESS: int | None = None

# Expected instance-of types (P31) for each coarse entity category
EXPECTED_TYPES: Dict[str, List[str]] = {
    "PERSON": [
        "Q5",         # human
        "Q11688446",  # Roman deity
        "Q178885",    # goddess
        "Q22989102",  # Greek
        "Q9577126",   # mythological creature
        "Q7236901",   # mythological king
        "Q22988604",  # mythological Greek character
        "Q8881642",   # mythological character
    ],
    "LOC": [
        "Q49156040",  # ancient people and places
        "Q618123",    # geographic location
        "Q3024240",   # former country
        "Q11772",     # Ancient Greece
        "Q1747689",   # Ancient Rome
        "Q182547",    # Roman province
        "Q82794",     # historical country
        "Q27916659",  # former place
        "Q15661340",  # ancient city
        "Q28171280",  # ancient place
        "Q6364769",   # mythological place
    ],
    "NORP": [
        "Q41710",     # ethnic group
        "Q4204501",   # ancient people
        "Q49156040",  # ancient people and places
    ],
}

# Ordered fields stored for every Wikidata candidate
WIKIDATA_CANDIDATE_FIELDS = [
    "label",
    "uri",
    "birth_year",
    "death_year",
    "types",
    "score",
    "aliases",
]


# =========================
# Low-level helpers
# =========================

def sanitize_filename(name: str) -> str:
    """
    Convert an arbitrary string into something that can be safely used as a filename:
    replace spaces with underscores and remove non-alphanumeric characters.
    """
    return re.sub(r"[^A-Za-z0-9_]", "", name.replace(" ", "_"))


def get_wikidata_date(entity_id: str, property_id: str) -> int | None:
    """
    Query the wbgetentities endpoint to extract a year from a date claim
    (birth: P569, death: P570). Returns the year as an integer or None.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "claims",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        entity = data.get("entities", {}).get(entity_id, {})
        claims = entity.get("claims", {}).get(property_id, [])
        if not claims:
            return None
        mainsnak = claims[0].get("mainsnak", {})
        datavalue = mainsnak.get("datavalue", {})
        value = datavalue.get("value", {})
        time_value = value.get("time", "")
        # Example format: '+0044-03-15T00:00:00Z'
        if time_value and len(time_value) >= 5:
            # strip leading '+' and split by '-'
            year_str = time_value.lstrip("+").split("-")[0]
            return int(year_str)
    except Exception:
        return None
    return None


def get_entity_types(entity_id: str) -> List[str]:
    """
    Get the list of instance-of (P31) types for a given Wikidata entity id.
    Returns a list of QIDs as strings.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "claims",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        claims = (
            data.get("entities", {})
            .get(entity_id, {})
            .get("claims", {})
            .get("P31", [])
        )
        types = []
        for claim in claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            ent_id = value.get("id")
            if ent_id:
                types.append(ent_id)
        return types
    except Exception:
        return []


def get_entity_aliases(entity_id: str, language: str = "en") -> List[str]:
    """
    Get aliases for an entity in a given language (defaults to English).
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "aliases",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        aliases_data = (
            data.get("entities", {})
            .get(entity_id, {})
            .get("aliases", {})
            .get(language, [])
        )
        return [entry.get("value", "") for entry in aliases_data]
    except Exception:
        return []


def _extract_year_from_time_value(time_value: str) -> int | None:
    """
    Extract the year (as int) from a Wikidata time string like '+0044-03-15T00:00:00Z'.
    """
    if not time_value or len(time_value) < 5:
        return None
    try:
        year_str = time_value.lstrip("+").split("-")[0]
        return int(year_str)
    except Exception:
        return None


def _batch_fetch_entities_meta(
    entity_ids: List[str],
    language: str = "en",
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch birth_year, death_year, instance-of types and aliases for multiple
    Wikidata ids using a single wbgetentities call. Results are cached.
    """
    clean_ids = [eid for eid in entity_ids if eid]
    ids_to_fetch = [eid for eid in clean_ids if eid not in ENTITY_META_CACHE]

    if ids_to_fetch:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": "|".join(ids_to_fetch),
            "format": "json",
            "props": "claims|aliases",
            "languages": language,
        }
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            resp.raise_for_status()
            entities = resp.json().get("entities", {})

            for eid, edata in entities.items():
                claims = edata.get("claims", {})

                birth_year = None
                p569 = claims.get("P569", [])
                if p569:
                    mainsnak = p569[0].get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})
                    value = datavalue.get("value", {})
                    time_val = value.get("time", "")
                    birth_year = _extract_year_from_time_value(time_val)

                death_year = None
                p570 = claims.get("P570", [])
                if p570:
                    mainsnak = p570[0].get("mainsnak", {})
                    datavalue = mainsnak.get("datavalue", {})
                    value = datavalue.get("value", {})
                    time_val = value.get("time", "")
                    death_year = _extract_year_from_time_value(time_val)

                types: List[str] = []
                for claim in claims.get("P31", []):
                    value = (
                        claim.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                    )
                    ent_id = value.get("id")
                    if ent_id:
                        types.append(ent_id)

                aliases_data = (
                    edata.get("aliases", {}).get(language, [])
                )
                aliases = [entry.get("value", "") for entry in aliases_data]

                ENTITY_META_CACHE[eid] = {
                    "birth_year": birth_year,
                    "death_year": death_year,
                    "types": types,
                    "aliases": aliases,
                }
        except Exception:
            # If batch fetch fails, fall through with whatever we have
            pass

    return {
        eid: ENTITY_META_CACHE.get(
            eid,
            {
                "birth_year": None,
                "death_year": None,
                "types": [],
                "aliases": [],
            },
        )
        for eid in clean_ids
    }


def _process_search_results(
    name: str,
    search_results: List[Dict[str, Any]],
    max_results: int,
    date_threshold: int,
    expected_category: str | None,
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process wbsearchentities results using cached/batched metadata lookup.
    """
    if not search_results:
        nores = {
            "term": "No results",
            "wikidata_id": "",
            "link": "",
            "birth_year": None,
            "death_year": None,
            "types": [],
            "score": 0,
            "aliases": [],
        }
        return [nores], [nores]

    candidate_ids = [item.get("id", "") for item in search_results if item.get("id")]
    meta_by_id = _batch_fetch_entities_meta(candidate_ids)

    results_with_date: List[Dict[str, Any]] = []
    results_without_date: List[Dict[str, Any]] = []

    for item in search_results:
        wikidata_id = item.get("id", "")
        if not wikidata_id:
            continue

        label = item.get("label", "") or ""
        link = f"https://www.wikidata.org/wiki/{wikidata_id}"
        meta = meta_by_id.get(
            wikidata_id,
            {"birth_year": None, "death_year": None, "types": [], "aliases": []},
        )
        birth_year = meta.get("birth_year")
        death_year = meta.get("death_year")
        types = meta.get("types", [])
        aliases = meta.get("aliases", [])

        if expected_category:
            if not any(t in EXPECTED_TYPES.get(expected_category, []) for t in types):
                continue

        score = score_wikidata_result(
            name,
            label,
            expected_category,
            birth_year,
            death_year,
            types,
            date_threshold=date_threshold,
        )

        result_entry = {
            "term": label,
            "wikidata_id": wikidata_id,
            "link": link,
            "birth_year": birth_year,
            "death_year": death_year,
            "types": types,
            "score": score,
            "aliases": aliases,
        }

        date_val = death_year if death_year is not None else birth_year

        if strict:
            if date_val is not None:
                if date_val <= date_threshold:
                    results_with_date.append(result_entry)
                else:
                    continue
            else:
                results_without_date.append(result_entry)
        else:
            if date_val is not None:
                results_with_date.append(result_entry)
            else:
                results_without_date.append(result_entry)

    results_with_date = sorted(
        results_with_date, key=lambda x: x.get("score", 0), reverse=True
    )[:max_results]
    results_without_date = sorted(
        results_without_date, key=lambda x: x.get("score", 0), reverse=True
    )[:max_results]

    return results_with_date, results_without_date


def score_wikidata_result(
    term: str,
    label: str,
    expected_category: str | None,
    birth_year: int | None,
    death_year: int | None,
    types: List[str],
    date_threshold: int = DATE_THRESHOLD,
) -> int:
    """
    Compute a heuristic score based on:
    - exact label match with the search term (+2),
    - compatibility of types with EXPECTED_TYPES[expected_category] (+2),
    - birth/death year <= date_threshold (+2).
    """
    score = 0
    if term.lower() == label.lower():
        score += 2

    compatible = (
        any(t in EXPECTED_TYPES.get(expected_category, []) for t in types)
        if expected_category
        else False
    )
    if compatible:
        score += 2

    date_val = death_year if death_year is not None else birth_year
    if date_val is not None and date_val <= date_threshold:
        score += 2

    return score


# =========================
# High-level Wikidata query
# =========================

def get_wikidata_details(
    name: str,
    max_results: int = MAX_RESULTS,
    date_threshold: int = DATE_THRESHOLD,
    expected_category: str | None = None,
    strict: bool = True,
    base_sleep: float = 0.2,
    error_sleep_min: float = 1.0,
    error_sleep_max: float = 4.0,
    max_retries: int = 3,
    json_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Optimized Wikidata search with caching, batched metadata retrieval and reduced sleeps.
    """
    name = (name or "").strip()
    if not name:
        empty_item = {
            "term": "",
            "wikidata_id": "",
            "link": "",
            "birth_year": None,
            "death_year": None,
            "types": [],
            "score": 0,
            "aliases": [],
        }
        return [empty_item], [empty_item]

    cache_key = (name, max_results, date_threshold, expected_category, strict)
    if cache_key in SEARCH_RESULT_CACHE:
        return SEARCH_RESULT_CACHE[cache_key]

    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "limit": max_results * 5,
        "type": "item",
    }

    # Try to load cached JSON from disk first
    data = None
    json_path: Path | None = None
    if json_dir is not None:
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / f"{sanitize_filename(name)}.json"
        if json_path.exists():
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None

    if data is not None:
        search_results = data.get("search", [])
        results_with_date, results_without_date = _process_search_results(
            name,
            search_results,
            max_results,
            date_threshold,
            expected_category,
            strict,
        )
        SEARCH_RESULT_CACHE[cache_key] = (results_with_date, results_without_date)
        print(
            f"[WIKIDATA-CACHED] '{name}': {len(results_with_date)} with_date, "
            f"{len(results_without_date)} without_date (strict={strict})"
        )
        return results_with_date, results_without_date

    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = SESSION.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if json_path is not None:
                try:
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            search_results = data.get("search", [])
            results_with_date, results_without_date = _process_search_results(
                name,
                search_results,
                max_results,
                date_threshold,
                expected_category,
                strict,
            )
            SEARCH_RESULT_CACHE[cache_key] = (results_with_date, results_without_date)

            print(
                f"[WIKIDATA] '{name}': {len(results_with_date)} with_date, "
                f"{len(results_without_date)} without_date (strict={strict})"
            )
            if base_sleep > 0:
                time.sleep(base_sleep)
            return results_with_date, results_without_date

        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                sleep_time = random.uniform(error_sleep_min, error_sleep_max)
                print(
                    f"[WIKIDATA] Error for '{name}' ({e}). "
                    f"Retrying in {sleep_time:.2f}s (attempt {attempt}/{max_retries})..."
                )
                time.sleep(sleep_time)
            else:
                print(
                    f"[WIKIDATA] Error for '{name}' after {max_retries} attempts. "
                    f"Returning error placeholder."
                )

    error_entry = {
        "term": "Error",
        "wikidata_id": "",
        "link": "",
        "birth_year": None,
        "death_year": None,
        "types": [],
        "score": 0,
        "aliases": [],
    }
    SEARCH_RESULT_CACHE[cache_key] = ([error_entry], [error_entry])
    return [error_entry], [error_entry]


# =========================
# DataFrame enrichment logic
# =========================

def _ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that we have at least:
    - entity, lemma, entity_type, author (author can be inferred from filename if missing).
    """
    if "author" not in df.columns:
        if "filename" in df.columns:
            df["author"] = df["filename"].astype(str).str.split("_", n=1).str[0]
        else:
            df["author"] = "Unknown"
    missing = [col for col in ["entity", "lemma", "entity_type"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input DataFrame: {missing}")
    return df


def _add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'entity_clean' and 'lemma_clean' columns for grouping.
    """
    df["entity_clean"] = df["entity"].astype(str).str.strip().str.lower()
    df["lemma_clean"] = df["lemma"].astype(str).str.strip().str.lower()
    return df


def _init_wikidata_columns(df: pd.DataFrame, max_results: int = MAX_RESULTS) -> pd.DataFrame:
    """
    Add empty columns for Wikidata candidates (entity/lemma × date/nodate × strict/lax)
    in a single batch to avoid DataFrame fragmentation.
    """
    prefixes = ["entity", "lemma"]
    modes = ["date", "nodate", "date_lax", "nodate_lax"]

    new_columns: list[str] = []
    for prefix in prefixes:
        for mode in modes:
            for i in range(1, max_results + 1):
                for field in WIKIDATA_CANDIDATE_FIELDS:
                    col_name = f"{prefix}_wikidata_{field}_{mode}_{i}"
                    if col_name not in df.columns:
                        new_columns.append(col_name)

    if not new_columns:
        return df

    block = pd.DataFrame("", index=df.index, columns=new_columns)
    df = pd.concat([df, block], axis=1)
    return df


def _wikidata_column_block(prefix: str, mode: str, max_results: int) -> list[str]:
    """
    Return the ordered list of Wikidata column names for the prefix/mode pair.
    """
    columns: list[str] = []
    for i in range(1, max_results + 1):
        for field in WIKIDATA_CANDIDATE_FIELDS:
            columns.append(f"{prefix}_wikidata_{field}_{mode}_{i}")
    return columns


def _fill_candidate_block(
    df: pd.DataFrame,
    indices: List[int],
    prefix: str,
    mode: str,
    candidates: List[Dict[str, Any]],
    max_results: int = MAX_RESULTS,
) -> None:
    """
    Fill the corresponding Wikidata columns for a given set of rows, prefix (entity/lemma),
    and mode (date/nodate/date_lax/nodate_lax) with the data from 'candidates'.
    """
    block_columns = _wikidata_column_block(prefix, mode, max_results)
    per_candidate = len(WIKIDATA_CANDIDATE_FIELDS)

    for row_idx in indices:
        values = [""] * len(block_columns)

        if not candidates:
            if block_columns:
                values[0] = "No results"
            df.loc[row_idx, block_columns] = values
            continue

        for idx, cand in enumerate(candidates[:max_results]):
            offset = idx * per_candidate
            values[offset + 0] = cand.get("term", "")
            values[offset + 1] = cand.get("link", "")
            values[offset + 2] = cand.get("birth_year", "")
            values[offset + 3] = cand.get("death_year", "")
            values[offset + 4] = ",".join(cand.get("types", []))
            values[offset + 5] = cand.get("score", "")
            values[offset + 6] = ", ".join(cand.get("aliases", []))

        df.loc[row_idx, block_columns] = values


def enrich_entities_with_wikidata(
    input_csv: Path = INPUT_CSV,
    output_csv: Path = OUTPUT_CSV,
    json_dir: Path = WIKIDATA_JSON_DIR,
    max_results: int = MAX_RESULTS,
    date_threshold: int = DATE_THRESHOLD,
    max_rows: int | None = MAX_ROWS_TO_PROCESS,
) -> pd.DataFrame:
    """
    Main pipeline:
    - Read the entities CSV.
    - Optionally restrict to the first 'max_rows' for testing.
    - Add helper columns and empty Wikidata columns.
    - Group by (entity_clean, lemma_clean, entity_type) to avoid repeated queries.
    - For each group, query Wikidata for:
        * entity strict + lax
        * lemma strict + lax
    - Fill the candidate columns for all rows in each group.
    - Save the resulting DataFrame to output_csv and return it.
    """
    if not input_csv.exists():
        print(f"[WARN] Input CSV not found: {input_csv}. Skipping Wikidata enrichment.")
        return pd.DataFrame()

    df = pd.read_csv(input_csv)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        print(f"[INFO] Processing only the first {max_rows} rows for testing.")

    if df.empty:
        print("[WARN] Input CSV is empty; no Wikidata queries will be made.")
        df.to_csv(output_csv, index=False)
        return df

    df = _ensure_core_columns(df)
    df = _add_clean_columns(df)
    df = _init_wikidata_columns(df, max_results=max_results)

    grouped = df.groupby(["entity_clean", "lemma_clean", "entity_type"])
    print(f"[INFO] Unique (entity, lemma, entity_type) groups: {len(grouped)}")

    for (ent_clean, lem_clean, ent_type), group in grouped:
        indices = group.index.tolist()
        entity_text = str(group.iloc[0]["entity"]).strip()
        lemma_text = str(group.iloc[0]["lemma"]).strip()
        expected_category = ent_type if ent_type in ["PERSON", "LOC", "NORP"] else None

        print(
            f"\n[GROUP] entity={entity_text!r}, lemma={lemma_text!r}, "
            f"type={ent_type!r}, rows={len(indices)}"
        )

        # Entity strict / lax
        ent_date_strict, ent_nodate_strict = get_wikidata_details(
            entity_text,
            max_results=max_results,
            date_threshold=date_threshold,
            expected_category=expected_category,
            strict=True,
            json_dir=json_dir,
        )
        ent_date_lax, ent_nodate_lax = get_wikidata_details(
            entity_text,
            max_results=max_results,
            date_threshold=date_threshold,
            expected_category=expected_category,
            strict=False,
            json_dir=json_dir,
        )

        _fill_candidate_block(df, indices, "entity", "date", ent_date_strict, max_results)
        _fill_candidate_block(df, indices, "entity", "nodate", ent_nodate_strict, max_results)
        _fill_candidate_block(df, indices, "entity", "date_lax", ent_date_lax, max_results)
        _fill_candidate_block(df, indices, "entity", "nodate_lax", ent_nodate_lax, max_results)

        # Lemma strict / lax
        lem_date_strict, lem_nodate_strict = get_wikidata_details(
            lemma_text,
            max_results=max_results,
            date_threshold=date_threshold,
            expected_category=expected_category,
            strict=True,
            json_dir=json_dir,
        )
        lem_date_lax, lem_nodate_lax = get_wikidata_details(
            lemma_text,
            max_results=max_results,
            date_threshold=date_threshold,
            expected_category=expected_category,
            strict=False,
            json_dir=json_dir,
        )

        _fill_candidate_block(df, indices, "lemma", "date", lem_date_strict, max_results)
        _fill_candidate_block(df, indices, "lemma", "nodate", lem_nodate_strict, max_results)
        _fill_candidate_block(df, indices, "lemma", "date_lax", lem_date_lax, max_results)
        _fill_candidate_block(df, indices, "lemma", "nodate_lax", lem_nodate_lax, max_results)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Wikidata-enriched CSV saved to: {output_csv}")
    return df


# =========================
# CLI entry point
# =========================

def main():
    enrich_entities_with_wikidata()


if __name__ == "__main__":
    main()
