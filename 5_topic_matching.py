"""
soldevila_topic_matching.py

Dictionary-based topic matching using Soldevila's "Diccionario de motivos amatorios"
and Levenshtein distance over Latin poems.

Pipeline (high level)
---------------------
1) Dictionary preprocessing:
   - Read Soldevila's dictionary plain-text file.
   - Detect topic headings (in uppercase at the beginning of lines).
   - For each line, propagate the last seen topic backwards if needed.
   - Normalize the text and generate word n-grams (length 3..15).
   - Store all fragments with their topic and dictionary section in a JSON cache.

2) Matching over poem files:
   - For each poem .txt file (possibly in nested folders):
       - For each line and each dictionary fragment:
           - Compute Levenshtein distance on:
               a) the full normalized line, and
               b) each n-gram of that line.
           - If the minimum distance is <= main threshold, record a match.
           - Additionally, for a list of "test thresholds", record extra matches
             with 'umbral_usado' set to that threshold.
   - Use a state JSON to keep track of the last processed file index,
     so the script can be resumed without re-processing everything.
   - Save raw matches to a CSV as we go (append mode).

3) Optional post-filter:
   - For a batch of results (e.g., from the last run),
     keep only the longest fragment per
     (file, line_number, method, threshold, topic).

This module is intentionally simple and configuration-driven so it can be used
in local environments, clusters, or notebooks, without Google Colab specifics.
"""

from __future__ import annotations

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

from nltk.util import ngrams
from Levenshtein import distance as levenshtein_distance
import glob


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

# Soldevila dictionary paths
DICTIONARY_TXT = BASE_DIR / "soldevila" / "Soldevila8.txt"
TOKENIZED_JSON = BASE_DIR / "soldevila" / "soldevila_ngrams.json"

# State file for resumable processing
STATE_PATH = BASE_DIR / "soldevila" / "topic_matching_state.json"

# Output CSVs
RAW_MATCHES_CSV = BASE_DIR / "outputs" / "soldevila_matches_raw.csv"
FILTERED_MATCHES_CSV = BASE_DIR / "outputs" / "soldevila_matches_filtered.csv"
TOPICS_DIR = BASE_DIR / "topics"

# Root folder with poems (.txt). You can point this to Propertius, Ovidius, etc.
# By default, process all author subfolders under BASE_DIR (e.g., Catulo/, Tibulo/, Propercio/)
TEXT_ROOT = BASE_DIR

# When falling back to files directly under BASE_DIR, use these prefixes to
# identify which .txt files correspond to poems (sample data case).
AUTHOR_PREFIXES = ("Catulo", "Tibulo", "Propercio")

# Mapping of author prefixes to per-author topic CSV filenames
AUTHOR_TOPIC_FILES: Dict[str, str] = {
    "Catulo": "Catullus_matches_clean.csv",
    "Tibulo": "Tibullus_matches_clean.csv",
    "Propercio": "Propertius_matches_clean.csv",
    "Ovidio": "Ovid_matches_clean.csv",
}

# Main Levenshtein threshold for "strong" matches
MAIN_THRESHOLD = 1

# Thresholds to log as additional rows (for later analysis)
TEST_THRESHOLDS = [1, 2, 3]

# Number of files to process per run (resumable batches)
BATCH_SIZE = 10


# =========================
# State management
# =========================

def load_state(state_path: Path) -> Dict[str, Any]:
    """
    Load a JSON state file with at least {'last_file_index': int}.
    If it does not exist, return a default state.
    """
    if not state_path.exists():
        return {"last_file_index": 0}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    """
    Save the state dictionary to a JSON file.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f">> [STATE] Saved to {state_path} => {state}")


def reset_state(state_path: Path = STATE_PATH) -> None:
    """
    Convenience function to reset the state to zero.
    """
    save_state(state_path, {"last_file_index": 0})


# =========================
# Text normalization and n-grams
# =========================

def normalize_text(text: str) -> str:
    """
    Normalize text:
      - remove non-word characters and underscores
      - convert to lowercase
      - strip leading/trailing spaces

    Note: Handling of '/' can be customized if needed; here we treat it as
    any other non-word character.
    """
    text = re.sub(r"[^\w]+", " ", text)  # remove punctuation, keep letters/digits/underscore
    text = text.lower().strip()
    return text


def generate_ngrams(text: str, min_n: int = 3, max_n: int = 15) -> List[str]:
    """
    Generate word n-grams from 'text' for n in [min_n, max_n].
    """
    tokens = text.split()
    ngrams_result: List[str] = []
    for n in range(min_n, max_n + 1):
        for ng in ngrams(tokens, n):
            ngrams_result.append(" ".join(ng))
    return ngrams_result


# =========================
# Dictionary preprocessing
# =========================

def detect_topic_from_line(line: str) -> Tuple[str | None, int]:
    """
    Given a line from the dictionary, detect if it starts with an uppercase topic.

    Returns:
        (topic_string, position_in_line) or (None, -1) if no topic is detected.

    Topic pattern: one or more uppercase words (possibly with accents) at the
    beginning of the line.
    """
    line_strip = line.strip()
    if not line_strip:
        return None, -1

    pattern = r"^([A-ZÁÉÍÓÚÑ]+(?: [A-ZÁÉÍÓÚÑ]+)*)"
    match = re.match(pattern, line_strip)
    if not match:
        return None, -1

    block = match.group(1).strip()
    if len(block) < 2:
        return None, -1

    pos_in_line = match.start(1)
    return block, pos_in_line


def propagate_topic_backwards(
    all_lines: List[str],
    topics_per_line: List[Tuple[str | None, int]],
    idx: int,
) -> Tuple[str | None, int]:
    """
    Given a line index 'idx', walk backwards until we find the last line
    that has a non-None topic. Return (topic, position).
    """
    for j in range(idx, -1, -1):
        topic_j, pos_j = topics_per_line[j]
        if topic_j is not None:
            return topic_j, pos_j
    return None, -1


def load_or_tokenize_dictionary(
    dictionary_txt: Path = DICTIONARY_TXT,
    tokenized_json: Path = TOKENIZED_JSON,
    force_retokenize: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load the dictionary fragments from JSON if available, otherwise:
      - read the dictionary text file
      - detect topics per line and propagate them
      - normalize each line and generate n-grams
      - store results in JSON

    Returns a list of dicts with keys:
      - 'section': e.g. "DiccLine 123"
      - 'fragment': the n-gram string
      - 'topic': the propagated topic (uppercase heading)
      - 'pos_in_line': integer position where topic starts in original line
    """
    if tokenized_json.exists() and not force_retokenize:
        print(f">> [DICTIONARY] Loading tokenization from: {tokenized_json}")
        with tokenized_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f">> [DICTIONARY] Processing tokenization (force_retokenize={force_retokenize})...")
    with dictionary_txt.open("r", encoding="utf-8") as f:
        all_lines = f.read().splitlines()

    topics_per_line: List[Tuple[str | None, int]] = [
        detect_topic_from_line(line) for line in all_lines
    ]

    dictionary_fragments: List[Dict[str, Any]] = []
    fragment_count = 0

    for i, line in enumerate(all_lines):
        line_strip = line.strip()
        if not line_strip:
            continue

        topic, pos = propagate_topic_backwards(all_lines, topics_per_line, i)
        if not topic:
            topic, pos = "SIN_TEMA", -1

        normalized_line = normalize_text(line_strip)
        if not normalized_line:
            continue

        for fragment in generate_ngrams(normalized_line, min_n=3, max_n=15):
            fragment_count += 1
            dictionary_fragments.append(
                {
                    "section": f"DiccLine {i+1}",
                    "fragment": fragment,
                    "topic": topic,
                    "pos_in_line": pos,
                }
            )

    print(f">> [DICTIONARY] Total n-grams generated = {fragment_count}")
    tokenized_json.parent.mkdir(parents=True, exist_ok=True)
    with tokenized_json.open("w", encoding="utf-8") as f:
        json.dump(dictionary_fragments, f, ensure_ascii=False, indent=2)
    print(f">> [DICTIONARY] Tokenization saved to: {tokenized_json}")
    return dictionary_fragments


# =========================
# Levenshtein matching
# =========================

def levenshtein_best_match(fragment: str, text: str, mode: str) -> Tuple[str, int]:
    """
    Compute the best Levenshtein match between 'fragment' and 'text'.

    mode = 'full' -> compare fragment with the full normalized line
    mode = 'ngram' -> compare fragment with every n-gram of the normalized line

    Returns:
        (best_match_string, distance)
    """
    min_dist = float("inf")
    best_match = ""

    normalized_text = normalize_text(text)

    if mode == "full":
        dist = levenshtein_distance(fragment, normalized_text)
        return normalized_text, dist

    elif mode == "ngram":
        for ng in generate_ngrams(normalized_text):
            dist = levenshtein_distance(fragment, ng)
            if dist < min_dist:
                min_dist = dist
                best_match = ng
        if min_dist is float("inf"):
            return "", min_dist
        return best_match, min_dist

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================
# Filtering: keep only the longest fragment per group
# =========================

def filter_longest_per_line_and_topic(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Given a list of result dicts, keep only the longest 'fragment_buscado'
    per (file, line_number, method, threshold, topic).

    This is useful to reduce noise from shorter overlapping matches.
    """
    best_per_group: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for item in results:
        key = (
            item["archivo_texto"],
            item["numero_linea_texto"],
            item["metodo"],
            item.get("umbral_usado", ""),
            item.get("topico", ""),
        )
        fragment = item["fragmento_buscado"]

        if key not in best_per_group:
            best_per_group[key] = item
        else:
            current_best = best_per_group[key]
            if len(fragment) > len(current_best["fragmento_buscado"]):
                best_per_group[key] = item

    return list(best_per_group.values())


# =========================
# Split filtered matches by author
# =========================

def split_matches_by_author(
    rows: List[Dict[str, Any]],
    topics_dir: Path = TOPICS_DIR,
    author_files: Dict[str, str] = AUTHOR_TOPIC_FILES,
) -> None:
    """
    Write per-author CSVs expected by the TEI annotation step.

    The author is inferred from the prefix of 'archivo_texto', e.g. "Catulo_",
    "Tibulo_", "Propercio_", "Ovidio_". Only prefixes present in author_files
    are considered.
    """
    if not rows:
        return

    topics_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = {prefix: [] for prefix in author_files}

    for row in rows:
        fname = str(row.get("archivo_texto", ""))
        for prefix in author_files:
            if fname.startswith(prefix):
                grouped[prefix].append(row)
                break

    fieldnames = list(rows[0].keys())
    for prefix, items in grouped.items():
        out_path = topics_dir / author_files[prefix]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if items:
                writer.writerows(items)
        print(f"[TOPICS] Saved {len(items)} rows for '{prefix}' to {out_path}")


# =========================
# Saving results
# =========================

FIELDNAMES = [
    "fragmento_buscado",
    "fragmento_encontrado",
    "distancia",
    "archivo_texto",
    "numero_linea_texto",
    "linea_texto",
    "seccion_diccionario",
    "topico",
    "metodo",
    "umbral_usado",
    "pos_in_line",
]


def save_results_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    Append results to a CSV file, creating the header if the file does not exist.
    """
    print(f">> [SAVE] CSV => {csv_path} ({len(results)} rows)")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for row in results:
            # Ensure optional keys exist
            row.setdefault("umbral_usado", "")
            row.setdefault("topico", "")
            row.setdefault("pos_in_line", "")
            writer.writerow(row)


# =========================
# Batch processing over text files
# =========================

def process_texts_in_batches(
    file_list: List[Path],
    dictionary_fragments: List[Dict[str, Any]],
    main_threshold: int = MAIN_THRESHOLD,
    test_thresholds: List[int] | None = None,
    state_path: Path = STATE_PATH,
    batch_size: int = BATCH_SIZE,
    raw_csv_path: Path = RAW_MATCHES_CSV,
) -> List[Dict[str, Any]]:
    """
    Main matching routine:
      - Reads the current state (last processed file index).
      - Processes 'batch_size' files from that index.
      - For each file:
          - For each line:
              - For each dictionary fragment:
                  - compute Levenshtein matches (full line + n-grams)
                  - record results if distance <= main_threshold and/or test thresholds.
      - Saves matches for each file to 'raw_csv_path' (append).
      - Updates the state file after each file.
      - Returns all results found in this batch (not the whole corpus).
    """
    if test_thresholds is None:
        test_thresholds = TEST_THRESHOLDS

    print(f"\n>> [BATCHES] Resumable mode with batch_size={batch_size}")
    state = load_state(state_path)
    last_idx = state.get("last_file_index", 0)

    start_idx = last_idx
    end_idx = min(start_idx + batch_size, len(file_list))

    if start_idx >= len(file_list):
        print("[INFO] No more files to process.")
        return []

    print(
        f"[INFO] Processing files {start_idx} to {end_idx - 1} "
        f"out of {len(file_list)} total."
    )
    files_to_process = file_list[start_idx:end_idx]

    batch_results: List[Dict[str, Any]] = []

    for idx, file_path in enumerate(files_to_process, start=start_idx):
        print(f"\n[PROCESS] idx={idx}: {file_path}")
        with file_path.open("r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        file_results: List[Dict[str, Any]] = []
        seen: set[Tuple[Any, ...]] = set()

        for line_number, line_text in enumerate(lines, start=1):
            for entry in dictionary_fragments:
                fragment = entry["fragment"]
                topic = entry["topic"]
                section = entry["section"]
                pos_in_line = entry.get("pos_in_line", -1)

                # Best match on full line and n-grams
                match_full, dist_full = levenshtein_best_match(fragment, line_text, "full")
                match_ng, dist_ng = levenshtein_best_match(fragment, line_text, "ngram")

                best_dist = min(dist_full, dist_ng)
                best_match = match_full if dist_full <= dist_ng else match_ng
                method = "texto_completo" if dist_full <= dist_ng else "ngrama"

                # 1) Strong match (main_threshold)
                if best_dist <= main_threshold:
                    key = (
                        fragment,
                        best_dist,
                        file_path.name,
                        section,
                        method,
                        "",
                        line_number,
                    )
                    if key not in seen:
                        seen.add(key)
                        file_results.append(
                            {
                                "fragmento_buscado": fragment,
                                "fragmento_encontrado": best_match,
                                "distancia": best_dist,
                                "archivo_texto": file_path.name,
                                "numero_linea_texto": line_number,
                                "linea_texto": line_text,
                                "seccion_diccionario": section,
                                "topico": topic,
                                "metodo": method,
                                "umbral_usado": "",
                                "pos_in_line": pos_in_line,
                            }
                        )

                # 2) Additional matches for test thresholds
                for th in test_thresholds:
                    if best_dist <= th:
                        key2 = (
                            fragment,
                            best_dist,
                            file_path.name,
                            section,
                            method,
                            th,
                            line_number,
                        )
                        if key2 not in seen:
                            seen.add(key2)
                            file_results.append(
                                {
                                    "fragmento_buscado": fragment,
                                    "fragmento_encontrado": best_match,
                                    "distancia": best_dist,
                                    "archivo_texto": file_path.name,
                                    "numero_linea_texto": line_number,
                                    "linea_texto": line_text,
                                    "seccion_diccionario": section,
                                    "topico": topic,
                                    "metodo": method,
                                    "umbral_usado": th,
                                    "pos_in_line": pos_in_line,
                                }
                            )
                        break  # once matched for a given th, do not add for higher thresholds

        if file_results:
            save_results_csv(file_results, raw_csv_path)
            batch_results.extend(file_results)
            print(
                f">> [FILE] {file_path} => {len(file_results)} matches saved "
                f"to {raw_csv_path}"
            )
        else:
            print(f">> [FILE] {file_path} => no matches found.")

        # Update state after each file
        state["last_file_index"] = idx + 1
        save_state(state_path, state)

    return batch_results


# =========================
# CLI entry point
# =========================

def main():
    # 1) Find all .txt files under TEXT_ROOT (recursively)
    text_root = TEXT_ROOT
    if text_root.exists():
        file_list = sorted(text_root.glob("**/*.txt"))
        print(f"[INFO] Found {len(file_list)} .txt files under {text_root}")
    else:
        fallback_files: list[Path] = []
        for prefix in AUTHOR_PREFIXES:
            fallback_files.extend(sorted(BASE_DIR.glob(f"{prefix}_*.txt")))
        file_list = fallback_files
        print(
            f"[INFO] TEXT_ROOT '{text_root}' not found. "
            f"Using {len(file_list)} files with prefixes {AUTHOR_PREFIXES} under {BASE_DIR}."
        )

    # 2) Load or build dictionary n-grams
    dictionary_path = DICTIONARY_TXT
    if not dictionary_path.exists():
        alternative = BASE_DIR / "Soldevila8-Juguete.txt"
        if alternative.exists():
            print(
                f"[INFO] Dictionary file '{dictionary_path}' not found. "
                f"Using sample file: {alternative}"
            )
            dictionary_path = alternative
        else:
            raise FileNotFoundError(
                f"Dictionary file not found: {dictionary_path} "
                f"(also checked {alternative})"
            )

    dictionary_fragments = load_or_tokenize_dictionary(
        dictionary_txt=dictionary_path,
        tokenized_json=TOKENIZED_JSON,
        force_retokenize=False,
    )

    # 3) Process in resumable batches -> raw matches CSV
    batch_results = process_texts_in_batches(
        file_list=file_list,
        dictionary_fragments=dictionary_fragments,
        main_threshold=MAIN_THRESHOLD,
        test_thresholds=TEST_THRESHOLDS,
        state_path=STATE_PATH,
        batch_size=BATCH_SIZE,
        raw_csv_path=RAW_MATCHES_CSV,
    )

    # 4) Optional: post-filter this batch and save to a filtered CSV
    if batch_results:
        filtered = filter_longest_per_line_and_topic(batch_results)
        save_results_csv(filtered, FILTERED_MATCHES_CSV)
        print(
            f"[FILTER] Filtered CSV saved to {FILTERED_MATCHES_CSV} "
            f"with {len(filtered)} rows."
        )
        split_matches_by_author(filtered)
    else:
        print("[INFO] No results in this batch; nothing to filter.")

    print(
        "\n[DEBUG] Batch finished. "
        "Re-run the script to process the next batch if files remain."
    )


if __name__ == "__main__":
    main()
