# src/entities_pipeline.py
"""
Named Entity Recognition pipeline for Latin texts using LatinCy,
TEI inline tagging, and quality control against printed Index Nominum.

This module is designed to be simple and reproducible for the TEI article.
It removes Google Colab-specific parts and absolute personal paths.
"""

from pathlib import Path
import csv
from typing import Dict

import spacy
import pandas as pd
from lxml import etree as ET
import fitz  # PyMuPDF


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

# Input plain-text folders (one poem per .txt file)
# If folders do not exist (sample data), the pipeline falls back to files whose
# names start with the author prefix (e.g., Catulo_*.txt) under BASE_DIR.
TEXT_FOLDERS = [
    BASE_DIR / "Catulo",
    BASE_DIR / "Tibulo",
    BASE_DIR / "Propercio",
]

# TEI input/output folders
TEI_VERSES_DIR = BASE_DIR / "tei_pipeline" / "03_tei_verses_counted"
TEI_TAGGED_DIR = BASE_DIR / "tei_pipeline" / "04_tei_entities_tagged"

# Entities and intermediate data
ENTITIES_DIR = BASE_DIR / "entities"
ENTITIES_CSV = ENTITIES_DIR / "entities.csv"
ENTITIES_LEMMA_CSV = ENTITIES_DIR / "entities_lemmatized.csv"
ENTITIES_INDEX_CHECKED_CSV = ENTITIES_DIR / "entities_index_checked.csv"
INDEX_MATCH_SUMMARY_CSV = ENTITIES_DIR / "index_match_summary.csv"

# Index Nominum PDFs (one per author)
INDEX_PDFS = {
    "Catulo": BASE_DIR / "indices" / "Index_Nominum_Catullus.pdf",
    "Tibulo": BASE_DIR / "indices" / "Index_Nominum_Tibulli.pdf",
    "Propercio": BASE_DIR / "indices" / "Index_Nominum_Propertius.pdf",
}


# =========================
# Helpers
# =========================

def load_nlp(model_name: str = "la_core_web_lg"):
    """Load the LatinCy model."""
    try:
        nlp = spacy.load(model_name)
        print(f"[OK] spaCy model loaded: {model_name}")
        return nlp
    except Exception as e:
        fallback_model = "la_core_web_sm"
        if model_name != fallback_model:
            print(
                f"[WARN] Could not load model '{model_name}': {e}. "
                f"Trying fallback '{fallback_model}'."
            )
            return load_nlp(fallback_model)

        print(
            "[WARN] Could not load any pretrained Latin model. "
            "Falling back to a blank 'la' pipeline (no entities will be detected)."
        )
        return spacy.blank("la")


def _resolve_author_files(folder: Path) -> list[Path]:
    """
    Return the text files associated with an author folder. If the folder does
    not exist, fall back to files with the author's prefix under BASE_DIR.
    """
    if folder.exists() and folder.is_dir():
        files = sorted(folder.glob("*.txt"))
        if files:
            return files

    fallback = sorted(folder.parent.glob(f"{folder.name}_*.txt"))
    if fallback:
        print(
            f"[INFO] Folder '{folder}' not found. Using "
            f"{len(fallback)} files matching '{folder.name}_*.txt'."
        )
    return fallback


# =========================
# 1) NER extraction to CSV
# =========================

def extract_entities_from_texts(
    text_folders: list[Path],
    nlp,
    output_csv: Path,
) -> Dict[str, Dict[str, int]]:
    """
    Run NER on all .txt files of the given folders and save a CSV with:
    filename, entity, entity_type.

    Returns a dictionary with counts per author and entity type.
    """
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[list[str]] = []
    author_counts: Dict[str, Dict[str, int]] = {}

    for folder in text_folders:
        files = _resolve_author_files(folder)
        if not files:
            print(f"[WARN] No .txt files found for: {folder}")
            continue

        author = folder.name
        author_counts.setdefault(author, {"PERSON": 0, "LOC": 0, "NORP": 0})
        print(f"[INFO] Processing author folder: {author}")

        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            doc = nlp(text)

            print(f"  File: {file_path.name}")
            for ent in doc.ents:
                rows.append([file_path.name, ent.text, ent.label_])
                # update counts
                author_counts[author][ent.label_] = author_counts[author].get(ent.label_, 0) + 1
                print(f"    {ent.text!r} ({ent.label_})")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "entity", "entity_type"])
        writer.writerows(rows)

    print(f"[OK] Entities CSV saved at: {output_csv}")
    print("\n[SUMMARY] Entity counts by author:")
    for author, counts in author_counts.items():
        print(
            f"  {author}: "
            f"{counts.get('PERSON', 0)} PERSON, "
            f"{counts.get('LOC', 0)} LOC, "
            f"{counts.get('NORP', 0)} NORP"
        )

    return author_counts


# =========================
# 2) Inline TEI tagging
# =========================

def create_tei_markup(entity_text: str, entity_type: str) -> str:
    """Return a TEI tag string corresponding to the entity type."""
    if entity_type == "PERSON":
        return f"<persName>{entity_text}</persName>"
    elif entity_type == "LOC":
        return f"<placeName>{entity_text}</placeName>"
    elif entity_type == "NORP":
        return f'<persName type="group">{entity_text}</persName>'
    return entity_text


def _apply_inline_markup(l_node: ET._Element, new_text: str) -> None:
    """
    Replace the text of <l> with new_text, interpreting it as XML fragments so
    that tags like <persName> are serialized as real elements instead of text.
    """
    wrapper_template = f"<wrapper>{new_text}</wrapper>"
    try:
        wrapper = ET.fromstring(wrapper_template)
    except ET.XMLSyntaxError:
        # If parsing fails, fall back to plain text (escaped)
        l_node.text = new_text
        # remove existing children
        for child in list(l_node):
            l_node.remove(child)
        return

    # Clear existing children
    for child in list(l_node):
        l_node.remove(child)

    l_node.text = wrapper.text
    for child in wrapper:
        l_node.append(child)


def tag_entities_in_tei(tei_input_dir: Path, entities_csv: Path, tei_output_dir: Path) -> None:
    """
    Read TEI files with <lg>/<l> lines and replace entity strings with simple
    TEI markup in the text.

    NOTE: This implementation uses string replacement. It does NOT yet create
    real TEI sub-elements (<persName/> etc.) in the XML tree. Improving that
    would require more fine-grained token-level logic.
    """
    tei_output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(entities_csv)

    files_processed = 0
    files_with_replacements = 0
    files_without_replacements: list[str] = []

    print(f"[INFO] Tagging entities in TEI files from: {tei_input_dir}")
    for file_path in tei_input_dir.glob("*.xml"):
        filename = file_path.name
        base_txt_name = filename.replace("_TEI.xml", ".txt")

        df_file = df[df["filename"] == base_txt_name]

        output_file_path = tei_output_dir / filename
        files_processed += 1

        if df_file.empty:
            print(f"  No entities found for {filename}. Copying unchanged.")
            # simple copy
            with open(file_path, "rb") as src, open(output_file_path, "wb") as dst:
                dst.write(src.read())
            files_without_replacements.append(filename)
            continue

        tree = ET.parse(str(file_path))
        root = tree.getroot()
        modified_flag = False

        # iterate over verse lines
        for l in root.xpath("//text//body//lg//l"):
            original_text = l.text or ""
            new_text = original_text

            for _, row in df_file.iterrows():
                entity_text = row["entity"]
                entity_type = row["entity_type"]
                tei_tag = create_tei_markup(entity_text, entity_type)
                if entity_text in new_text:
                    new_text = new_text.replace(entity_text, tei_tag)
                    print(
                        f"  Replacing '{entity_text}' with '{tei_tag}' "
                        f"in line n={l.get('n')} of {filename}"
                    )

            if new_text != original_text:
                modified_flag = True
                _apply_inline_markup(l, new_text)

        tree.write(str(output_file_path), encoding="utf-8", pretty_print=True, xml_declaration=True)
        if modified_flag:
            files_with_replacements += 1
            print(f"[OK] Updated TEI file: {filename}")
        else:
            print(f"[INFO] No replacements applied in: {filename}")
            files_without_replacements.append(filename)

    print("\n[SUMMARY] TEI entity tagging")
    print(f"  Total files processed: {files_processed}")
    print(f"  Files with replacements: {files_with_replacements}")
    if files_without_replacements:
        print("  Files without replacements:")
        for fname in files_without_replacements:
            print(f"    - {fname}")
    else:
        print("  All files had at least one replacement.")


# =========================
# 3) Lemmatization of entities
# =========================

def lemmatize_entities(
    entities_csv: Path,
    output_csv: Path,
    nlp,
) -> None:
    """
    Read the entities CSV, lemmatize the 'entity' field with LatinCy,
    and save a new CSV with an additional 'lemma' column.
    """
    df = pd.read_csv(entities_csv)

    def lemmatize_text(text: str) -> str:
        doc = nlp(str(text))
        return " ".join(tok.lemma_ for tok in doc)

    df["lemma"] = df["entity"].apply(lemmatize_text)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[OK] Lemmatized entities CSV saved at: {output_csv}")


# =========================
# 4) Index Nominum text extraction
# =========================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full plain text from a PDF using PyMuPDF (fitz)."""
    if not pdf_path.exists():
        print(f"[WARN] Index PDF not found: {pdf_path}")
        return ""
    doc = fitz.open(pdf_path)
    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def _resolve_index_pdf(pdf_path: Path) -> Path | None:
    """
    If the configured PDF path does not exist (e.g., sample data keeps the files
    directly under BASE_DIR), try falling back to BASE_DIR/<filename>.
    """
    if pdf_path.exists():
        return pdf_path
    fallback = BASE_DIR / pdf_path.name
    if fallback.exists():
        print(f"[INFO] Using fallback Index PDF: {fallback}")
        return fallback
    return None


def load_index_texts(index_pdfs: Dict[str, Path]) -> Dict[str, str]:
    """
    Return a dictionary {author: full_index_text} from the given PDF paths.
    """
    index_texts: Dict[str, str] = {}
    for author, pdf_path in index_pdfs.items():
        resolved = _resolve_index_pdf(pdf_path)
        if resolved is None:
            print(f"[WARN] Index PDF not available for author '{author}'.")
            index_texts[author] = ""
            continue
        print(f"[INFO] Extracting Index Nominum for {author} from: {resolved}")
        index_texts[author] = extract_text_from_pdf(resolved)
    return index_texts


# =========================
# 5) Check entities against Index Nominum
# =========================

def check_entities_in_indices(
    entities_lemma_csv: Path,
    index_texts: Dict[str, str],
    output_csv: Path,
) -> None:
    """
    For each entity (raw and lemma), check whether its string is present
    in the corresponding author's Index Nominum text. Adds:
    - in_index_raw   (bool)
    - in_index_lemma (bool)
    """
    df = pd.read_csv(entities_lemma_csv)

    # Infer author from filename prefix (e.g., "Catulo_...", "Tibulo_...")
    df["author"] = df["filename"].str.split("_", n=1).str[0]

    def in_index(term: str, author: str) -> bool:
        base = index_texts.get(author, "")
        return term.lower() in base.lower()

    df["in_index_raw"] = df.apply(
        lambda row: in_index(str(row["entity"]), row["author"]), axis=1
    )
    df["in_index_lemma"] = df.apply(
        lambda row: in_index(str(row["lemma"]), row["author"]), axis=1
    )

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[OK] Entities with index check saved at: {output_csv}")


# =========================
# 6) Summary table: counts and percentages
# =========================

def _count_index_matches(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Helper: count how many entities are in/not in index for a given boolean column.
    Returns a DataFrame with index (author, entity_type) and columns:
    - in_index
    - not_in_index
    """
    return (
        df.groupby(["author", "entity_type"])[col_name]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={True: "in_index", False: "not_in_index"})
    )


def _percentages(df_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'in_index' and 'not_in_index',
    compute percentage of 'in_index' and 'not_in_index'.
    """
    if df_counts.empty:
        return pd.DataFrame(columns=["pct_in_index", "pct_not_in_index"])

    df_percent = df_counts.copy().astype(float)
    if "in_index" not in df_percent.columns:
        df_percent["in_index"] = 0.0
    if "not_in_index" not in df_percent.columns:
        df_percent["not_in_index"] = 0.0

    total = df_percent["in_index"] + df_percent["not_in_index"]
    total = total.replace(0, 1)  # avoid division by zero
    df_percent["pct_in_index"] = (df_percent["in_index"] / total) * 100
    df_percent["pct_not_in_index"] = 100 - df_percent["pct_in_index"]
    return df_percent[["pct_in_index", "pct_not_in_index"]]


def build_index_match_summary(
    entities_index_checked_csv: Path,
    summary_csv: Path,
) -> pd.DataFrame:
    """
    Build a summary table combining:
    - raw vs lemma index matches
    - percentages per author and entity_type.

    Saves a CSV and returns the DataFrame.
    """
    df = pd.read_csv(entities_index_checked_csv)
    df["author"] = df["filename"].str.split("_", n=1).str[0]

    if df.empty:
        empty = pd.DataFrame()
        empty.to_csv(summary_csv, index=False)
        print("[WARN] No entities available to build index match summary.")
        return empty

    raw_counts = _count_index_matches(df, "in_index_raw")
    lemma_counts = _count_index_matches(df, "in_index_lemma")

    raw_pct = _percentages(raw_counts).rename(
        columns={
            "pct_in_index": "pct_in_index_raw",
            "pct_not_in_index": "pct_not_in_index_raw",
        }
    )
    lemma_pct = _percentages(lemma_counts).rename(
        columns={
            "pct_in_index": "pct_in_index_lemma",
            "pct_not_in_index": "pct_not_in_index_lemma",
        }
    )

    summary = (
        raw_counts.add_suffix("_raw")
        .join(lemma_counts.add_suffix("_lemma"))
        .join(raw_pct)
        .join(lemma_pct)
    )

    summary.to_csv(summary_csv)
    print(f"[OK] Index match summary saved at: {summary_csv}")
    return summary


# =========================
# Main pipeline
# =========================

def main():
    # Make sure base folders exist
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    TEI_TAGGED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load NLP model
    nlp = load_nlp("la_core_web_lg")

    # 2) Extract entities from plain-text Latin poems
    extract_entities_from_texts(TEXT_FOLDERS, nlp, ENTITIES_CSV)

    # 3) Inline tagging in TEI (optional if you only need CSVs)
    if TEI_VERSES_DIR.exists():
        tag_entities_in_tei(TEI_VERSES_DIR, ENTITIES_CSV, TEI_TAGGED_DIR)
    else:
        print(f"[WARN] TEI verses folder not found: {TEI_VERSES_DIR}")

    # 4) Lemmatize entities
    lemmatize_entities(ENTITIES_CSV, ENTITIES_LEMMA_CSV, nlp)

    # 5) Load Index Nominum texts
    index_texts = load_index_texts(INDEX_PDFS)

    # 6) Check entities (raw and lemma) against Index Nominum
    check_entities_in_indices(ENTITIES_LEMMA_CSV, index_texts, ENTITIES_INDEX_CHECKED_CSV)

    # 7) Build summary table
    build_index_match_summary(ENTITIES_INDEX_CHECKED_CSV, INDEX_MATCH_SUMMARY_CSV)


if __name__ == "__main__":
    main()
