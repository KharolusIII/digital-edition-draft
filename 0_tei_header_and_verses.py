# tei_header_and_verses.py
"""
Generate basic TEI files (header + body) from plain text Latin poems,
split verses into <lg>/<l>, fix XML headers, and validate against TEI P5.

This script is intentionally simple and self-contained so it can be used
both locally and in platforms like Google Colab (with paths adapted by the user).
"""

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET_ElementTree
import re
import requests
from io import BytesIO
import lxml.etree as ET


# =========================
# Configuration (edit here)
# =========================

# Root directory for input and output (change this on your machine)
BASE_DIR = Path("./data_root")

# Input folders with plain-text poems (one poem per file)
# If these folders do not exist (e.g., sample data has files directly under BASE_DIR),
# the script will fall back to files whose names start with the author prefix
# such as "Catulo_*.txt".
INPUT_FOLDERS = [
    BASE_DIR / "Catulo",
    BASE_DIR / "Tibulo",
    BASE_DIR / "Propercio",
]

# Output folders for each stage
OUTPUT_HEADER_DIR = BASE_DIR / "tei_pipeline" / "02_tei_header_generated"
OUTPUT_VERSES_DIR = BASE_DIR / "tei_pipeline" / "03_tei_verses_counted"

# Folders with TEI files that already contain topic markup (for header fix)
TOPIC_TEI_FOLDERS = [
    BASE_DIR / "tei_pipeline" / "07_topics_combined",
    BASE_DIR / "tei_pipeline" / "05_topics_standoff",
    BASE_DIR / "tei_pipeline" / "06_topics_flatten",
]

# Folder where we will store TEI files with fixed XML header
OUTPUT_HEADER_FIXED_DIR = BASE_DIR / "outputs" / "tei_header_fixed"

# Validation error log
VALIDATION_ERROR_LOG = BASE_DIR / "outputs" / "tei_validation_errors.txt"

# TEI RelaxNG schema URL
TEI_RELAXNG_URL = "https://vault.tei-c.org/P5/current/xml/tei/custom/schema/relaxng/tei_all.rng"


# Author full names (Latin)
FULL_NAMES = {
    "Catulo": "Gaius Valerius Catullus",
    "Tibulo": "Publilius Tibullus",
    "Propercio": "Sextus Propertius",
}

# Bibliographic details of the editions (as described in the article)
EDITIONS = {
    "Catulo": {
        "bibl_author": "Merrill, E. T.",
        "date": "1893",
        "bibl_title": "Catullus; edited by Elmer Truesdell Merrill",
        "publisher": "Boston Ginn",
        "url": "http://archive.org/details/catulluseditedby00catuuoft",
    },
    "Tibulo": {
        "bibl_author": "Postgate, J. P.",
        "date": "1915",
        "bibl_title": "Tibulli aliorumque carminum libri tres. Scriptorum classicorum bibliotheca Oxoniensis",
        "publisher": "Scriptorum classicorum bibliotheca Oxoniensis",
        "url": "https://archive.org/details/tibullialiorumqu00tibuuoft",
    },
    "Propercio": {
        "bibl_author": "Müller, L.",
        "date": "1898",
        "bibl_title": "Sex. Propertii Elegiae",
        "publisher": "Teubner",
        "url": "https://archive.org/details/elegiaerecensuit00propuoft",
    },
}

# Editors and project name – you can anonymize these if needed
EDITORS = [
    "Editor 1",
    "Editor 2",
    "Editor 3",
]
PROJECT_NAME = "Digital Latin Poetry Project"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
LICENSE_LABEL = "CC BY 4.0"


# =========================
# Utility helpers
# =========================

def _resolve_text_files(folder: Path) -> list[Path]:
    """
    Return a list of .txt files for the requested folder.
    If the folder does not exist (e.g., sample data is flat), look for files
    whose names start with the folder name (Catulo_*.txt).
    """
    if folder.exists() and folder.is_dir():
        files = sorted(folder.glob("*.txt"))
        if files:
            return files

    # Fall back to files with prefix under BASE_DIR (folder.parent)
    fallback = sorted(folder.parent.glob(f"{folder.name}_*.txt"))
    if fallback:
        print(
            f"[INFO] Folder '{folder}' not found. Using {len(fallback)} "
            f"files matching pattern '{folder.name}_*.txt' under {folder.parent}"
        )
    return fallback


# =========================
# TEI header generation
# =========================

def create_header(poem_title: str, folder_author: str, edition_details: dict) -> Element:
    """
    Create a <teiHeader> element for a given poem and author.
    """
    header = Element("teiHeader")
    file_desc = SubElement(header, "fileDesc")

    # titleStmt: title and author
    title_stmt = SubElement(file_desc, "titleStmt")

    # Create xml:id from the poem title, removing the author prefix
    raw_title = poem_title.replace("_", " ")
    if raw_title.lower().startswith(folder_author.lower()):
        remaining = raw_title[len(folder_author):].lstrip()
    else:
        remaining = raw_title
    xml_id_value = "".join(remaining.lower().split())

    title_el = SubElement(title_stmt, "title")
    title_el.set("xml:id", xml_id_value)
    title_el.text = remaining

    author_el = SubElement(title_stmt, "author")
    author_el.text = FULL_NAMES.get(folder_author, folder_author)

    # Responsibility statements (digital edition editors)
    for ed in EDITORS:
        resp_stmt = SubElement(title_stmt, "respStmt")
        name_el = SubElement(resp_stmt, "name")
        name_el.text = ed
        resp_el = SubElement(resp_stmt, "resp")
        resp_el.text = "Digital edition"

    # publicationStmt: publisher and license
    publication_stmt = SubElement(file_desc, "publicationStmt")
    publisher_el = SubElement(publication_stmt, "publisher")
    publisher_el.text = PROJECT_NAME

    availability_el = SubElement(publication_stmt, "availability")
    licence_el = SubElement(availability_el, "licence")
    ref_el = SubElement(licence_el, "ref")
    ref_el.set("target", LICENSE_URL)
    ref_el.text = LICENSE_LABEL

    # sourceDesc: bibliographic source
    source_desc = SubElement(file_desc, "sourceDesc")
    bibl = SubElement(source_desc, "bibl")

    bibl_author = SubElement(bibl, "author")
    bibl_author.text = edition_details["bibl_author"]

    bibl_date = SubElement(bibl, "date")
    bibl_date.text = edition_details["date"]

    bibl_title = SubElement(bibl, "title")
    bibl_title.text = edition_details["bibl_title"]

    if edition_details.get("publisher"):
        bibl_publisher = SubElement(bibl, "publisher")
        bibl_publisher.text = edition_details["publisher"]

    bibl_ptr = SubElement(bibl, "ptr")
    bibl_ptr.set("target", edition_details["url"])

    # encodingDesc: application / model information
    encoding_desc = SubElement(header, "encodingDesc")
    app_info = SubElement(encoding_desc, "appInfo")

    # TEI order: version first, then ident (no slashes)
    application = SubElement(
        app_info,
        "application",
        version="3.8.0",
        ident="latincy_la_core_web_lg",
    )

    label_el = SubElement(application, "label")
    label_el.text = "la_core_web_lg v3.8.0"

    ptr_el = SubElement(application, "ptr")
    ptr_el.set("target", "https://huggingface.co/latincy/la_core_web_lg")

    return header


def generate_tei_files(input_folders: list[Path], output_folder: Path) -> None:
    """
    Generate basic TEI files (header + <text><body><p>) from plain text poems.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for folder in input_folders:
        files = _resolve_text_files(folder)
        if not files:
            print(f"[WARN] No .txt files found for: {folder}")
            continue

        folder_author = folder.name
        edition_details = EDITIONS.get(folder_author)
        if not edition_details:
            print(f"[ERROR] No edition details for author '{folder_author}'")
            continue

        for file_path in files:
            try:
                poem_title = file_path.stem
                raw_title = poem_title.replace("_", " ")
                if raw_title.lower().startswith(folder_author.lower()):
                    clean_title = raw_title[len(folder_author):].lstrip().replace(" ", "_")
                else:
                    clean_title = poem_title

                text = file_path.read_text(encoding="utf-8")

                tei = Element("TEI")
                header = create_header(poem_title, folder_author, edition_details)
                tei.append(header)

                text_el = SubElement(tei, "text")
                body = SubElement(text_el, "body")
                p = SubElement(body, "p")
                p.text = text

                output_file_path = output_folder / f"{folder_author}_{clean_title}_TEI.xml"
                tree = ElementTree(tei)
                tree.write(output_file_path, encoding="utf-8", xml_declaration=True)
                print(f"[OK] TEI file generated: {output_file_path}")
            except Exception as e:
                print(f"[ERROR] Failed processing {file_path}: {e}")


# =========================
# Verse counting (<lg>/<l>)
# =========================

def count_verses_and_update_tei(input_file_path: Path, output_file_path: Path) -> bool:
    """
    Read a TEI file with <body><p>text</p>, split lines into verses,
    and create <lg type="poema"><l n="...">verse</l></lg>.
    Returns True if verses were created.
    """
    print(f"Processing file: {input_file_path}")
    tree = ET_ElementTree.parse(input_file_path)
    root = tree.getroot()
    body = root.find(".//body")

    if body is None:
        print(f"No <body> in {input_file_path}.")
        return False

    paragraph = body.find(".//p")
    if paragraph is None:
        print(f"No <p> in {input_file_path}. Creating empty container.")
        paragraph = SubElement(body, "p")
        text = ""
    else:
        text = paragraph.text or ""

    verses = text.split("\n")
    verse_count = 0
    lg_container = SubElement(body, "lg", {"type": "poema"})

    for verse in verses:
        if verse.strip():
            verse_count += 1
            l_el = SubElement(lg_container, "l", {"n": str(verse_count)})
            l_el.text = verse
        # empty lines are ignored

    # Remove original paragraph
    body.remove(paragraph)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_file_path, encoding="utf-8", xml_declaration=True)
    print(f"Updated and saved: {output_file_path}")
    return verse_count > 0


def process_all_tei_for_verses(input_folder: Path, output_folder: Path) -> None:
    """
    Apply verse counting to all TEI files in a folder.
    """
    files_processed = 0
    files_updated = 0
    files_without_verses = []

    for tei_file in input_folder.glob("*.xml"):
        output_file = output_folder / tei_file.name
        processed = count_verses_and_update_tei(tei_file, output_file)
        files_processed += 1
        if processed:
            files_updated += 1
        else:
            files_without_verses.append(tei_file.name)

    print("\nSummary:")
    print(f"Total files processed: {files_processed}")
    print(f"Files with verses created: {files_updated}")
    if files_without_verses:
        print("Files without verses:")
        for fname in files_without_verses:
            print(f"  - {fname}")
    else:
        print("All files were updated with verses.")


# =========================
# Fix XML header and xmlns
# =========================

XML_MODEL_HEADER = """<?xml-model href="https://vault.tei-c.org/P5/current/xml/tei/custom/schema/relaxng/tei_all.rng"
  schematypens="http://relaxng.org/ns/structure/1.0"
  type="application/xml"?>
"""

def fix_xml_header_in_folder(input_folders: list[Path], output_folder: Path) -> None:
    """
    Remove XML declaration, add xml-model header if missing,
    and ensure <TEI> has the TEI namespace.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    files_processed = 0

    for tei_folder in input_folders:
        print(f"[INFO] Processing folder: {tei_folder}")
        if not tei_folder.exists():
            print(f"[WARN] Folder does not exist: {tei_folder}")
            continue

        for file_path in tei_folder.rglob("*.xml"):
            print(f"[DEBUG] File: {file_path.name}")
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] Could not read {file_path.name}: {e}")
                continue

            # Remove existing XML declaration
            content, count1 = re.subn(
                r'<\?xml version="1\.0" encoding="[^"]+"\?>\s*', "", content
            )
            content, count2 = re.subn(
                r"<\?xml version='1\.0' encoding='[^']+'\?>\s*", "", content
            )

            # Prepend xml-model header if missing
            if not content.lstrip().startswith("<?xml-model"):
                content = XML_MODEL_HEADER + content

            # Ensure TEI namespace on <TEI>
            content, count3 = re.subn(
                r"<TEI(\s|>)",
                r'<TEI xmlns="http://www.tei-c.org/ns/1.0"\1',
                content,
                count=1,
            )

            output_file_path = output_folder / file_path.name
            try:
                output_file_path.write_text(content, encoding="utf-8")
                files_processed += 1
            except Exception as e:
                print(f"[ERROR] Could not write {file_path.name}: {e}")

    print(f"\n[INFO] Header fix finished. Files processed: {files_processed}")


# =========================
# TEI validation (RelaxNG)
# =========================

def validate_tei_folders(folders: list[Path], error_log_path: Path) -> None:
    """
    Validate all TEI files in the given folders against the TEI RelaxNG schema.
    Save validation errors to a text file.
    """
    try:
        response = requests.get(TEI_RELAXNG_URL)
        response.raise_for_status()
        schema_doc = ET.parse(BytesIO(response.content))
        schema = ET.RelaxNG(schema_doc)
    except Exception as e:
        print(f"Error loading schema: {e}")
        return

    errors = []

    for folder in folders:
        for xml_file in folder.rglob("*.xml"):
            try:
                doc = ET.parse(str(xml_file))
            except ET.XMLSyntaxError as e:
                errors.append((xml_file, f"Syntax error: {e}"))
                continue

            if not schema.validate(doc):
                errors.append((xml_file, schema.error_log))
            else:
                print(f"{xml_file} is valid.")

    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with error_log_path.open("w", encoding="utf-8") as f:
        for archivo, error in errors:
            f.write(f"File: {archivo}\n")
            f.write("Errors:\n")
            f.write(str(error))
            f.write("\n" + "-" * 40 + "\n")

    if errors:
        print(f"Validation errors saved to: {error_log_path}")
    else:
        print("All files are valid.")


# =========================
# Main entry point
# =========================

def main():
    # 1) Generate TEI header + body
    generate_tei_files(INPUT_FOLDERS, OUTPUT_HEADER_DIR)

    # 2) Count verses and create <lg>/<l>
    process_all_tei_for_verses(OUTPUT_HEADER_DIR, OUTPUT_VERSES_DIR)

    # 3) (Optional) Fix XML header and namespace for topic-marked TEI
    #    Uncomment if you already have TEI files with topic annotations.
    # fix_xml_header_in_folder(TOPIC_TEI_FOLDERS, OUTPUT_HEADER_FIXED_DIR)

    # 4) (Optional) Validate TEI files
    # validate_tei_folders([OUTPUT_VERSES_DIR], VALIDATION_ERROR_LOG)


if __name__ == "__main__":
    main()
