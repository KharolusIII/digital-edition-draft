"""
tei_relaxng_header_and_validation.py

Utilities to:

1) Inject an xml-model processing instruction referencing the TEI P5 Relax NG
   schema into TEI XML files, and ensure that the <TEI> root element declares
   the TEI namespace.

2) Validate a set of TEI XML files against the TEI Relax NG schema (tei_all.rng).

This module is designed to be simple and repository-friendly:
- No Google Colab or Google Drive specifics.
- Paths are configured via the CONFIG section below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
import re
import shutil

import lxml.etree as ET
import requests
from io import BytesIO


# =========================
# CONFIGURATION
# =========================

BASE_DIR = Path("./data_root")

# Input folders with TEI XML files (recursively scanned)
# Input-output targets for header update and validation
VALIDATION_TARGETS: List[Tuple[Path, str]] = [
    (BASE_DIR / "tei_pipeline" / "05_topics_standoff", "standoff"),
    (BASE_DIR / "tei_pipeline" / "06_topics_flatten", "flatten"),
    (BASE_DIR / "tei_pipeline" / "07_topics_combined", "combined"),
]

# Base output folder for TEI files with updated header + namespace
OUTPUT_TEI_BASE: Path = (
    BASE_DIR / "tei_pipeline" / "08_tei_header_with_relaxng"
)

# Relax NG schema URL for TEI P5 (tei_all.rng)
TEI_RELAXNG_URL: str = (
    "https://vault.tei-c.org/P5/current/xml/tei/custom/schema/relaxng/tei_all.rng"
)

# xml-model processing instruction to add at the top of each TEI file
XML_MODEL_PI: str = (
    '<?xml-model href="https://vault.tei-c.org/P5/current/xml/tei/custom/schema/relaxng/tei_all.rng"\n'
    '  schematypens="http://relaxng.org/ns/structure/1.0"\n'
    '  type="application/xml"?>\n'
)

# TEI namespace to enforce on the <TEI> root element
TEI_NS = "http://www.tei-c.org/ns/1.0"


# =========================
# HEADER MANIPULATION
# =========================

def strip_xml_declaration(content: str) -> Tuple[str, int]:
    """
    Remove any XML declaration like:
        <?xml version="1.0" encoding="UTF-8"?>
    (either with double or single quotes).

    Returns (new_content, num_replacements).
    """
    content, count1 = re.subn(
        r'<\?xml version="1\.0" encoding="[^"]+"\?>\s*', "", content
    )
    content, count2 = re.subn(
        r"<\?xml version='1\.0' encoding='[^']+'\?>\s*", "", content
    )
    return content, count1 + count2


def ensure_xml_model_pi(content: str, xml_model_pi: str = XML_MODEL_PI) -> str:
    """
    Ensure that the content starts with the given xml-model processing
    instruction (after stripping leading whitespace).

    If an xml-model is already present at the beginning, do nothing.
    Otherwise, prepend xml_model_pi.
    """
    stripped = content.lstrip()
    if stripped.startswith("<?xml-model"):
        # Already present
        return content
    return xml_model_pi + content


def ensure_tei_namespace(content: str, tei_ns: str = TEI_NS) -> str:
    """
    Ensure that the root <TEI> element declares the TEI namespace:
        xmlns="http://www.tei-c.org/ns/1.0"

    If this xmlns is already present anywhere in the document, do nothing.
    Otherwise, inject it into the first <TEI ...> tag.
    """
    if f'xmlns="{tei_ns}"' in content:
        # Namespace already declared somewhere
        return content

    # Insert xmlns into the first <TEI ...> occurrence
    pattern = r"<TEI(\s|>)"
    replacement = rf'<TEI xmlns="{tei_ns}"\1'
    new_content, count = re.subn(pattern, replacement, content, count=1)
    if count == 0:
        # No <TEI> found; warn via print (or logging) and return original content
        print("[WARN] No <TEI> root element found to insert xmlns.")
        return content
    return new_content


def process_single_tei_file(
    input_path: Path,
    output_path: Path,
    xml_model_pi: str = XML_MODEL_PI,
    tei_ns: str = TEI_NS,
) -> bool:
    """
    Read a TEI XML file, strip XML declaration (if any), ensure the xml-model PI
    and TEI namespace declaration on <TEI>, then write the result to output_path.

    Returns True if the file was processed successfully, False otherwise.
    """
    try:
        text = input_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Could not read {input_path}: {e}")
        return False

    original_len = len(text)
    print(f"[DEBUG] {input_path.name}: original content length = {original_len} chars")

    # 1) Remove XML declaration
    text, removed = strip_xml_declaration(text)
    print(f"[DEBUG] {input_path.name}: stripped XML declaration (removed={removed})")

    # 2) Ensure xml-model PI
    text = ensure_xml_model_pi(text, xml_model_pi=xml_model_pi)

    # 3) Ensure TEI namespace on root <TEI>
    text = ensure_tei_namespace(text, tei_ns=tei_ns)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"[INFO] Updated header in: {input_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Could not write {output_path}: {e}")
        return False


def process_tei_folder(
    input_folder: Path,
    output_folder: Path,
) -> int:
    """
    Process all .xml TEI files recursively in the given input folder and
    write updated versions (with xml-model + xmlns) to output_folder.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    files_processed = 0

    print(f"[INFO] Processing folder: {input_folder}")
    if not input_folder.exists():
        print(f"[WARN] Folder does not exist: {input_folder}")
        return 0

    xml_files = list(input_folder.rglob("*.xml"))
    print(f"[DEBUG] Found {len(xml_files)} XML file(s) in {input_folder}")

    for input_path in xml_files:
        output_path = output_folder / input_path.name
        ok = process_single_tei_file(input_path, output_path)
        if ok:
            files_processed += 1

    print(f"[INFO] Finished header update for {input_folder}. Files processed: {files_processed}")
    return files_processed


# =========================
# RELAX NG VALIDATION
# =========================

def load_relaxng_schema(schema_url: str = TEI_RELAXNG_URL) -> ET.RelaxNG:
    """
    Load a Relax NG schema from a URL using requests and return an lxml.etree.RelaxNG object.
    """
    print(f"[INFO] Downloading Relax NG schema from: {schema_url}")
    response = requests.get(schema_url)
    response.raise_for_status()

    schema_doc = ET.parse(BytesIO(response.content))
    schema = ET.RelaxNG(schema_doc)
    print("[INFO] Relax NG schema loaded successfully.")
    return schema


def validate_tei_files(
    folders: Iterable[Path],
    schema: ET.RelaxNG,
    errors_output: Path,
) -> int:
    """
    Validate all .xml files found recursively under 'folders' against the given
    Relax NG schema.

    Writes a text report of validation errors to errors_output.

    Returns the number of files with validation errors (including syntax errors).
    """
    errores_validacion: List[Tuple[Path, str]] = []

    for folder in folders:
        if not folder.exists():
            print(f"[WARN] Folder does not exist (validation): {folder}")
            continue

        print(f"[INFO] Validating XML files in: {folder}")
        for xml_file in folder.rglob("*.xml"):
            try:
                doc = ET.parse(str(xml_file))
            except ET.XMLSyntaxError as e:
                msg = f"XML syntax error: {e}"
                errores_validacion.append((xml_file, msg))
                continue

            if not schema.validate(doc):
                errores_validacion.append((xml_file, str(schema.error_log)))
            else:
                print(f"[OK] {xml_file} is valid.")

    errors_output.parent.mkdir(parents=True, exist_ok=True)
    with errors_output.open("w", encoding="utf-8") as f:
        for file_path, error in errores_validacion:
            f.write(f"File: {file_path}\n")
            f.write("Errors:\n")
            f.write(error)
            f.write("\n" + "-" * 40 + "\n")

    if errores_validacion:
        print(
            f"[INFO] Validation finished with errors. "
            f"Report saved to '{errors_output}'."
        )
    else:
        print("[INFO] All files are valid against the Relax NG schema.")

    return len(errores_validacion)


# =========================
# CLI ENTRY POINT
# =========================

def main():
    # Load Relax NG schema once
    try:
        schema = load_relaxng_schema(TEI_RELAXNG_URL)
    except Exception as e:
        print(f"[ERROR] Could not load Relax NG schema: {e}")
        return

    for input_folder, label in VALIDATION_TARGETS:
        output_folder = OUTPUT_TEI_BASE / label
        process_tei_folder(
            input_folder=input_folder,
            output_folder=output_folder,
        )

        errors_output = OUTPUT_TEI_BASE / f"{label}_validation_errors.txt"
        validate_tei_files(
            folders=[output_folder],
            schema=schema,
            errors_output=errors_output,
        )


if __name__ == "__main__":
    main()
