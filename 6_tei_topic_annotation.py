"""
tei_topic_annotation.py

Topic auto-annotation for TEI XML files using Soldevila-based matches.

This module takes as input one or more CSV files with topic matches at
(line-level) for each poem, and produces TEI files annotated with:

- FLATTEN mode:
    * adds @ana="#TOPIC" attributes directly on <l> elements
- STANDOFF mode:
    * inserts a <standOff> block with <spanGrp type="topics"> and
      <span> elements referencing line IDs (from/to="#lineN")
- COMBINED mode:
    * does both: flatten annotations on <l> and stand-off spans

The CSVs are assumed to be the output of a dictionary-based matching step
(e.g. 'soldevila_topic_matching.py'), with at least the following columns:

    - 'archivo_texto'       (original .txt file name)
    - 'topico'              (topic label, e.g. "HASTÍO")
    - 'numero_linea_texto'  (line number in the poem, integer)
    - 'umbral_usado'        (Levenshtein threshold actually used)
    - 'seccion_diccionario' (e.g. "DiccLine 1234", optional)

Typical TEI file naming convention:
    <Author>_<Work>_<Poem>.txt  -->  <Author>_<Work>_<Poem>_TEI.xml

Example:
    Catulo_Carmen_001.txt  -->  Catulo_Carmen_001_TEI.xml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any
from collections import defaultdict
import csv
import re
import shutil

import lxml.etree as ET


# =========================
# Configuration (edit here)
# =========================

BASE_DIR = Path("./data_root")

# CSV files with topic matches (you can override this in main() or from another script)
CSV_FILES: List[Path] = [
    BASE_DIR / "topics" / "Catullus_matches_clean.csv",
    BASE_DIR / "topics" / "Tibullus_matches_clean.csv",
    BASE_DIR / "topics" / "Propertius_matches_clean.csv",
    # BASE_DIR / "topics" / "Ovid_matches_clean.csv",
]

# TEI input and output directories
TEI_INPUT_DIR = BASE_DIR / "tei_pipeline" / "04_tei_entities_tagged"
TEI_OUTPUT_DIR_STANDOFF = BASE_DIR / "tei_pipeline" / "05_topics_standoff"
TEI_OUTPUT_DIR_FLATTEN = BASE_DIR / "tei_pipeline" / "06_topics_flatten"
TEI_OUTPUT_DIR_COMBINED = BASE_DIR / "tei_pipeline" / "07_topics_combined"

# Levenshtein (or match) threshold used to filter rows
THRESHOLD = 1

# Optional: ignore dictionary sections above this line number
# (used to discard known false positives, e.g. Soldevila's "zoophilia" tail)
IGNORE_DICTIONARY_LINES_ABOVE: int | None = 4963  # set to None to disable

# Namespace constants
TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP = {None: TEI_NS}


# =========================
# Helpers
# =========================

def tei(tag: str) -> str:
    """Qualified TEI tag."""
    return f"{{{TEI_NS}}}{tag}"


def normalize_topic(topic: str) -> str:
    """
    Replace spaces with underscores in topic labels for use in IDs and @ana.
    Example: "AMOR IMPOSIBLE" -> "AMOR_IMPOSIBLE"
    """
    return topic.replace(" ", "_")


def has_tei_namespace(root: ET._Element) -> bool:
    """Return True if the document uses the TEI default namespace."""
    return isinstance(root.tag, str) and root.tag.startswith(f"{{{TEI_NS}}}")


def qualify(tag: str, use_ns: bool) -> str:
    """Return the qualified tag name depending on whether TEI namespace is used."""
    return tei(tag) if use_ns else tag


# =========================
# 1) Read CSVs and build topic ranges
# =========================

def _parse_dictionary_line_number(section: str) -> int | None:
    """
    Extract the integer after 'DiccLine ' in a section string, e.g.:

        'DiccLine 4500' -> 4500

    Returns None if no line number is found.
    """
    if not section:
        return None
    match = re.search(r"DiccLine\s+(\d+)", section)
    if not match:
        return None
    return int(match.group(1))


def read_matches_and_group(
    csv_paths: Iterable[Path],
    threshold: int = THRESHOLD,
    ignore_dictionary_lines_above: int | None = IGNORE_DICTIONARY_LINES_ABOVE,
) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    """
    Read topic matches from one or more CSV files and return:

        topics_by_file[archivo_txt][topic] = [(start_line, end_line), ...]

    Steps:
      - Keep only rows where 'umbral_usado' <= threshold.
      - If 'ignore_dictionary_lines_above' is not None and the row has
        'seccion_diccionario', discard rows whose dictionary line number
        is greater than that value (used to remove known false positives).
      - Group lines by (archivo_texto, topico).
      - Merge contiguous line numbers into ranges (start, end).

    Required CSV columns:
      * archivo_texto
      * topico
      * numero_linea_texto
      * umbral_usado   (optional: treated as 0 if missing or not numeric)
      * seccion_diccionario (optional)
    """
    line_sets: Dict[Tuple[str, str], set[int]] = defaultdict(set)

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"[WARN] CSV not found, skipping: {csv_path}")
            continue
        print(f"[INFO] Reading matches from: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                archivo = (row.get("archivo_texto") or "").strip()
                topic = (row.get("topico") or "").strip()
                if not archivo or not topic:
                    continue

                # Threshold filtering (umbral_usado)
                umbral_str = (row.get("umbral_usado") or "").strip()
                umbral_val = int(umbral_str) if umbral_str.isdigit() else 0
                if umbral_val > threshold:
                    continue

                # Optional dictionary section filtering (e.g. Soldevila "zoophilia" tail)
                if ignore_dictionary_lines_above is not None:
                    section_str = (row.get("seccion_diccionario") or "").strip()
                    line_num = _parse_dictionary_line_number(section_str)
                    if line_num is not None and line_num > ignore_dictionary_lines_above:
                        # Skip this hit as a false positive
                        continue

                line_str = (row.get("numero_linea_texto") or "").strip()
                if not line_str.isdigit():
                    continue
                line_int = int(line_str)

                line_sets[(archivo, topic)].add(line_int)

    # Merge contiguous lines into ranges per (archivo, topic)
    range_map: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for (archivo, topic), lines in line_sets.items():
        sorted_lines = sorted(lines)
        ranges: List[Tuple[int, int]] = []
        start = None
        prev = None
        for ln in sorted_lines:
            if start is None:
                start = ln
                prev = ln
            else:
                if ln == prev + 1:
                    prev = ln
                else:
                    ranges.append((start, prev))
                    start = ln
                    prev = ln
        if start is not None:
            ranges.append((start, prev))
        range_map[(archivo, topic)] = ranges

    # Organize by archivo_txt
    topics_by_file: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for (archivo, topic), ranges in range_map.items():
        topics_by_file[archivo][topic].extend(ranges)

    print(f"[INFO] Built topic ranges for {len(topics_by_file)} input text files.")
    return topics_by_file


# =========================
# 2) TEI structure helpers
# =========================

def derive_tei_name_from_txt(txt_name: str) -> str:
    """
    Given a .txt file name like 'Catulo_Carmen_001.txt',
    derive the TEI file name used in the pipeline, e.g.:

        Catulo_Carmen_001.txt -> Catulo_Carmen_001_TEI.xml

    If the base already starts with the author name followed by '_',
    we keep it and add '_TEI.xml'.
    """
    base = txt_name.replace(".txt", "")
    author = base.split("_")[0]
    if base.startswith(author + "_"):
        return f"{base}_TEI.xml"
    return f"{author}_{base}_TEI.xml"


def flatten_tei_remove_lg(root: ET._Element) -> None:
    """
    Remove <lg> (line group) elements while preserving their <l> children.

    Each <l> child is re-inserted at the position of the <lg> in the parent,
    then <lg> is removed.
    """
    use_ns = has_tei_namespace(root)
    all_lg = root.findall(".//" + qualify("lg", use_ns))
    for lg_node in all_lg:
        parent = lg_node.getparent()
        if parent is None:
            continue
        idx = parent.index(lg_node)
        for child in list(lg_node):
            parent.insert(idx, child)
            idx += 1
        parent.remove(lg_node)


def assign_line_ids(root: ET._Element) -> None:
    """
    Assign xml:id="lineN" to each <l n="N"> element.

    This is required if you want stand-off spans pointing to #lineN.
    """
    use_ns = has_tei_namespace(root)
    all_l = root.findall(".//" + qualify("l", use_ns))
    for l_node in all_l:
        n_str = l_node.get("n")
        if n_str and n_str.isdigit():
            l_node.set(f"{{{XML_NS}}}id", f"line{n_str}")


# =========================
# 3) FLATTEN annotations (@ana on <l>)
# =========================

def apply_flatten_annotations(
    root: ET._Element,
    topic_ranges: Dict[str, List[Tuple[int, int]]],
) -> None:
    """
    For each <l> element, if its 'n' (line number) falls within any topic range,
    add or extend @ana with the topic(s) as '#TOPIC_NORMALIZED'.
    """
    # Build map: line_number -> <l> element
    l_by_num: Dict[int, ET._Element] = {}
    use_ns = has_tei_namespace(root)
    for l_el in root.findall(".//" + qualify("l", use_ns)):
        n_str = l_el.get("n")
        if n_str and n_str.isdigit():
            l_by_num[int(n_str)] = l_el

    for topic, ranges in topic_ranges.items():
        topic_norm = normalize_topic(topic)
        annotation_value = f"#{topic_norm}"
        for start_ln, end_ln in ranges:
            for ln in range(start_ln, end_ln + 1):
                l_node = l_by_num.get(ln)
                if l_node is None:
                    continue
                current_ana = (l_node.get("ana") or "").strip()
                if annotation_value not in current_ana.split():
                    new_ana = (
                        f"{current_ana} {annotation_value}".strip()
                        if current_ana
                        else annotation_value
                    )
                    l_node.set("ana", new_ana)


# =========================
# 4) STAND-OFF annotations (<standOff>/<spanGrp>/<span>)
# =========================

def build_standoff_block(
    root: ET._Element,
    topic_ranges: Dict[str, List[Tuple[int, int]]],
) -> None:
    """
    Create or replace a <standOff> block with a <spanGrp type="topics">.

    Each topic + range becomes a <span> with:
        - xml:id   = TOPIC_NORMALIZED or TOPIC_NORMALIZED-2, etc. (if multiple ranges)
        - from     = "#line{start}"
        - to       = "#line{end}"
        - ana      = "#TOPIC_NORMALIZED"
    """
    # Remove existing <standOff> if present at the TEI root level
    use_ns = has_tei_namespace(root)
    existing = root.find(qualify("standOff", use_ns))
    if existing is not None:
        root.remove(existing)

    element_kwargs = {"nsmap": NSMAP} if use_ns else {}
    stand_off = ET.Element(qualify("standOff", use_ns), **element_kwargs)
    span_grp = ET.Element(qualify("spanGrp", use_ns))
    span_grp.set("type", "topics")

    for topic, ranges in topic_ranges.items():
        topic_norm = normalize_topic(topic)
        for i, (start_ln, end_ln) in enumerate(ranges, start=1):
            if len(ranges) == 1:
                xml_id = topic_norm
            else:
                xml_id = f"{topic_norm}-{i}"
            span_el = ET.Element(qualify("span", use_ns))
            span_el.set(f"{{{XML_NS}}}id", xml_id)
            span_el.set("from", str(start_ln))
            span_el.set("to", str(end_ln))
            span_el.set("ana", f"#{topic_norm}")
            span_grp.append(span_el)

    stand_off.append(span_grp)

    # Insert standOff before <text> if present, otherwise append at the end
    text_node = root.find(qualify("text", use_ns))
    if text_node is not None:
        idx = list(root).index(text_node)
        root.insert(idx, stand_off)
    else:
        root.append(stand_off)


# =========================
# 5) Main processing over TEI files
# =========================

def annotate_tei_files(
    tei_input_dir: Path,
    tei_output_dir: Path,
    topics_by_file: Dict[str, Dict[str, List[Tuple[int, int]]]],
    mode: str = "combined",
    copy_unannotated: bool = True,
) -> None:
    """
    Annotate TEI files according to the given mode:

        mode = "flatten"   -> only @ana on <l>
        mode = "standoff"  -> only standOff/spanGrp/span
        mode = "combined"  -> both flatten + standOff, also flattens <lg>

    - tei_input_dir: directory where input TEI files live.
    - tei_output_dir: directory where annotated TEI files will be written.
    - topics_by_file: mapping
        topics_by_file[archivo_txt][topic] = [(start, end), ...]
    - copy_unannotated:
        if True, TEI files with no topics are simply copied to tei_output_dir.
    """
    tei_output_dir.mkdir(parents=True, exist_ok=True)
    processed_tei_names: set[str] = set()

    for archivo_txt, topic_ranges in topics_by_file.items():
        tei_name = derive_tei_name_from_txt(archivo_txt)
        tei_input_path = tei_input_dir / tei_name
        if not tei_input_path.exists():
            print(f"[WARN] TEI not found, skipping: {tei_input_path}")
            continue

        processed_tei_names.add(tei_name)
        tei_output_path = tei_output_dir / tei_name

        print(f"[INFO] Annotating TEI ({mode}): {tei_input_path} -> {tei_output_path}")

        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(str(tei_input_path), parser)
        root = tree.getroot()

        # For combined mode, we also flatten <lg> groups into <l> siblings.
        if mode == "combined":
            flatten_tei_remove_lg(root)
            apply_flatten_annotations(root, topic_ranges)
            build_standoff_block(root, topic_ranges)

        elif mode == "flatten":
            apply_flatten_annotations(root, topic_ranges)

        elif mode == "standoff":
            build_standoff_block(root, topic_ranges)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        tree.write(str(tei_output_path), encoding="utf-8", pretty_print=True)

    # Optionally copy TEI files without topics
    if copy_unannotated:
        print("[INFO] Copying TEI files without topic annotations...")
        for file_path in tei_input_dir.glob("*.xml"):
            if file_path.name not in processed_tei_names:
                dst = tei_output_dir / file_path.name
                shutil.copy2(file_path, dst)
                print(f"[INFO] Copied TEI without topics: {file_path} -> {dst}")


# =========================
# CLI entry point (example usage)
# =========================

def main():
    # 1) Resolve CSV files
    csv_paths = [path for path in CSV_FILES if path.exists()]
    if not csv_paths:
        fallback_filtered = BASE_DIR / "outputs" / "soldevila_matches_filtered.csv"
        fallback_raw = BASE_DIR / "outputs" / "soldevila_matches_raw.csv"
        if fallback_filtered.exists():
            csv_paths = [fallback_filtered]
            print(
                f"[INFO] CSV list empty. Using fallback filtered matches: {fallback_filtered}"
            )
        elif fallback_raw.exists():
            csv_paths = [fallback_raw]
            print(f"[INFO] CSV list empty. Using fallback raw matches: {fallback_raw}")
        else:
            print("[WARN] No topic CSV files found; skipping TEI annotation.")
            return

    # 2) Determine TEI input directory (fallback to outputs if needed)
    tei_input_dir = TEI_INPUT_DIR
    if not tei_input_dir.exists():
        fallback_tei = BASE_DIR / "outputs" / "tei_entities_tagged"
        if fallback_tei.exists():
            print(
                f"[INFO] TEI input dir '{tei_input_dir}' not found. "
                f"Using fallback: {fallback_tei}"
            )
            tei_input_dir = fallback_tei
        else:
            print(f"[WARN] TEI input directory not found: {tei_input_dir}")
            return

    # 3) Read and group topic matches from CSVs
    topics_by_file = read_matches_and_group(
        csv_paths=csv_paths,
        threshold=THRESHOLD,
        ignore_dictionary_lines_above=IGNORE_DICTIONARY_LINES_ABOVE,
    )

    if not topics_by_file:
        print("[WARN] No topic matches found; skipping TEI annotation.")
        return

    # 4) Annotate TEI files in three different modes (you can comment out what you don't need)

    # Stand-off only
    annotate_tei_files(
        tei_input_dir=tei_input_dir,
        tei_output_dir=TEI_OUTPUT_DIR_STANDOFF,
        topics_by_file=topics_by_file,
        mode="standoff",
        copy_unannotated=True,
    )

    # Flatten only (@ana on <l>)
    annotate_tei_files(
        tei_input_dir=tei_input_dir,
        tei_output_dir=TEI_OUTPUT_DIR_FLATTEN,
        topics_by_file=topics_by_file,
        mode="flatten",
        copy_unannotated=True,
    )

    # Combined (flatten + stand-off, removing <lg>)
    annotate_tei_files(
        tei_input_dir=tei_input_dir,
        tei_output_dir=TEI_OUTPUT_DIR_COMBINED,
        topics_by_file=topics_by_file,
        mode="combined",
        copy_unannotated=True,
    )

    print("\n[DONE] TEI topic annotation finished for all modes.")


if __name__ == "__main__":
    main()
