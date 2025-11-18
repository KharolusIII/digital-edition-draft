"""
tei_visualizations.py

Visualizations for the "De Catulo a Wikidata" TEI pipeline.

Includes:

1) Topic bar charts per author (from <standOff>/<spanGrp type="topicos">).
2) Topic search: list TEI files where a given topic appears.
3) Topic co-occurrence graphs per author.
4) Entity co-occurrence graphs per author from TEI-XML entities
   (<persName>, <placeName>, groups in persName[@type="group"]).
5) Entity co-occurrence graphs per author from the lemmatized entities CSV.

All code is written to be repository-friendly:
- No Google Colab or Google Drive dependencies.
- All paths configurable in the CONFIG section below.
- Plots are produced with matplotlib + networkx.

You can import this module from a notebook and call the functions
you need, or adapt the `main()` entry point at the bottom.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Any, Optional

import html

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
from lxml import etree as ET


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("./data_root")

# Folder with TEI files that already have:
# - topics encoded in standOff/spanGrp/span
# - entities encoded as persName/placeName in the text.
TEI_FOLDER = BASE_DIR / "tei_pipeline" / "08_tei_header_with_relaxng" / "combined"

# CSV with lemmatized entities (one row per entity occurrence)
LEMMA_ENTITIES_CSV = BASE_DIR / "entities" / "entities_lemmatized.csv"

# Where to store visual outputs (plots, CSV summaries, etc.)
VIS_OUTPUT_DIR = BASE_DIR / "tei_pipeline" / "09_tei_visualizations"
PLOTS_DIR = VIS_OUTPUT_DIR / "plots"
TOPIC_BAR_PLOTS_DIR = PLOTS_DIR / "topic_bars"
TOPIC_GRAPH_PLOTS_DIR = PLOTS_DIR / "topic_graphs"
ENTITY_GRAPH_PLOTS_DIR = PLOTS_DIR / "entity_graphs"
DATA_OUTPUTS_DIR = VIS_OUTPUT_DIR / "data"
TOPIC_QUERY_OUTPUT_CSV = DATA_OUTPUTS_DIR / "topic_query_results.csv"

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Colors per author (based on filename prefix)
AUTHOR_COLORS: Dict[str, str] = {
    "Catulo": "salmon",
    "Tibulo": "#d87d7d",
    "Propercio": "turquoise",
}
DEFAULT_COLOR = "gray"

# Optional manual overrides for entity types (used in graphs)
MANUAL_ENTITY_TYPE_OVERRIDES: Dict[str, str] = {
    "Lesbia": "person",
    "Cynthia": "person",
}


# ============================================================
# COMMON HELPERS
# ============================================================

def normalize_topic(topic: str) -> str:
    """
    Normalize a topic string for use as an identifier:
    - remove leading '#'
    - strip whitespace
    - replace internal spaces with underscores
    """
    if not topic:
        return ""
    topic = topic.strip()
    if topic.startswith("#"):
        topic = topic[1:]
    return topic.replace(" ", "_")


def get_author_from_filename(path: Path) -> str:
    """
    Extract author from the file name by taking the first token
    before '_' in the stem. Example:
        'Catulo_Carmen_001_TEI.xml' -> 'Catulo'
    """
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def get_poem_title_from_tei(root: ET._Element, fallback: str) -> str:
    """
    Extract poem title from:
        //teiHeader/fileDesc/titleStmt/title
    or fall back to the provided string.
    """
    titles = root.xpath(".//teiHeader/fileDesc/titleStmt/title", namespaces=NS)
    if titles and titles[0].text:
        return titles[0].text.strip()
    return fallback


def ensure_outputs_dir() -> None:
    """
    Ensure that all visualization output folders exist.
    """
    for path in [
        VIS_OUTPUT_DIR,
        PLOTS_DIR,
        TOPIC_BAR_PLOTS_DIR,
        TOPIC_GRAPH_PLOTS_DIR,
        ENTITY_GRAPH_PLOTS_DIR,
        DATA_OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) TOPIC COUNTS (BAR CHARTS)
# ============================================================

def extract_topics_from_standoff(root: ET._Element) -> List[str]:
    """
    Extract all topics from <standOff><spanGrp type="topicos"><span ...>.

    The topic is taken from:
       - @ana, if present (expected format '#TOPIC_NAME'), or
       - @type as a fallback.

    Returns a list of normalized topic identifiers.
    """
    span_grps = root.xpath(
        ".//tei:standOff//tei:spanGrp[@type='topicos' or @type='topics']",
        namespaces=NS,
    )
    if not span_grps:
        return []

    spans: List[ET._Element] = []
    for group in span_grps:
        spans.extend(group.xpath(".//tei:span", namespaces=NS))

    topics: List[str] = []

    for span in spans:
        val = span.get("ana") or span.get("type") or ""
        val = normalize_topic(val)
        if val:
            topics.append(val)

    return topics


def aggregate_topic_counts_by_author(
    tei_folder: Path,
    filename_filter: Optional[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    For each TEI file (optionally filtered by 'filename_filter'),
    read topics from standOff, count them per file, and then aggregate per author.

    Returns:
        aggregated_counts[author][topic] = total_count
    """
    aggregated_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    tei_files = [
        f
        for f in tei_folder.glob("*.xml")
        if not filename_filter or filename_filter in f.name
    ]
    filter_msg = filename_filter if filename_filter else "* (all files)"
    print(f"[INFO] Found {len(tei_files)} TEI files matching '{filter_msg}'.")

    for file_path in tei_files:
        print(f"[DEBUG] Processing file: {file_path.name}")
        try:
            tree = ET.parse(str(file_path))
        except Exception as e:
            print(f"[ERROR] Could not parse {file_path.name}: {e}")
            continue

        root = tree.getroot()
        author = get_author_from_filename(file_path)
        topics = extract_topics_from_standoff(root)
        if not topics:
            print(f"[WARN] No topics found in {file_path.name}")
            continue

        # Count topics in this file
        file_counts: Dict[str, int] = defaultdict(int)
        for t in topics:
            file_counts[t] += 1

        # Accumulate per author
        for topic, count in file_counts.items():
            aggregated_counts[author][topic] += count
        print(f"[DEBUG] Counts in {file_path.name}: {dict(file_counts)}")

    return aggregated_counts


def plot_topic_bars_by_author(
    aggregated_counts: Dict[str, Dict[str, int]],
    output_dir: Optional[Path] = None,
) -> None:
    """
    For each author in aggregated_counts, draw a horizontal bar chart
    of topics sorted by frequency (descending).
    """
    for author, topics_dict in aggregated_counts.items():
        if not topics_dict:
            print(f"[WARN] No topics for author '{author}'")
            continue

        sorted_topics = sorted(
            topics_dict.items(), key=lambda x: x[1], reverse=True
        )
        topics, counts = zip(*sorted_topics)

        color = AUTHOR_COLORS.get(author, DEFAULT_COLOR)

        plt.figure(figsize=(12, 16))
        plt.barh(topics, counts, color=color)
        plt.xlabel("Occurrences", fontsize=9)
        plt.title(f"Topics in poems by {author}", fontsize=9)
        plt.ylim(-0.5, len(topics) - 0.5)
        plt.gca().invert_yaxis()
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        print(f"[DEBUG] Plotting topics for {author}: {topics} -> {counts}")

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{author}_topic_bars.png"
            plt.savefig(output_path, dpi=300)
            print(f"[OK] Topic bar chart saved: {output_path}")
            plt.close()
        else:
            plt.show()


# ============================================================
# 2) TOPIC SEARCH (BY NAME)
# ============================================================

def find_topic_in_standoff_files(
    tei_folder: Path,
    topic_name: str,
    filename_filter: Optional[str] = None,
    output_csv: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Search for a given topic in all TEI files whose names contain 'filename_filter'.

    Topic matching is done on the normalized topic string as extracted from
    @ana or @type.

    Returns a list of dicts:
      { "file": filename, "occurrences": n }

    If output_csv is provided, the results are also saved to a CSV.
    """
    ensure_outputs_dir()

    topic_norm = normalize_topic(topic_name)
    print(f"[INFO] Searching for topic: '{topic_norm}'")

    tei_files = [
        f
        for f in tei_folder.glob("*.xml")
        if not filename_filter or filename_filter in f.name
    ]
    filter_msg = filename_filter if filename_filter else "* (all files)"
    print(f"[INFO] Found {len(tei_files)} TEI files with '{filter_msg}'.")

    results: List[Dict[str, Any]] = []

    for file_path in tei_files:
        try:
            tree = ET.parse(str(file_path))
        except Exception as e:
            print(f"[ERROR] Could not parse {file_path.name}: {e}")
            continue

        root = tree.getroot()
        spans = root.xpath(
            ".//tei:standOff//tei:spanGrp[@type='topicos']//tei:span",
            namespaces=NS,
        )

        count = 0
        for span in spans:
            val = span.get("ana") or span.get("type") or ""
            if normalize_topic(val) == topic_norm:
                count += 1

        if count > 0:
            print(f"[MATCH] Topic '{topic_norm}' appears in {file_path.name} ({count} spans)")
            results.append({"file": file_path.name, "occurrences": count})

    if output_csv is not None:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Saved topic search results to: {output_csv}")

    return results


# ============================================================
# 3) TOPIC CO-OCCURRENCE GRAPHS
# ============================================================

def extract_topic_data_from_tei(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract basic topic data from a TEI file:

    Returns:
      {
        "author": author_name,
        "poem": poem_title,
        "topics": [topic1, topic2, ...]  (normalized)
      }
    or None if parsing fails or no topics are found.
    """
    try:
        tree = ET.parse(str(file_path))
    except Exception as e:
        print(f"[ERROR] Could not parse {file_path.name}: {e}")
        return None

    root = tree.getroot()
    topics = extract_topics_from_standoff(root)
    if not topics:
        print(f"[WARN] No topics found in {file_path.name}")
        return None

    author = get_author_from_filename(file_path)
    poem = get_poem_title_from_tei(root, fallback=file_path.stem)

    return {"author": author, "poem": poem, "topics": topics}


def group_entries_by_author(
    entries: Iterable[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group generic entries (with 'author' key) by author.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        author = entry.get("author", "UNKNOWN")
        groups[author].append(entry)
    return groups


def build_author_topic_cooccurrence_graph(
    entries: List[Dict[str, Any]]
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build a topic co-occurrence graph for a single author.

    Nodes: topics (string).
    Node attribute:
      - 'count' = number of poems where the topic appears.
      - 'type'  = 'topic'.
    Edges: co-occurrence of topics in the same poem.
    Edge attribute:
      - 'weight' = number of poems where both topics co-occur.
    """
    G = nx.Graph()
    topic_freq: Dict[str, int] = defaultdict(int)

    for entry in entries:
        topics_set = set(entry.get("topics", []))  # unique topics per poem
        for t in topics_set:
            topic_freq[t] += 1

        topics_list = list(topics_set)
        n = len(topics_list)
        for i in range(n):
            for j in range(i + 1, n):
                t1, t2 = topics_list[i], topics_list[j]
                if G.has_edge(t1, t2):
                    G[t1][t2]["weight"] += 1
                else:
                    G.add_edge(t1, t2, weight=1)

    for t, freq in topic_freq.items():
        if t not in G.nodes:
            G.add_node(t)
        G.nodes[t]["count"] = freq
        G.nodes[t]["type"] = "topic"

    return G, topic_freq


def draw_topic_graph(
    G: nx.Graph,
    author: str,
    node_threshold: int = 2,
    edge_threshold: int = 1,
    k: float = 5.0,
    figsize: Tuple[int, int] = (12, 10),
    font_size: int = 12,
    node_size_multiplier: int = 300,
    edge_color: str = "lightgray",
    output_dir: Optional[Path] = None,
) -> None:
    """
    Draw a topic co-occurrence graph:

    - Keep only nodes with 'count' >= node_threshold.
    - Keep only edges with 'weight' >= edge_threshold.
    - Use a spring layout with parameter k.
    - Node size is proportional to 'count'.
    """
    nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "topic" and d.get("count", 0) >= node_threshold
    ]
    if not nodes:
        print(f"[WARN] No topics with count >= {node_threshold} for {author}")
        return

    G_filtered = G.subgraph(nodes).copy()
    edges_to_draw = [
        (u, v)
        for u, v, d in G_filtered.edges(data=True)
        if d.get("weight", 1) >= edge_threshold
    ]

    pos = nx.spring_layout(G_filtered, k=k, seed=42)

    color = AUTHOR_COLORS.get(author, DEFAULT_COLOR)
    node_sizes = [
        G_filtered.nodes[n].get("count", 1) * node_size_multiplier
        for n in G_filtered.nodes()
    ]
    labels = {
        n: f"{n}\n({G_filtered.nodes[n].get('count', 0)})"
        for n in G_filtered.nodes()
    }

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        G_filtered,
        pos,
        node_size=node_sizes,
        node_color=color,
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edgelist=edges_to_draw,
        edge_color=edge_color,
        width=1.5,
        alpha=0.7,
    )
    nx.draw_networkx_labels(
        G_filtered,
        pos,
        labels=labels,
        font_size=font_size,
    )

    plt.title(
        f"Topic Graph for {author} "
        f"(nodes count ≥ {node_threshold}, edges weight ≥ {edge_threshold})",
        fontsize=18,
    )
    plt.axis("off")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"{author}_topic_graph_nodes{node_threshold}_edges{edge_threshold}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"[OK] Topic graph saved: {output_path}")
        plt.close()
    else:
        plt.show()


def build_topic_graphs_for_all_authors(
    tei_folder: Path,
    filename_filter: Optional[str] = None,
) -> Dict[str, Tuple[nx.Graph, Dict[str, int]]]:
    """
    Build topic co-occurrence graphs for all authors:

    Returns:
      graphs[author] = (G_topic, topic_freq)
    """
    tei_files = [
        f
        for f in tei_folder.glob("*.xml")
        if not filename_filter or filename_filter in f.name
    ]
    filter_msg = filename_filter if filename_filter else "* (all files)"
    print(f"[INFO] Found {len(tei_files)} TEI files with '{filter_msg}'.")

    entries: List[Dict[str, Any]] = []
    for file_path in tei_files:
        data = extract_topic_data_from_tei(file_path)
        if data:
            entries.append(data)

    grouped = group_entries_by_author(entries)
    graphs: Dict[str, Tuple[nx.Graph, Dict[str, int]]] = {}

    for author, author_entries in grouped.items():
        G_topic, topic_freq = build_author_topic_cooccurrence_graph(author_entries)
        graphs[author] = (G_topic, topic_freq)
        total = sum(topic_freq.values())
        print(f"[INFO] Author: {author} - Total topic occurrences: {total}")

    return graphs


# ============================================================
# 4) ENTITY CO-OCCURRENCE GRAPHS FROM TEI
# ============================================================

def extract_entities_from_tei_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract entities from a TEI file using real TEI markup:

    - <persName>          -> 'person' (unless @type == 'group')
    - <persName type="group"> -> 'group'
    - <placeName>         -> 'place'

    Returns:
      {
        "author": author_name,
        "poem": poem_title,
        "entities": {
            "person": [names...],
            "group": [names...],
            "place": [names...],
        }
      }
    or None if parsing fails.
    """
    try:
        tree = ET.parse(str(file_path))
    except Exception as e:
        print(f"[ERROR] Could not parse {file_path.name}: {e}")
        return None

    root = tree.getroot()
    author = get_author_from_filename(file_path)
    poem = get_poem_title_from_tei(root, fallback=file_path.stem)

    people: List[str] = []
    groups: List[str] = []
    places: List[str] = []

    # All verse lines
    lines = root.xpath(".//tei:text/tei:body//tei:l", namespaces=NS)
    for l_el in lines:
        # Persons
        for p_el in l_el.xpath(".//tei:persName", namespaces=NS):
            text = "".join(p_el.itertext()).strip()
            if not text:
                continue
            if p_el.get("type") == "group":
                groups.append(text)
            else:
                people.append(text)
        # Places
        for pl_el in l_el.xpath(".//tei:placeName", namespaces=NS):
            text = "".join(pl_el.itertext()).strip()
            if not text:
                continue
            places.append(text)

    entities = {"person": people, "group": groups, "place": places}
    return {"author": author, "poem": poem, "entities": entities}


def build_author_entity_graph(
    entries: List[Dict[str, Any]]
) -> Tuple[nx.Graph, Dict[Tuple[str, str], int]]:
    """
    Build an entity co-occurrence graph for a single author.

    Entities are keyed as (name, type) with type in {"person", "group", "place"}.

    Nodes: (name, type)
      - node['count'] = number of poems where entity appears
      - node['type']  = "person"/"group"/"place"
    Edges: co-occurrence of entities in the same poem
      - edge['weight'] = number of poems where the two entities co-occur
    """
    G = nx.Graph()
    entity_freq: Dict[Tuple[str, str], int] = defaultdict(int)

    for entry in entries:
        entities = entry.get("entities", {})
        person_list = entities.get("person", [])
        group_list = entities.get("group", [])
        place_list = entities.get("place", [])

        entity_set = set(
            [(e, "person") for e in person_list]
            + [(e, "group") for e in group_list]
            + [(e, "place") for e in place_list]
        )

        for ent in entity_set:
            entity_freq[ent] += 1

        entity_list = list(entity_set)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                e1, e2 = entity_list[i], entity_list[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    for ent, freq in entity_freq.items():
        if ent not in G.nodes:
            G.add_node(ent)
        G.nodes[ent]["count"] = freq
        G.nodes[ent]["type"] = ent[1]

    return G, entity_freq


def apply_manual_type_overrides_to_graph(
    G: nx.Graph,
    overrides: Dict[str, str] = MANUAL_ENTITY_TYPE_OVERRIDES,
) -> None:
    """
    Apply manual type overrides to nodes in an entity graph.

    Overrides keys are entity names (e.g. "Lesbia"), and values are types
    ("person", "group", "place").
    """
    for node in G.nodes:
        # node may be a (name, type) tuple or a string, depending on construction
        if isinstance(node, tuple):
            name = node[0]
        else:
            name = node
        if name in overrides:
            G.nodes[node]["type"] = overrides[name]


def draw_entity_graph(
    G: nx.Graph,
    author: str,
    node_threshold: int = 1,
    edge_threshold: int = 1,
    k: float = 5.0,
    figsize: Tuple[int, int] = (12, 10),
    font_size: int = 12,
    node_size_multiplier: int = 300,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Draw an entity co-occurrence graph:

    - Keep nodes with node['count'] >= node_threshold.
    - Keep edges with edge['weight'] >= edge_threshold.
    - Node shapes/colors depend on entity type:
        - person: circle
        - place: square
        - group: hexagon
    """
    print(f"\n[INFO] Drawing entity graph for {author}")
    print(f"[INFO] Original nodes: {len(G.nodes())}, edges: {len(G.edges())}")

    nodes = [n for n, d in G.nodes(data=True) if d.get("count", 0) >= node_threshold]
    print(f"[DEBUG] Nodes with count ≥ {node_threshold}: {len(nodes)}")
    if not nodes:
        print(f"[WARN] No entities with count ≥ {node_threshold} for {author}")
        return

    G_filtered = G.subgraph(nodes).copy()
    edges_to_draw = [
        (u, v)
        for u, v, d in G_filtered.edges(data=True)
        if d.get("weight", 1) >= edge_threshold
    ]
    print(f"[DEBUG] Edges with weight ≥ {edge_threshold}: {len(edges_to_draw)}")
    if not edges_to_draw:
        print(f"[WARN] No edges with weight ≥ {edge_threshold} for {author}")

    pos = nx.spring_layout(G_filtered, k=k, seed=42)

    base_color = mcolors.to_rgb(AUTHOR_COLORS.get(author, DEFAULT_COLOR))
    type_colors = {
        "person": mcolors.to_hex([min(1, base_color[0] * 1.0), base_color[1] * 0.8, base_color[2] * 0.8]),
        "place":  mcolors.to_hex([base_color[0] * 0.8, min(1, base_color[1] * 1.0), base_color[2] * 0.8]),
        "group":  mcolors.to_hex([base_color[0] * 0.8, base_color[1] * 0.8, min(1, base_color[2] * 1.0)]),
    }
    type_shapes = {
        "person": "o",  # circle
        "place": "s",   # square
        "group": "h",   # hexagon
    }

    plt.figure(figsize=figsize)

    # Draw nodes by type
    for ent_type in ["person", "place", "group"]:
        nodelist = [
            n for n in G_filtered.nodes()
            if G_filtered.nodes[n].get("type") == ent_type
        ]
        print(f"[DEBUG] Drawing {len(nodelist)} nodes of type '{ent_type}'")
        if not nodelist:
            continue
        sizes = [
            G_filtered.nodes[n].get("count", 1) * node_size_multiplier
            for n in nodelist
        ]
        nx.draw_networkx_nodes(
            G_filtered,
            pos,
            nodelist=nodelist,
            node_shape=type_shapes[ent_type],
            node_size=sizes,
            node_color=type_colors[ent_type],
            alpha=0.9,
        )

    # Draw edges and labels
    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edgelist=edges_to_draw,
        edge_color="lightgray",
        width=1.5,
        alpha=0.7,
    )
    node_labels = {
        n: f"{n[0] if isinstance(n, tuple) else n}\n({G_filtered.nodes[n].get('count', 0)})"
        for n in G_filtered.nodes()
    }
    nx.draw_networkx_labels(
        G_filtered, pos, labels=node_labels, font_size=font_size
    )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Person",
            markerfacecolor=type_colors["person"],
            markeredgecolor="k",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Place",
            markerfacecolor=type_colors["place"],
            markeredgecolor="k",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="h",
            color="w",
            label="Group",
            markerfacecolor=type_colors["group"],
            markeredgecolor="k",
            markersize=10,
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=12)
    plt.title(
        f"Entity Graph for {author} "
        f"(nodes count ≥ {node_threshold}, edges weight ≥ {edge_threshold})",
        fontsize=18,
    )
    plt.axis("off")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"{author}_entity_graph_nodes{node_threshold}_edges{edge_threshold}.png"
        )
        plt.savefig(output_path, dpi=300)
        print(f"[OK] Entity graph saved: {output_path}")
        plt.close()
    else:
        plt.show()


def build_entity_graphs_from_tei(
    tei_folder: Path,
    filename_filter: Optional[str] = None,
) -> Dict[str, Tuple[nx.Graph, Dict[Tuple[str, str], int]]]:
    """
    Build entity co-occurrence graphs from TEI markup for all authors.

    If filename_filter is not None, only TEI files whose names contain
    that substring are processed.

    Returns:
      graphs[author] = (G_entity, entity_freq)
    """
    if filename_filter is None:
        tei_files = list(tei_folder.glob("*.xml"))
    else:
        tei_files = [
            f for f in tei_folder.glob("*.xml")
            if filename_filter in f.name
        ]
    print(f"[INFO] Found {len(tei_files)} TEI files for entity graphs (TEI).")

    entries: List[Dict[str, Any]] = []
    for file_path in tei_files:
        data = extract_entities_from_tei_file(file_path)
        if data:
            entries.append(data)

    grouped = group_entries_by_author(entries)
    graphs: Dict[str, Tuple[nx.Graph, Dict[Tuple[str, str], int]]] = {}

    for author, author_entries in grouped.items():
        G_entity, entity_freq = build_author_entity_graph(author_entries)
        apply_manual_type_overrides_to_graph(G_entity, MANUAL_ENTITY_TYPE_OVERRIDES)
        graphs[author] = (G_entity, entity_freq)
        total = sum(entity_freq.values())
        print(f"[INFO] Author: {author} - Total unique entities: {total}")

    return graphs


# ============================================================
# 5) ENTITY CO-OCCURRENCE GRAPHS FROM LEMMATIZED CSV
# ============================================================

def build_entity_graphs_from_lemmas(
    csv_path: Path,
) -> Dict[str, Tuple[nx.Graph, Dict[Tuple[str, str], int]]]:
    """
    Build entity co-occurrence graphs from a lemmatized entities CSV.

    The CSV is expected to have at least:
      - 'filename'
      - 'lemma'
      - 'entity_type'  (values like 'PERSON', 'LOC', 'NORP')

    Entities are grouped by (author, poem), where:
      poem = filename without '.txt'
      author = text before first '_' in poem.

    Returns:
      graphs[author] = (G_entity, entity_freq)
    """
    type_map = {
        "PERSON": "person",
        "LOC": "place",
        "NORP": "group",
    }

    df = pd.read_csv(csv_path)
    df["poem"] = df["filename"].str.replace(".txt", "", regex=False).str.strip()
    df["author"] = df["poem"].apply(
        lambda x: x.split("_")[0] if "_" in x else x
    )

    grouped = df.groupby(["author", "poem"])
    graphs: Dict[str, Tuple[nx.Graph, Dict[Tuple[str, str], int]]] = {}
    author_graphs: Dict[str, Tuple[nx.Graph, Dict[Tuple[str, str], int]]] = defaultdict(
        lambda: (nx.Graph(), defaultdict(int))
    )

    for (author, poem), group in grouped:
        group_valid = group.dropna(subset=["lemma", "entity_type"])

        entity_set = set()
        for _, row in group_valid.iterrows():
            lemma = str(row["lemma"]).strip()
            raw_type = str(row["entity_type"]).strip().upper()
            mapped_type = type_map.get(raw_type)
            if mapped_type:
                entity_set.add((lemma, mapped_type))

        G, freq = author_graphs[author]

        for ent in entity_set:
            freq[ent] += 1

        ent_list = list(entity_set)
        for i in range(len(ent_list)):
            for j in range(i + 1, len(ent_list)):
                e1, e2 = ent_list[i], ent_list[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

        for ent in entity_set:
            if ent not in G.nodes:
                G.add_node(ent)
            G.nodes[ent]["count"] = freq[ent]
            G.nodes[ent]["type"] = ent[1]

    # Apply manual overrides and finalize dict
    for author, (G, freq) in author_graphs.items():
        apply_manual_type_overrides_to_graph(G, MANUAL_ENTITY_TYPE_OVERRIDES)
        graphs[author] = (G, freq)
        total = sum(freq.values())
        print(
            f"[INFO] Author: {author} - Total unique entities (lemma+type): {total}"
        )

    return graphs


# ============================================================
# MAIN ENTRY POINT (EXAMPLE)
# ============================================================

def main():
    """
    Example entry point.

    By default this function:
      - Computes topic counts per author and plots bar charts.
      - Builds topic graphs per author and shows them with generic thresholds.

    You can comment/uncomment or extend this function as needed.
    """
    ensure_outputs_dir()

    # 1) Topic counts and bar plots
    aggregated_counts = aggregate_topic_counts_by_author(TEI_FOLDER)
    print("\n[INFO] Topic counts per author:")
    for author, topics in aggregated_counts.items():
        total_author = sum(topics.values())
        print(f"  Author: {author} - Total: {total_author}")
    plot_topic_bars_by_author(aggregated_counts, output_dir=TOPIC_BAR_PLOTS_DIR)

    # 2) Topic graphs
    topic_graphs = build_topic_graphs_for_all_authors(TEI_FOLDER)
    for author, (G_topic, topic_freq) in topic_graphs.items():
        # Example generic thresholds; adjust as needed
        draw_topic_graph(
            G_topic,
            author,
            node_threshold=1,
            edge_threshold=1,
            k=4.0,
            figsize=(12, 8),
            font_size=10,
            node_size_multiplier=200,
            output_dir=TOPIC_GRAPH_PLOTS_DIR,
        )

    # 3) (Optional) entity graphs from lemmas
    # entity_graphs = build_entity_graphs_from_lemmas(LEMMA_ENTITIES_CSV)
    # for author, (G_entity, entity_freq) in entity_graphs.items():
    #     draw_entity_graph(
    #         G_entity,
    #         author,
    #         node_threshold=3,
    #         edge_threshold=2,
    #         k=3.5,
    #         figsize=(12, 8),
    #         font_size=11,
    #         node_size_multiplier=250,
    #         output_dir=ENTITY_GRAPH_PLOTS_DIR,
    #     )

    print("\n[INFO] tei_visualizations main() finished.")


if __name__ == "__main__":
    main()
