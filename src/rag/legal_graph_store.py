import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.rag.legal_utils import record_ref_key


logger = logging.getLogger("LegalGraphStore")


class DeterministicLegalGraphStore:
    """Local graph DB adapter backed by the deterministic graph JSON export.

    The class intentionally exposes graph-store operations instead of leaking the
    JSON format. Replacing it with Neo4j later should only require another class
    implementing the same methods.
    """

    def __init__(self, graph_path: str | Path = "data/graph/legal_graph.json"):
        self.graph_path = Path(graph_path)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.ref_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.figure_code_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.loaded = False
        self.load()

    def load(self) -> None:
        if not self.graph_path.exists():
            logger.warning("Graph file does not exist yet: %s", self.graph_path)
            return
        with self.graph_path.open("r", encoding="utf-8") as f:
            graph = json.load(f)
        self.nodes = {node["id"]: node for node in graph.get("nodes", []) if node.get("id")}
        for edge in graph.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            if source:
                self.out_edges[source].append(edge)
            if target:
                self.in_edges[target].append(edge)

        for node_id, node in self.nodes.items():
            if node.get("type") == "legal_chunk":
                ref = node.get("legal_reference") or {}
                key = "|".join(
                    p
                    for p in [
                        ref.get("document") or node.get("doc_name") or "",
                        f"D{ref.get('article')}" if ref.get("article") else "",
                        f"K{ref.get('clause')}" if ref.get("clause") else "",
                        f"P{ref.get('point')}" if ref.get("point") else "",
                    ]
                    if p
                )
                if key:
                    self.ref_to_nodes[key].append(node_id)
            if node.get("type") == "figure" and node.get("code"):
                self.figure_code_to_nodes[str(node.get("code")).replace(".", "").upper()].append(node_id)
            if node.get("type") == "sign" and node.get("normalized_code"):
                self.figure_code_to_nodes[str(node.get("normalized_code")).replace(".", "").upper()].append(node_id)
        self.loaded = True
        logger.info("Loaded legal graph: nodes=%s edges=%s", len(self.nodes), sum(len(v) for v in self.out_edges.values()))

    def lookup_record_nodes(self, records: list[dict[str, Any]]) -> list[str]:
        node_ids = []
        for record in records:
            node_id = record.get("source_chunk_id") or record.get("id")
            if node_id in self.nodes:
                node_ids.append(node_id)
                continue
            key = record_ref_key(record)
            node_ids.extend(self.ref_to_nodes.get(key, []))
        return list(dict.fromkeys(node_ids))

    def lookup_ref(self, document: str, article: str, clause: str = "", point: str = "") -> list[str]:
        candidates = []
        keys = []
        if point:
            keys.append(f"{document}|D{article}|K{clause}|P{point}")
        if clause:
            keys.append(f"{document}|D{article}|K{clause}")
        if article:
            keys.append(f"{document}|D{article}")
        for key in keys:
            candidates.extend(self.ref_to_nodes.get(key, []))
        return list(dict.fromkeys(candidates))

    def expand(
        self,
        seed_node_ids: list[str],
        *,
        depth: int = 2,
        include_edge_types: set[str] | None = None,
        max_nodes: int = 40,
    ) -> list[dict[str, Any]]:
        if not self.loaded:
            return []
        include_edge_types = include_edge_types or {
            "PARENT_OF",
            "CITES",
            "HAS_TABLE",
            "HAS_FIGURE",
            "HAS_SIGN",
            "REPRESENTS_SIGN",
            "HAS_PENALTY",
            "HAS_PROCEDURE",
            "HAS_ARTICLE",
            "HAS_CLAUSE",
            "HAS_POINT",
            "HAS_CHUNK",
        }
        seen = set()
        queue = deque((node_id, 0, "seed") for node_id in seed_node_ids if node_id in self.nodes)
        expanded = []

        while queue and len(expanded) < max_nodes:
            node_id, dist, via = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = dict(self.nodes[node_id])
            node["graph_distance"] = dist
            node["graph_via"] = via
            expanded.append(node)
            if dist >= depth:
                continue

            related_edges = list(self.out_edges.get(node_id, [])) + list(self.in_edges.get(node_id, []))
            for edge in related_edges:
                if edge.get("type") not in include_edge_types:
                    continue
                neighbor = edge.get("target") if edge.get("source") == node_id else edge.get("source")
                if not neighbor and edge.get("target_ref"):
                    for candidate in self.ref_to_nodes.get(edge["target_ref"], []):
                        queue.append((candidate, dist + 1, edge.get("type", "RELATED")))
                    continue
                if neighbor and neighbor not in seen:
                    queue.append((neighbor, dist + 1, edge.get("type", "RELATED")))
        return expanded

    def related_asset_nodes(self, seed_node_ids: list[str]) -> list[dict[str, Any]]:
        nodes = []
        for seed in seed_node_ids:
            for edge in self.out_edges.get(seed, []):
                if edge.get("type") not in {"HAS_TABLE", "HAS_FIGURE"}:
                    continue
                target = edge.get("target")
                if target in self.nodes:
                    nodes.append(self.nodes[target])
        return nodes
