import json
import logging
import heapq
from collections import defaultdict
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
        self.ref_prefix_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.ref_suffix_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.chunk_ref: dict[str, dict[str, str]] = {}
        self.figure_code_to_nodes: dict[str, list[str]] = defaultdict(list)
        self.loaded = False
        self.load()

    def load(self) -> None:
        self.nodes = {}
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.ref_to_nodes = defaultdict(list)
        self.ref_prefix_to_nodes = defaultdict(list)
        self.ref_suffix_to_nodes = defaultdict(list)
        self.chunk_ref = {}
        self.figure_code_to_nodes = defaultdict(list)
        self.loaded = False

        if not self.graph_path.exists():
            logger.warning("Graph file does not exist yet: %s", self.graph_path)
            return

        graph = self._read_graph(self.graph_path)
        if graph is None:
            fallback_paths = self._fallback_graph_paths()
            if not fallback_paths:
                logger.warning("No loadable graph JSON found for %s", self.graph_path)
                return
            graph = self._merge_graphs(fallback_paths)
            logger.warning(
                "Graph %s is not a loadable JSON graph; merged %s per-document graph files instead.",
                self.graph_path,
                len(fallback_paths),
            )

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
                document = str(ref.get("document") or node.get("doc_name") or "")
                article = str(ref.get("article") or "")
                clause = str(ref.get("clause") or "")
                point = str(ref.get("point") or "")
                key = self._ref_key(document, article, clause, point)
                self.chunk_ref[node_id] = {
                    "document": document,
                    "article": article,
                    "clause": clause,
                    "point": point,
                }
                if key:
                    self.ref_to_nodes[key].append(node_id)
                for prefix_key in self._ref_prefix_keys(document, article, clause, point):
                    self.ref_prefix_to_nodes[prefix_key].append(node_id)
                for suffix_key in self._ref_suffix_keys(article, clause, point):
                    self.ref_suffix_to_nodes[suffix_key].append(node_id)
            if node.get("type") in {"article", "clause", "point"}:
                document = str(node.get("doc_name") or "")
                article = str(node.get("article") or node.get("num") or "")
                clause = str(node.get("clause") or (node.get("num") if node.get("type") == "clause" else "") or "")
                point = str(node.get("point") or (node.get("num") if node.get("type") == "point" else "") or "")
                key = self._ref_key(document, article, clause, point)
                if key:
                    self.ref_to_nodes[key].append(node_id)
            if node.get("type") == "figure" and node.get("code"):
                self.figure_code_to_nodes[str(node.get("code")).replace(".", "").upper()].append(node_id)
            if node.get("type") == "sign" and node.get("normalized_code"):
                self.figure_code_to_nodes[str(node.get("normalized_code")).replace(".", "").upper()].append(node_id)
        self.loaded = True
        logger.info("Loaded legal graph: nodes=%s edges=%s", len(self.nodes), sum(len(v) for v in self.out_edges.values()))

    def _ref_key(self, document: str, article: str = "", clause: str = "", point: str = "") -> str:
        return "|".join(
            p
            for p in [
                document or "",
                f"D{article}" if article else "",
                f"K{clause}" if clause else "",
                f"P{point}" if point else "",
            ]
            if p
        )

    def _ref_prefix_keys(self, document: str, article: str = "", clause: str = "", point: str = "") -> list[str]:
        keys: list[str] = []
        if article:
            keys.append(self._ref_key(document, article))
        if article and clause:
            keys.append(self._ref_key(document, article, clause))
        if article and clause and point:
            keys.append(self._ref_key(document, article, clause, point))
        return [key for key in keys if key]

    def _ref_suffix_keys(self, article: str = "", clause: str = "", point: str = "") -> list[str]:
        keys: list[str] = []
        if article:
            keys.append(f"D{article}")
        if article and clause:
            keys.append(f"D{article}|K{clause}")
        if article and clause and point:
            keys.append(f"D{article}|K{clause}|P{point}")
        return keys

    def _parse_ref_key(self, key: str) -> dict[str, str]:
        parts = [part for part in str(key or "").split("|") if part]
        out = {"document": "", "article": "", "clause": "", "point": ""}
        if parts and not parts[0].startswith(("D", "K", "P")):
            out["document"] = parts[0]
            parts = parts[1:]
        for part in parts:
            if part.startswith("D"):
                out["article"] = part[1:]
            elif part.startswith("K"):
                out["clause"] = part[1:]
            elif part.startswith("P"):
                out["point"] = part[1:]
        return out

    def _resolve_target_ref(self, target_ref: str, *, limit: int = 24) -> list[str]:
        if not target_ref:
            return []
        exact = self.ref_to_nodes.get(target_ref) or self.ref_prefix_to_nodes.get(target_ref) or []
        if exact:
            return list(dict.fromkeys(exact))[:limit]

        parsed = self._parse_ref_key(target_ref)
        suffixes = self._ref_suffix_keys(parsed.get("article", ""), parsed.get("clause", ""), parsed.get("point", ""))
        for suffix in reversed(suffixes):
            candidates = self.ref_suffix_to_nodes.get(suffix, [])
            if candidates:
                return list(dict.fromkeys(candidates))[:limit]
        return []

    def _edge_weight(self, edge: dict[str, Any], *, reverse: bool = False) -> float:
        edge_type = edge.get("type") or "RELATED"
        weights = {
            "HAS_PENALTY": 0.20,
            "HAS_PROCEDURE": 0.25,
            "HAS_TABLE": 0.35,
            "HAS_FIGURE": 0.35,
            "HAS_SIGN": 0.35,
            "REPRESENTS_SIGN": 0.35,
            "HAS_CHUNK": 0.40,
            "HAS_POINT": 0.45,
            "HAS_CLAUSE": 0.50,
            "HAS_ARTICLE": 0.55,
            "PARENT_OF": 0.55,
            "CITES": 0.65,
        }
        weight = weights.get(edge_type, 0.90)
        if reverse and edge_type == "CITES":
            weight += 0.25
        return weight

    def _read_graph(self, path: Path) -> dict[str, Any] | None:
        try:
            with path.open("r", encoding="utf-8") as f:
                first = f.read(256)
                if first.startswith("version https://git-lfs.github.com/spec/"):
                    return None
                f.seek(0)
                graph = json.load(f)
            if isinstance(graph, dict) and isinstance(graph.get("nodes"), list) and isinstance(graph.get("edges"), list):
                return graph
        except Exception as exc:
            logger.warning("Failed to load graph JSON %s: %s", path, exc)
        return None

    def _fallback_graph_paths(self) -> list[Path]:
        graph_dir = self.graph_path.parent
        if not graph_dir.exists():
            return []
        return sorted(
            path
            for path in graph_dir.glob("*_graph.json")
            if path.name != self.graph_path.name and path.is_file()
        )

    def _merge_graphs(self, paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges = set()
        for path in paths:
            graph = self._read_graph(path)
            if not graph:
                continue
            for node in graph.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    nodes[node_id] = node
            for edge in graph.get("edges", []):
                key = (
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("target_ref"),
                    edge.get("type"),
                    edge.get("raw"),
                )
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(edge)
        return {"nodes": list(nodes.values()), "edges": edges}

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
        keys = list(reversed(self._ref_prefix_keys(document, article, clause, point)))
        for key in keys:
            candidates.extend(self.ref_to_nodes.get(key, []))
            candidates.extend(self.ref_prefix_to_nodes.get(key, []))
        return list(dict.fromkeys(candidates))

    def lookup_ref_prefix(
        self,
        document: str,
        article: str,
        clause: str = "",
        point: str = "",
        *,
        limit: int = 80,
    ) -> list[str]:
        keys = list(reversed(self._ref_prefix_keys(document, article, clause, point)))
        candidates: list[str] = []
        for key in keys:
            candidates.extend(self.ref_prefix_to_nodes.get(key, []))
            if candidates:
                break
        return list(dict.fromkeys(candidates))[:limit]

    def expand(
        self,
        seed_node_ids: list[str],
        *,
        depth: int = 2,
        include_edge_types: set[str] | None = None,
        max_nodes: int = 40,
        weighted: bool = True,
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
        if not weighted:
            return self._expand_bfs(
                seed_node_ids,
                depth=depth,
                include_edge_types=include_edge_types,
                max_nodes=max_nodes,
            )

        seen: set[str] = set()
        best_cost: dict[str, float] = {}
        queue: list[tuple[float, int, int, str, str]] = []
        sequence = 0
        for node_id in seed_node_ids:
            if node_id not in self.nodes:
                continue
            heapq.heappush(queue, (0.0, 0, sequence, node_id, "seed"))
            best_cost[node_id] = 0.0
            sequence += 1
        expanded: list[dict[str, Any]] = []

        while queue and len(expanded) < max_nodes:
            cost, dist, _seq, node_id, via = heapq.heappop(queue)
            if node_id in seen:
                continue
            if cost > best_cost.get(node_id, cost):
                continue
            seen.add(node_id)
            node = dict(self.nodes[node_id])
            node["graph_distance"] = dist
            node["graph_cost"] = round(cost, 4)
            node["graph_via"] = via
            expanded.append(node)
            if dist >= depth:
                continue

            for edge, reverse in self._related_edges(node_id):
                if edge.get("type") not in include_edge_types:
                    continue
                edge_type = edge.get("type", "RELATED")
                neighbors = self._edge_neighbors(node_id, edge)
                for neighbor in neighbors:
                    if neighbor not in self.nodes or neighbor in seen:
                        continue
                    next_cost = cost + self._edge_weight(edge, reverse=reverse)
                    if next_cost >= best_cost.get(neighbor, 999999.0):
                        continue
                    best_cost[neighbor] = next_cost
                    heapq.heappush(queue, (next_cost, dist + 1, sequence, neighbor, edge_type))
                    sequence += 1
        return expanded

    def _expand_bfs(
        self,
        seed_node_ids: list[str],
        *,
        depth: int,
        include_edge_types: set[str],
        max_nodes: int,
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        queue = [(node_id, 0, "seed") for node_id in seed_node_ids if node_id in self.nodes]
        expanded: list[dict[str, Any]] = []
        index = 0
        while index < len(queue) and len(expanded) < max_nodes:
            node_id, dist, via = queue[index]
            index += 1
            if node_id in seen:
                continue
            seen.add(node_id)
            node = dict(self.nodes[node_id])
            node["graph_distance"] = dist
            node["graph_cost"] = float(dist)
            node["graph_via"] = via
            expanded.append(node)
            if dist >= depth:
                continue
            for edge, _reverse in self._related_edges(node_id):
                if edge.get("type") not in include_edge_types:
                    continue
                for neighbor in self._edge_neighbors(node_id, edge):
                    if neighbor and neighbor not in seen:
                        queue.append((neighbor, dist + 1, edge.get("type", "RELATED")))
        return expanded

    def _related_edges(self, node_id: str) -> list[tuple[dict[str, Any], bool]]:
        return (
            [(edge, False) for edge in self.out_edges.get(node_id, [])]
            + [(edge, True) for edge in self.in_edges.get(node_id, [])]
        )

    def _edge_neighbors(self, node_id: str, edge: dict[str, Any]) -> list[str]:
        neighbor = edge.get("target") if edge.get("source") == node_id else edge.get("source")
        if neighbor:
            return [neighbor]
        if edge.get("target_ref"):
            return self._resolve_target_ref(str(edge["target_ref"]))
        return []

    def same_ref_context(
        self,
        seed_node_ids: list[str],
        *,
        max_nodes: int = 120,
        per_seed: int = 32,
    ) -> list[dict[str, Any]]:
        """Returns nearby legal chunks by structural ref, e.g. parent clause and sibling points.

        This complements edge traversal: extracted violation chunks often split
        "fine amount" at clause level and "behavior" at point level, so the
        answer needs both even when graph edges are sparse or ordered badly.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seed in seed_node_ids:
            ref = self.chunk_ref.get(seed)
            if not ref:
                continue
            document = ref.get("document", "")
            article = ref.get("article", "")
            clause = ref.get("clause", "")
            point = ref.get("point", "")
            candidate_groups: list[tuple[str, list[str], float]] = []
            if article and clause:
                candidate_groups.append((
                    "same_clause_context",
                    self.lookup_ref_prefix(document, article, clause, limit=per_seed),
                    0.28,
                ))
            if article and point:
                candidate_groups.append((
                    "same_article_context",
                    self.lookup_ref_prefix(document, article, limit=per_seed),
                    0.80,
                ))
            elif article and not clause:
                candidate_groups.append((
                    "same_article_context",
                    self.lookup_ref_prefix(document, article, limit=per_seed),
                    0.55,
                ))
            for via, candidates, cost in candidate_groups:
                for node_id in candidates:
                    if node_id in seen or node_id not in self.nodes:
                        continue
                    seen.add(node_id)
                    node = dict(self.nodes[node_id])
                    node["graph_distance"] = 1
                    node["graph_cost"] = cost
                    node["graph_via"] = via
                    out.append(node)
                    if len(out) >= max_nodes:
                        return out
        return out

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
