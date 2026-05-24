import logging
import json
from typing import Any

from src.rag.legal_utils import record_ref_key
from src.rag.rag_store_config import RAGStoreConfig


logger = logging.getLogger("Neo4jLegalGraphStore")


class Neo4jLegalGraphStore:
    """Neo4j-backed graph store with the same read interface as DeterministicLegalGraphStore."""

    def __init__(self, config: RAGStoreConfig | None = None):
        self.config = config or RAGStoreConfig()
        self.driver = self._driver()
        try:
            self.driver.verify_connectivity()
        except Exception as exc:
            self.driver.close()
            raise RuntimeError(
                "Cannot connect to Neo4j graph backend at "
                f"{self.config.neo4j_uri}. Start it with "
                "`docker compose -f docker-compose.rag.yml up -d neo4j` "
                "and sync graph data, or run evaluation with RAG_GRAPH_BACKEND=local."
            ) from exc
        self.loaded = True

    def _driver(self):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise RuntimeError("Neo4j graph backend requires neo4j. Install requirements first.") from exc
        return GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def lookup_record_nodes(self, records: list[dict[str, Any]]) -> list[str]:
        ids = []
        ref_keys = []
        for record in records:
            node_id = record.get("source_chunk_id") or record.get("id")
            if node_id:
                ids.append(node_id)
            key = record_ref_key(record)
            if key:
                ref_keys.append(key)
        query = """
        MATCH (n:LegalNode)
        WHERE n.id IN $ids OR n.ref_key IN $ref_keys
        RETURN DISTINCT n.id AS id
        """
        with self.driver.session(database=self.config.neo4j_database) as session:
            rows = session.run(query, ids=list(dict.fromkeys(ids)), ref_keys=list(dict.fromkeys(ref_keys)))
            return [row["id"] for row in rows if row["id"]]

    def lookup_ref(self, document: str, article: str, clause: str = "", point: str = "") -> list[str]:
        keys = []
        if point:
            keys.append(f"{document}|D{article}|K{clause}|P{point}")
        if clause:
            keys.append(f"{document}|D{article}|K{clause}")
        if article:
            keys.append(f"{document}|D{article}")
        query = """
        MATCH (n:LegalNode {type: 'legal_chunk'})
        WHERE n.ref_key IN $keys
        RETURN DISTINCT n.id AS id
        """
        with self.driver.session(database=self.config.neo4j_database) as session:
            rows = session.run(query, keys=keys)
            return [row["id"] for row in rows if row["id"]]

    def expand(
        self,
        seed_node_ids: list[str],
        *,
        depth: int = 2,
        include_edge_types: set[str] | None = None,
        max_nodes: int = 40,
    ) -> list[dict[str, Any]]:
        if not seed_node_ids:
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
        safe_depth = max(0, min(int(depth), 4))
        query = f"""
        MATCH p=(n:LegalNode)-[*0..{safe_depth}]-(m:LegalNode)
        WHERE n.id IN $ids
          AND all(r IN relationships(p) WHERE type(r) IN $edge_types)
        WITH m, min(length(p)) AS dist
        RETURN m.data_json AS data_json, dist
        ORDER BY dist ASC
        LIMIT $limit
        """
        with self.driver.session(database=self.config.neo4j_database) as session:
            rows = session.run(
                query,
                ids=list(dict.fromkeys(seed_node_ids)),
                edge_types=list(include_edge_types),
                limit=max_nodes,
            )
            out = []
            for row in rows:
                node = json.loads(row["data_json"] or "{}")
                node["graph_distance"] = row["dist"]
                node["graph_via"] = "neo4j"
                out.append(node)
            return out

    def related_asset_nodes(self, seed_node_ids: list[str]) -> list[dict[str, Any]]:
        query = """
        MATCH (n:LegalNode)-[r]->(asset:LegalNode)
        WHERE n.id IN $ids AND type(r) IN ['HAS_TABLE', 'HAS_FIGURE']
        RETURN asset.data_json AS data_json
        """
        with self.driver.session(database=self.config.neo4j_database) as session:
            rows = session.run(query, ids=list(dict.fromkeys(seed_node_ids)))
            return [json.loads(row["data_json"] or "{}") for row in rows if row["data_json"]]
