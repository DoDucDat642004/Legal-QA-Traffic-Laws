import argparse
import json
import logging
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv(os.path.join(project_root, ".env"))

from src.rag.legal_utils import normalized_legal_reference, source_text
from src.rag.rag_store_config import RAGStoreConfig
from src.rag.qdrant_vector_store import QdrantLegalVectorStore
from src.rag.record_expander import load_expanded_records, load_processed_records
from src.rag.traffic_sign_catalog import TrafficSignCatalog


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RAGStoreSync")


def _clean_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_jsonable(item) for key, item in value.items()}
    return value


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\x00", "")


def _json(value: Any) -> str:
    return json.dumps(_clean_jsonable(value), ensure_ascii=False)


def _point_id(record: dict[str, Any]) -> str:
    from src.rag.qdrant_vector_store import _point_id as qdrant_point_id

    return qdrant_point_id(record)


def _safe_rel_type(value: str) -> str:
    rel = re.sub(r"[^A-Z0-9_]+", "_", (value or "RELATED").upper()).strip("_")
    return rel or "RELATED"


def _asset_paths(records: list[dict[str, Any]], catalog: TrafficSignCatalog) -> list[str]:
    paths = set()
    for record in records:
        for value in [record.get("image_path"), *((record.get("rag_metadata") or {}).get("image_paths") or [])]:
            if value:
                paths.add(str(value))
    for entry in catalog.entries.values():
        for value in [entry.image_path, entry.crop_image_path]:
            if value:
                paths.add(str(value))
    return sorted(paths)


class PostgresLegalRepository:
    def __init__(self, config: RAGStoreConfig):
        self.config = config
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError("PostgreSQL sync requires psycopg[binary]. Install requirements first.") from exc
        self.psycopg = psycopg

    def sync(
        self,
        canonical_records: list[dict[str, Any]],
        expanded_records: list[dict[str, Any]],
        catalog: TrafficSignCatalog,
    ) -> None:
        with self.psycopg.connect(self.config.postgres_dsn) as conn:
            self._create_schema(conn)
            batch_size = int(os.getenv("POSTGRES_SYNC_COMMIT_BATCH_SIZE", "500"))
            with conn.cursor() as cur:
                synced_records = 0
                for record_index, record in enumerate(canonical_records):
                    ref = normalized_legal_reference(record)
                    cur.execute(
                        """
                        INSERT INTO legal_records
                            (record_pk, source_chunk_id, record_id, doc_name, record_type, legal_reference, source_text, record)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                        ON CONFLICT (record_pk) DO UPDATE SET
                            source_chunk_id = EXCLUDED.source_chunk_id,
                            record_id = EXCLUDED.record_id,
                            doc_name = EXCLUDED.doc_name,
                            record_type = EXCLUDED.record_type,
                            legal_reference = EXCLUDED.legal_reference,
                            source_text = EXCLUDED.source_text,
                            record = EXCLUDED.record,
                            updated_at = now()
                        """,
                        (
                            f"{record.get('id') or record.get('source_chunk_id') or 'record'}::{record_index}",
                            record.get("source_chunk_id") or record.get("id"),
                            record.get("id"),
                            record.get("doc_name") or ref.get("document"),
                            record.get("record_type"),
                            _json(ref),
                            _clean_text(source_text(record)),
                            _json(record),
                        ),
                    )
                    synced_records += 1

                    for table in record.get("tables") or []:
                        if not isinstance(table, dict) or not table.get("id"):
                            continue
                        cur.execute(
                            """
                            INSERT INTO legal_tables
                                (table_pk, table_id, source_chunk_id, doc_name, caption, table_text, headers, rows, legal_reference, image_path, table_payload)
                            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                            ON CONFLICT (table_pk) DO UPDATE SET
                                table_id = EXCLUDED.table_id,
                                source_chunk_id = EXCLUDED.source_chunk_id,
                                doc_name = EXCLUDED.doc_name,
                                caption = EXCLUDED.caption,
                                table_text = EXCLUDED.table_text,
                                headers = EXCLUDED.headers,
                                rows = EXCLUDED.rows,
                                legal_reference = EXCLUDED.legal_reference,
                                image_path = EXCLUDED.image_path,
                                table_payload = EXCLUDED.table_payload,
                                updated_at = now()
                            """,
                            (
                                f"{record.get('id') or record.get('source_chunk_id') or 'record'}::{record_index}::{table.get('id')}",
                                table.get("id"),
                                record.get("source_chunk_id") or record.get("id"),
                                record.get("doc_name") or ref.get("document"),
                                _clean_text(table.get("caption")),
                                _clean_text(table.get("text")),
                                _json(table.get("headers") or []),
                                _json(table.get("rows") or []),
                                _json(ref),
                                table.get("image_path"),
                                _json(table),
                            ),
                        )
                    if synced_records % batch_size == 0:
                        conn.commit()
                        logger.info("PostgreSQL canonical sync progress: %s/%s", synced_records, len(canonical_records))
                conn.commit()
                logger.info("PostgreSQL canonical records synced: %s", synced_records)

                synced_rag_records = 0
                for record in expanded_records:
                    meta = record.get("rag_metadata") or {}
                    cur.execute(
                        """
                        INSERT INTO rag_records
                            (point_id, source_chunk_id, modality, doc_name, article, clause_num, point_num, rag_text, metadata, record)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (point_id) DO UPDATE SET
                            source_chunk_id = EXCLUDED.source_chunk_id,
                            modality = EXCLUDED.modality,
                            doc_name = EXCLUDED.doc_name,
                            article = EXCLUDED.article,
                            clause_num = EXCLUDED.clause_num,
                            point_num = EXCLUDED.point_num,
                            rag_text = EXCLUDED.rag_text,
                            metadata = EXCLUDED.metadata,
                            record = EXCLUDED.record,
                            updated_at = now()
                        """,
                        (
                            _point_id(record),
                            record.get("source_chunk_id") or record.get("id"),
                            record.get("rag_modality", "text"),
                            meta.get("doc") or record.get("doc_name"),
                            str(meta.get("article") or ""),
                            str(meta.get("clause") or ""),
                            str(meta.get("point") or ""),
                            _clean_text(record.get("rag_text")),
                            _json(meta),
                            _json(record),
                        ),
                    )
                    synced_rag_records += 1
                    if synced_rag_records % batch_size == 0:
                        conn.commit()
                        logger.info("PostgreSQL RAG record sync progress: %s/%s", synced_rag_records, len(expanded_records))
                conn.commit()
                logger.info("PostgreSQL expanded RAG records synced: %s", synced_rag_records)

                synced_signs = 0
                for entry in catalog.entries.values():
                    cur.execute(
                        """
                        INSERT INTO sign_catalog
                            (normalized_code, display_code, name, meaning, visual_features, sign_group,
                             legal_reference, source_chunk_id, image_path, catalog_payload)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                        ON CONFLICT (normalized_code) DO UPDATE SET
                            display_code = EXCLUDED.display_code,
                            name = EXCLUDED.name,
                            meaning = EXCLUDED.meaning,
                            visual_features = EXCLUDED.visual_features,
                            sign_group = EXCLUDED.sign_group,
                            legal_reference = EXCLUDED.legal_reference,
                            source_chunk_id = EXCLUDED.source_chunk_id,
                            image_path = EXCLUDED.image_path,
                            catalog_payload = EXCLUDED.catalog_payload,
                            updated_at = now()
                        """,
                        (
                            entry.normalized_code,
                            entry.code,
                            entry.name,
                            entry.meaning,
                            entry.visual_features,
                            entry.group,
                            _json(entry.legal_reference),
                            entry.source_chunk_id,
                            entry.crop_image_path or entry.image_path,
                            _json(entry.to_record()),
                        ),
                    )
                    synced_signs += 1
                    if synced_signs % batch_size == 0:
                        conn.commit()
                        logger.info("PostgreSQL sign catalog sync progress: %s/%s", synced_signs, len(catalog.entries))
            conn.commit()
        logger.info("PostgreSQL sync completed.")

    def _create_schema(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS legal_records (
                    record_pk text PRIMARY KEY,
                    source_chunk_id text,
                    record_id text,
                    doc_name text,
                    record_type text,
                    legal_reference jsonb NOT NULL DEFAULT '{}'::jsonb,
                    source_text text,
                    record jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_legal_records_source ON legal_records (source_chunk_id);
                CREATE INDEX IF NOT EXISTS idx_legal_records_doc ON legal_records (doc_name);

                CREATE TABLE IF NOT EXISTS rag_records (
                    point_id text PRIMARY KEY,
                    source_chunk_id text,
                    modality text,
                    doc_name text,
                    article text,
                    clause_num text,
                    point_num text,
                    rag_text text,
                    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                    record jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_rag_records_source ON rag_records (source_chunk_id);
                CREATE INDEX IF NOT EXISTS idx_rag_records_modality ON rag_records (modality);
                CREATE INDEX IF NOT EXISTS idx_rag_records_metadata ON rag_records USING gin (metadata);

                CREATE TABLE IF NOT EXISTS legal_tables (
                    table_pk text PRIMARY KEY,
                    table_id text,
                    source_chunk_id text,
                    doc_name text,
                    caption text,
                    table_text text,
                    headers jsonb NOT NULL DEFAULT '[]'::jsonb,
                    rows jsonb NOT NULL DEFAULT '[]'::jsonb,
                    legal_reference jsonb NOT NULL DEFAULT '{}'::jsonb,
                    image_path text,
                    table_payload jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_legal_tables_source ON legal_tables (source_chunk_id);
                CREATE INDEX IF NOT EXISTS idx_legal_tables_rows ON legal_tables USING gin (rows);

                CREATE TABLE IF NOT EXISTS sign_catalog (
                    normalized_code text PRIMARY KEY,
                    display_code text,
                    name text,
                    meaning text,
                    visual_features text,
                    sign_group text,
                    legal_reference jsonb NOT NULL DEFAULT '{}'::jsonb,
                    source_chunk_id text,
                    image_path text,
                    catalog_payload jsonb NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_sign_catalog_group ON sign_catalog (sign_group);
                """
            )
        conn.commit()


class Neo4jGraphRepository:
    def __init__(self, config: RAGStoreConfig):
        self.config = config
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            raise RuntimeError("Neo4j sync requires neo4j. Install requirements first.") from exc
        self.driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password))

    def sync_graph(self, graph_path: Path) -> None:
        with graph_path.open("r", encoding="utf-8") as f:
            graph = json.load(f)
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        with self.driver.session(database=self.config.neo4j_database) as session:
            session.run("CREATE CONSTRAINT legal_node_id IF NOT EXISTS FOR (n:LegalNode) REQUIRE n.id IS UNIQUE")
            for node in nodes:
                props = {
                    "id": node.get("id"),
                    "type": node.get("type"),
                    "doc_name": node.get("doc_name") or node.get("name") or "",
                    "ref_key": self._ref_key(node),
                    "data_json": _json(node),
                }
                session.run(
                    "MERGE (n:LegalNode {id: $id}) SET n += $props",
                    id=node.get("id"),
                    props=props,
                )
            for edge_index, edge in enumerate(edges):
                source = edge.get("source")
                target = edge.get("target")
                if not source or not target:
                    continue
                rel_type = _safe_rel_type(edge.get("type") or "RELATED")
                edge_key = f"{source}->{target}:{rel_type}:{edge_index}"
                session.run(
                    f"""
                    MATCH (a:LegalNode {{id: $source}})
                    MATCH (b:LegalNode {{id: $target}})
                    MERGE (a)-[r:{rel_type} {{edge_key: $edge_key}}]->(b)
                    SET r.data_json = $data
                    """,
                    source=source,
                    target=target,
                    edge_key=edge_key,
                    data=_json(edge),
                )
        self.driver.close()
        logger.info("Neo4j sync completed: nodes=%s edges=%s.", len(nodes), len(edges))

    def _ref_key(self, node: dict[str, Any]) -> str:
        ref = node.get("legal_reference") or {}
        parts = [ref.get("document") or node.get("doc_name") or ""]
        if ref.get("article"):
            parts.append(f"D{ref.get('article')}")
        if ref.get("clause"):
            parts.append(f"K{ref.get('clause')}")
        if ref.get("point"):
            parts.append(f"P{ref.get('point')}")
        return "|".join(str(part) for part in parts if part)


class MinioAssetRepository:
    def __init__(self, config: RAGStoreConfig):
        self.config = config
        try:
            from minio import Minio
        except Exception as exc:
            raise RuntimeError("MinIO sync requires minio. Install requirements first.") from exc
        self.client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=config.minio_secure,
        )

    def sync_assets(self, asset_paths: list[str]) -> None:
        if not self.client.bucket_exists(self.config.minio_bucket):
            self.client.make_bucket(self.config.minio_bucket)
        uploaded = 0
        root = Path(project_root)
        for value in asset_paths:
            path = Path(value)
            if not path.is_absolute():
                path = root / path
            if not path.exists() or not path.is_file():
                continue
            object_name = str(path.relative_to(root)).replace("\\", "/")
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.client.fput_object(self.config.minio_bucket, object_name, str(path), content_type=content_type)
            uploaded += 1
        logger.info("MinIO asset sync completed: uploaded=%s bucket=%s.", uploaded, self.config.minio_bucket)


def summarize(processed_dir: Path, graph_path: Path) -> dict[str, Any]:
    canonical_records = load_processed_records(processed_dir)
    expanded_records = load_expanded_records(processed_dir)
    catalog = TrafficSignCatalog(expanded_records)
    table_records = [record for record in expanded_records if record.get("rag_modality") == "table" or record.get("table")]
    sign_records = [record for record in expanded_records if record.get("rag_modality") == "sign"]
    graph_counts = {"nodes": 0, "edges": 0}
    if graph_path.exists():
        with graph_path.open("r", encoding="utf-8") as f:
            graph = json.load(f)
        graph_counts = {"nodes": len(graph.get("nodes", [])), "edges": len(graph.get("edges", []))}
    return {
        "canonical_records": len(canonical_records),
        "expanded_records": len(expanded_records),
        "table_records": len(table_records),
        "sign_records": len(sign_records),
        "sign_catalog_entries": len(catalog.entries),
        "asset_paths": len(_asset_paths(expanded_records, catalog)),
        "graph_nodes": graph_counts["nodes"],
        "graph_edges": graph_counts["edges"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync legal RAG data into PostgreSQL, Qdrant, Neo4j, and MinIO.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--graph-path", default="data/graph/legal_graph.json")
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--dry-run", action="store_true", help="Only print counts; do not connect to external stores.")
    parser.add_argument("--skip-postgres", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--skip-minio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    graph_path = Path(args.graph_path)
    stats = summarize(processed_dir, graph_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    config = RAGStoreConfig()
    canonical_records = load_processed_records(processed_dir)
    expanded_records = load_expanded_records(processed_dir)
    catalog = TrafficSignCatalog(expanded_records)

    if not args.skip_postgres:
        PostgresLegalRepository(config).sync(canonical_records, expanded_records, catalog)
    if not args.skip_qdrant:
        QdrantLegalVectorStore(
            processed_path=processed_dir,
            embedding_model=args.embedding_model,
            force_reindex=True,
            config=config,
        )
    if not args.skip_neo4j:
        Neo4jGraphRepository(config).sync_graph(graph_path)
    if not args.skip_minio:
        MinioAssetRepository(config).sync_assets(_asset_paths(expanded_records, catalog))


if __name__ == "__main__":
    main()
