import argparse
import json

from src.rag.legal_graph_rag import LegalGraphRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the hybrid Legal Graph RAG stack.")
    parser.add_argument("query", help="Vietnamese traffic-law question.")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--graph", default="data/graph/legal_graph.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expand-depth", type=int, default=2)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON response.")
    args = parser.parse_args()

    rag = LegalGraphRAG(
        args.processed,
        graph_path=args.graph,
        force_reindex=args.force_reindex,
        use_reranker=not args.no_reranker,
    )
    result = rag.query(args.query, top_k=args.top_k, expand_depth=args.expand_depth)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    print("\n[Căn cứ truy xuất]")
    for ref in result["references"]:
        print(f"- {ref['reference_text']} | {ref['source_chunk_id']} | {', '.join(ref.get('retrieval_reasons') or [])}")
    if result["images"]:
        print("\n[Ảnh liên quan]")
        for image in result["images"]:
            print(f"- {image}")


if __name__ == "__main__":
    main()
