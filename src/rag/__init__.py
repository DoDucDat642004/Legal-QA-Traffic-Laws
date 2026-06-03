__all__ = ["LegalGraphRAG"]


def __getattr__(name):
    if name == "LegalGraphRAG":
        from src.rag.legal_graph_rag import LegalGraphRAG

        return LegalGraphRAG
    raise AttributeError(name)
