from typing import Optional
from vectorstore import get_index
from embeddings import get_embedding


def retrieve(query: str, k: int = 3, filename: str | None = None) -> list[dict]:
    index = get_index()
    query_embedding = get_embedding(query)

    filter_dict = {"filename": {"$eq": filename}} if filename else None

    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
        include_values=False,
        filter=filter_dict,
    )

    matches = []
    for match in results["matches"]:
        matches.append(
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "page": match["metadata"].get("page"),
                "filename": match["metadata"].get("filename"),
                "chunk_index": match["metadata"].get("chunk_index"),
            }
        )

    return matches


def format_context(matches: list[dict]) -> str:
    context_parts = []

    for i, match in enumerate(matches, 1):
        source = f"[{match['filename']} - Page {match['page']}]"
        context_parts.append(f"--- Source {i} {source} ---\n{match['text']}")

    return "\n\n".join(context_parts)
