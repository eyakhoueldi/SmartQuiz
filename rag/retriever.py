from rag.embeddings import search


def retrieve_context(query: str, n_results: int = 5) -> str:
    chunks = search(query, n_results=n_results)
    if not chunks:
        return "No relevant context found."
    return "\n\n".join(chunks)