from rag.embeddings import search


def retrieve_context(query: str, n_results: int = 5) -> str:
    # Fetch the top-n semantically similar chunks for the given query
    chunks = search(query, n_results=n_results)
    if not chunks:
        # Return a fallback string so callers always receive a usable context value
        return "No relevant context found."
    # Join chunks with double newlines to preserve readability when passed to an LLM
    return "\n\n".join(chunks)