import sys
import os
# Ensure the project root is on the path so sibling packages can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Name of the ChromaDB collection used to store document chunks
COLLECTION_NAME = "smartquiz_docs"
# Local directory where ChromaDB persists its data on disk
VECTORSTORE_PATH = "./vectorstore"

# Shared embedding function using a lightweight sentence-transformer model
_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_collection():
    # Connect to (or create) the persistent ChromaDB store at the configured path
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    # Fetch the collection if it exists, otherwise create it with our embedding function
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,  # type: ignore[arg-type]
    )
    return collection


def add_chunks(chunks: list, doc_name: str) -> int:
    collection = get_collection()
    # Build a unique ID for each chunk using the document name and its index
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    # Insert the chunks (ChromaDB auto-embeds them via _embed_fn)
    collection.add(documents=chunks, ids=ids)
    # Return the number of chunks stored
    return len(chunks)


def search(query: str, n_results: int = 5) -> list:
    collection = get_collection()
    # Run a semantic similarity search against the stored chunks
    results = collection.query(query_texts=[query], n_results=n_results)
    # Return the matched documents for the first (and only) query, or empty list
    return results["documents"][0] if results["documents"] else []


def clear_collection():
    # Drop the entire collection from the persistent store (irreversible)
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    client.delete_collection(COLLECTION_NAME)