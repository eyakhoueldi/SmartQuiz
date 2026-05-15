import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = "smartquiz_docs"
VECTORSTORE_PATH = "./vectorstore"

# Note: chromadb's type stubs have a known variance bug with EmbeddingFunction.
# The cast trick doesn't fully resolve it either, so we use type: ignore here.
_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_collection():
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,  # type: ignore[arg-type]
    )
    return collection


def add_chunks(chunks: list, doc_name: str) -> int:
    collection = get_collection()
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)


def search(query: str, n_results: int = 5) -> list:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0] if results["documents"] else []


def clear_collection():
    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    client.delete_collection(COLLECTION_NAME)