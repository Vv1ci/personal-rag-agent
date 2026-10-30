import json
from pathlib import Path

import chromadb

from rag.embeddings.local_embedding import load_embedding_model, embed_texts


from rag.config import settings
DEFAULT_CHROMA_PATH = str(settings.chroma_db_dir)
DEFAULT_COLLECTION_NAME = "personal_rag_chunks"


def load_chunks(chunks_file: str) -> list[dict]:
    """
    Load chunks from a JSON file.
    """
    path = Path(chunks_file)

    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def clean_metadata_for_chroma(metadata: dict) -> dict:
    """
    Convert metadata into ChromaDB-compatible metadata.

    ChromaDB metadata values should be simple types:
    str, int, float, or bool.

    We convert None page_number to -1.
    """
    cleaned = {}

    for key, value in metadata.items():
        if value is None:
            cleaned[key] = -1
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)

    return cleaned


def get_chroma_collection(
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME
):
    """
    Create or load a persistent ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=persist_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Personal RAG chunks collection"}
    )

    return collection


def reset_chroma_collection(
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME
):
    """
    Delete and recreate a ChromaDB collection.

    Useful when rebuilding the vector database from scratch.
    """
    client = chromadb.PersistentClient(path=persist_path)

    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        print(f"No existing collection to delete: {collection_name}")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Personal RAG chunks collection"}
    )

    return collection


def add_chunks_to_chroma(
    chunks: list[dict],
    collection,
    model,
    batch_size: int = 32
) -> None:
    """
    Add chunks to ChromaDB in batches.

    Args:
        chunks: list of chunk dictionaries
        collection: ChromaDB collection
        model: SentenceTransformer embedding model
        batch_size: number of chunks to process per batch
    """
    total_chunks = len(chunks)

    for start in range(0, total_chunks, batch_size):
        end = start + batch_size
        batch = chunks[start:end]

        ids = [chunk["chunk_id"] for chunk in batch]
        documents = [chunk["text"] for chunk in batch]
        metadatas = [
            clean_metadata_for_chroma(chunk["metadata"])
            for chunk in batch
        ]

        embeddings = embed_texts(documents, model)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Added chunks {start} to {min(end, total_chunks)} / {total_chunks}")


def build_chroma_from_chunks(
    chunks_file: str = "data/processed/all_chunks.json",
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    reset: bool = True
):
    """
    Build ChromaDB collection from processed chunks JSON.
    """
    chunks = load_chunks(chunks_file)

    print(f"Loaded chunks: {len(chunks)}")

    model = load_embedding_model()

    if reset:
        collection = reset_chroma_collection(
            persist_path=persist_path,
            collection_name=collection_name
        )
    else:
        collection = get_chroma_collection(
            persist_path=persist_path,
            collection_name=collection_name
        )

    add_chunks_to_chroma(
        chunks=chunks,
        collection=collection,
        model=model
    )

    print("\nChromaDB build completed.")
    print(f"Collection count: {collection.count()}")

    return collection


def query_chroma(
    query: str,
    collection,
    model,
    top_k: int = 3
) -> dict:
    """
    Query ChromaDB using a query embedding.
    """
    query_embedding = embed_texts([query], model)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results


if __name__ == "__main__":
    build_chroma_from_chunks()