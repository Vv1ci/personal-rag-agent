import math
import os
from functools import lru_cache

from rag.config import settings
# Downloads happen explicitly through the launcher setup action.

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = settings.embedding_model


@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    local_files_only: bool = True
) -> SentenceTransformer:
    """
    Load a sentence-transformers embedding model.

    The model is cached after the first load, so repeated calls in the same
    Python process will reuse the same model instead of loading weights again.
    """
    try:
        print(f"Loading embedding model: {model_name}")

        model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only
        )

        return model

    except Exception as error:
        raise RuntimeError(
            "Failed to load the embedding model locally. "
            "If this is your first time using this model, connect to a stable network "
            "and run: python download_embedding_model.py once to download it. "
            "After it is cached, ask_rag.py can run offline."
        ) from error


def embed_texts(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """
    Convert a list of texts into embedding vectors.
    """
    embeddings = model.encode(texts)

    return embeddings.tolist()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0

    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm1 += a * a
        norm2 += b * b

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))


def semantic_search(
    query: str,
    documents: list[dict],
    model: SentenceTransformer,
    top_k: int = 3
) -> list[dict]:
    """
    Search documents by embedding similarity.
    """
    query_embedding = embed_texts([query], model)[0]

    document_texts = [doc["text"] for doc in documents]
    document_embeddings = embed_texts(document_texts, model)

    results = []

    for doc, doc_embedding in zip(documents, document_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)

        results.append({
            "chunk_id": doc.get("chunk_id"),
            "text": doc["text"],
            "metadata": doc.get("metadata", {}),
            "score": score
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    model = load_embedding_model()

    texts = [
        "Transformer uses attention to process tokens.",
        "Embedding converts text into vectors.",
        "I like playing basketball.",
    ]

    embeddings = embed_texts(texts, model)

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))
    print("First 10 numbers of first embedding:")
    print(embeddings[0][:10])
