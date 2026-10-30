import math
from rag.preprocessing.tokenizer import tokenize_text


TOY_WORD_EMBEDDINGS = {
    "transformer": [0.9, 0.8, 0.1],
    "attention": [0.95, 0.85, 0.1],
    "embedding": [0.8, 0.7, 0.2],
    "vector": [0.75, 0.65, 0.2],
    "api": [0.1, 0.2, 0.9],
    "backend": [0.1, 0.25, 0.85],
    "ollama": [0.2, 0.3, 0.8],
    "basketball": [0.1, 0.9, 0.1],
    "swimming": [0.1, 0.85, 0.1],
}


def average_vectors(vectors: list[list[float]]) -> list[float]:
    """
    Average a list of vectors.
    """
    if not vectors:
        return [0.0, 0.0, 0.0]

    vector_length = len(vectors[0])
    result = []

    for i in range(vector_length):
        total = 0.0

        for vector in vectors:
            total += vector[i]

        result.append(total / len(vectors))

    return result


def text_to_vector(text: str) -> list[float]:
    """
    Convert text into a toy sentence vector by averaging known word vectors.
    """
    tokens = tokenize_text(text)

    vectors = []

    for token in tokens:
        if token in TOY_WORD_EMBEDDINGS:
            vectors.append(TOY_WORD_EMBEDDINGS[token])

    return average_vectors(vectors)


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


def semantic_search(query: str, documents: list[dict], top_k: int = 2) -> list[dict]:
    """
    Search documents by toy embedding similarity.
    """
    query_vector = text_to_vector(query)

    results = []

    for doc in documents:
        doc_vector = text_to_vector(doc["text"])
        score = cosine_similarity(query_vector, doc_vector)

        results.append({
            "doc_id": doc["doc_id"],
            "text": doc["text"],
            "score": score
        })

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    documents = [
        {
            "doc_id": "doc1",
            "text": "Transformer uses attention."
        },
        {
            "doc_id": "doc2",
            "text": "Embedding converts text into vector."
        },
        {
            "doc_id": "doc3",
            "text": "Backend calls API and Ollama."
        },
        {
            "doc_id": "doc4",
            "text": "Basketball and swimming are sports."
        }
    ]

    query = "How does attention work in transformer?"

    results = semantic_search(query, documents)

    for result in results:
        print("=" * 50)
        print("doc_id:", result["doc_id"])
        print("score:", result["score"])
        print("text:", result["text"])