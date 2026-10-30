from rag.embeddings.local_embedding import load_embedding_model
from rag.vectorstore.chroma_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    get_chroma_collection,
    query_chroma
)


def distance_to_relevance_score(distance: float) -> float:
    """
    Convert a distance value into a more intuitive relevance score.

    ChromaDB returns distance, where smaller usually means more relevant.
    This function converts it into a score where larger means more relevant.

    This is a simple display score, not ChromaDB's original score.
    """
    return 1 / (1 + distance)


def format_chroma_results(results: dict) -> list[dict]:
    """
    Convert ChromaDB query results into a cleaner list of dictionaries.

    ChromaDB returns batch-style results:
    results["ids"] = [[id1, id2, id3]]

    We usually query one question at a time, so we take index [0].
    """
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    formatted_results = []

    for index in range(len(ids)):
        distance = distances[index]

        formatted_results.append({
            "rank": index + 1,
            "id": ids[index],
            "text": documents[index],
            "metadata": metadatas[index],
            "distance": distance,
            "relevance_score": distance_to_relevance_score(distance)
        })

    return formatted_results


def retrieve(
    query: str,
    top_k: int = 3,
    persist_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a user query.

    Args:
        query: user question
        top_k: number of chunks to retrieve
        persist_path: ChromaDB storage folder
        collection_name: ChromaDB collection name

    Returns:
        A list of retrieved chunks with text, metadata, distance, and relevance score.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    collection = get_chroma_collection(
        persist_path=persist_path,
        collection_name=collection_name
    )

    if collection.count() == 0:
        return []

    model = load_embedding_model()

    from rag.documents import INDEX_LOCK
    with INDEX_LOCK:
        count = collection.count()
        if not count:
            return []
        raw_results = query_chroma(query=query, collection=collection, model=model, top_k=min(top_k, count))

    return format_chroma_results(raw_results)


def build_context_from_results(results: list[dict]) -> str:
    """
    Build a context string from retrieved chunks.

    This will be useful later when sending context to an LLM.
    """
    context_parts = []

    for result in results:
        metadata = result["metadata"]

        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", -1)

        if page_number == -1:
            source = f"{source_file}"
        else:
            source = f"{source_file}, page {page_number}"

        context_part = (
            f"[Source: {source}]\n"
            f"{result['text']}"
        )

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    query = "What is attention in transformer models?"

    results = retrieve(query=query, top_k=3)

    print("Query:", query)
    print("\n===== RETRIEVED RESULTS =====")

    for result in results:
        print("=" * 60)
        print("rank:", result["rank"])
        print("id:", result["id"])
        print("distance:", result["distance"])
        print("relevance_score:", result["relevance_score"])
        print("metadata:", result["metadata"])
        print("text preview:", result["text"][:300])

    print("\n===== CONTEXT FOR LLM =====")
    context = build_context_from_results(results)
    print(context[:1000])