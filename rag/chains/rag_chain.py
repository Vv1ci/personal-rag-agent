from rag.retrieval.retriever import retrieve
from rag.prompting.prompt_builder import build_rag_prompt, build_no_context_prompt
from rag.llm.ollama_client import generate_with_ollama, DEFAULT_MODEL


DEFAULT_MIN_RELEVANCE_SCORE = 0.42


def format_sources(retrieved_results: list[dict]) -> list[dict]:
    """
    Format source information from retrieved results.
    """
    sources = []

    for result in retrieved_results:
        metadata = result["metadata"]

        source = {
            "text": result["text"],
            "rank": result["rank"],
            "id": result["id"],
            "source_file": metadata.get("source_file", "unknown"),
            "page_number": metadata.get("page_number", -1),
            "distance": result["distance"],
            "relevance_score": result["relevance_score"]
        }

        sources.append(source)

    return sources


def filter_results_by_relevance(
    retrieved_results: list[dict],
    min_relevance_score: float
) -> list[dict]:
    """
    Keep only retrieved results whose relevance score is high enough.
    """
    filtered_results = []

    for result in retrieved_results:
        if result["relevance_score"] >= min_relevance_score:
            filtered_results.append(result)

    return filtered_results


def answer_with_rag(
    query: str,
    top_k: int = 3,
    language: str = "en",
    model: str = DEFAULT_MODEL,
    min_relevance_score: float = DEFAULT_MIN_RELEVANCE_SCORE
) -> dict:
    """
    Full RAG pipeline:
    query -> retrieve -> filter -> prompt -> LLM answer -> answer with sources.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    retrieved_results = retrieve(
        query=query,
        top_k=top_k
    )

    filtered_results = filter_results_by_relevance(
        retrieved_results=retrieved_results,
        min_relevance_score=min_relevance_score
    )

    if filtered_results:
        prompt = build_rag_prompt(
            query=query,
            retrieved_results=filtered_results,
            language=language
        )

        answer = generate_with_ollama(
            prompt=prompt,
            model=model
        )
    else:
        prompt = build_no_context_prompt(
            query=query,
            language=language
        )

        if language == "zh":
            answer = "根据已提供资料，我不知道。"
        else:
            answer = "I don't know based on the provided context."

    return {
        "query": query,
        "answer": answer,
        "sources": format_sources(filtered_results),
        "retrieved_results": retrieved_results,
        "filtered_results": filtered_results,
        "min_relevance_score": min_relevance_score,
        "prompt": prompt
    }


if __name__ == "__main__":
    query = "What is attention in transformer models?"

    result = answer_with_rag(
        query=query,
        top_k=3,
        language="en"
    )

    print("===== QUERY =====")
    print(result["query"])

    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")
    for source in result["sources"]:
        print(source)

    print("\n===== ALL RETRIEVED RESULTS =====")
    for item in result["retrieved_results"]:
        print(
            item["rank"],
            item["id"],
            item["relevance_score"]
        )

    print("\n===== FILTERED RESULTS =====")
    for item in result["filtered_results"]:
        print(
            item["rank"],
            item["id"],
            item["relevance_score"]
        )