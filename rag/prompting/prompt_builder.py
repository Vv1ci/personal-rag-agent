from typing import Literal


PromptLanguage = Literal["en", "zh"]


def format_source(metadata: dict) -> str:
    """
    Format metadata into a readable source string.

    Examples:
    ai_notes.md
    sample.pdf, page 2
    """
    source_file = metadata.get("source_file", "unknown")
    page_number = metadata.get("page_number", -1)

    if page_number == -1:
        return source_file

    return f"{source_file}, page {page_number}"


def build_context_from_retrieved_results(retrieved_results: list[dict]) -> str:
    """
    Convert retrieved chunks into a context string for the LLM.

    Each chunk includes its source so the LLM can cite it.
    """
    context_parts = []

    for result in retrieved_results:
        source = format_source(result["metadata"])
        text = result["text"]

        context_part = (
            f"[Source: {source}]\n"
            f"{text}"
        )

        context_parts.append(context_part)

    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(
    query: str,
    retrieved_results: list[dict],
    language: PromptLanguage = "en"
) -> str:
    """
    Build a RAG prompt using retrieved context and user query.

    Args:
        query: user question
        retrieved_results: results returned by retrieve()
        language: answer language, "en" or "zh"

    Returns:
        A complete prompt string for the LLM.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not retrieved_results:
        raise ValueError("retrieved_results cannot be empty.")

    context = build_context_from_retrieved_results(retrieved_results)

    if language == "zh":
        prompt = f"""
你是一个严谨的 RAG 问答助手。

请只根据下面提供的 Context 回答用户问题。
如果 Context 中没有足够信息，请直接说“根据已提供资料，我不知道”，不要编造。
回答时请尽量引用来源，例如：[Source: 文件名, page 页码]。
如果多个来源都相关，请综合回答并分别标注来源。

Context:
{context}

Question:
{query}

Answer:
"""
    else:
        prompt = f"""
You are a careful RAG question-answering assistant.

Use only the provided Context to answer the user's question.
If the Context does not contain enough information, say: "I don't know based on the provided context."
Do not make up facts.
When possible, cite the source using the format: [Source: file name, page number].
If multiple sources are relevant, synthesize them and cite the relevant sources.

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt.strip()


def build_no_context_prompt(
    query: str,
    language: PromptLanguage = "en"
) -> str:
    """
    Build a fallback prompt when no relevant context is retrieved.
    """
    if language == "zh":
        return f"""
你是一个严谨的问答助手。

当前没有检索到相关资料。
请不要编造答案。
请告诉用户：根据已提供资料，我不知道。

Question:
{query}

Answer:
""".strip()

    return f"""
You are a careful question-answering assistant.

No relevant context was retrieved.
Do not make up an answer.
Tell the user: I don't know based on the provided context.

Question:
{query}

Answer:
""".strip()


if __name__ == "__main__":
    sample_results = [
        {
            "rank": 1,
            "id": "ai_notes.md_page_none_chunk_0",
            "text": "Transformer uses attention to process tokens. Attention helps the model focus on important parts of the input.",
            "metadata": {
                "source_file": "ai_notes.md",
                "file_type": ".md",
                "page_number": -1,
                "local_chunk_id": 0,
                "start_index": 0,
                "end_index": 500
            },
            "distance": 0.3,
            "relevance_score": 0.76
        }
    ]

    query = "What is attention in transformer models?"

    prompt = build_rag_prompt(
        query=query,
        retrieved_results=sample_results,
        language="en"
    )

    print(prompt)