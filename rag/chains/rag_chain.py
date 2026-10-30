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
            "rank":…10136 tokens truncated…dict],
    chunk_size: int,
    overlap_units: int = 1
) -> list[dict]:
    """
    Pack paragraph or sentence units into chunks.

    The function tries to keep complete units together.
    It also avoids infinite loops when overlap units make the next chunk too long.
    """
    chunks = []

    current_units = []
    current_length = 0

    chunk_id = 0
    index = 0

    while index < len(units):
        unit = units[index]
        unit_text = unit["text"]

        if not current_units:
            current_units.append(unit)
            current_length = len(unit_text)
            index += 1
            continue

        extra_length = len(unit_text) + 2

        if current_length + extra_length <= chunk_size:
            current_units.append(unit)
            current_length += extra_length
            index += 1
            continue

        chunk_text = "\n\n".join(item["text"] for item in current_units)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_index": current_units[0]["start_index"],
            "end_index": current_units[-1]["end_index"]
        })

        chunk_id += 1

        overlap = []

        if overlap_units > 0:
            overlap = current_units[-overlap_units:]

        overlap_length = len("\n\n".join(item["text"] for item in overlap)) if overlap else 0

        # Important safety check:
        # If keeping overlap would make it impossible to add the next unit,
        # drop the overlap so the loop can make progress.
        if overlap and overlap_length + 2 + len(unit_text) <= chunk_size:
            current_units = overlap
            current_length = overlap_length
        else:
            current_units = []
            current_length = 0

    if current_units:
        chunk_text = "\n\n".join(item["text"] for item in current_units)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_index": current_units[0]["start_index"],
            "end_index": current_units[-1]["end_index"]
        })

    return chunks


def chunk_text_smart(
    text: str,
    chunk_size: int = 500,
    overlap_units: int = 1,
    character_overlap: int = 50
) -> list[dict]:
    """
    Split text into smarter chunks.

    Strategy:
    1. Split by paragraphs.
    2. If a paragraph is too long, split it by sentences.
    3. If a sentence is still too long, split it by characters.
    4. Pack units into chunks without breaking natural boundaries when possible.
    """
    if text is None:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap_units < 0:
        raise ValueError("overlap_units cannot be negative")

    if character_overlap < 0:
        raise ValueError("character_overlap cannot be negative")

    if character_overlap >= chunk_size:
        raise ValueError("character_overlap must be smaller than chunk_size")

    paragraph_units = split_text_into_paragraph_units(text)

    final_units = []

    for paragraph_unit in paragraph_units:
        if len(paragraph_unit["text"]) <= chunk_size:
            final_units.append(paragraph_unit)
            continue

        sentence_units = split_long_unit_into_sentence_units(paragraph_unit)

        for sentence_unit in sentence_units:
            if len(sentence_unit["text"]) <= chunk_size:
                final_units.append(sentence_unit)
            else:
                character_units = split_too_long_unit_by_characters(
                    unit=sentence_unit,
                    chunk_size=chunk_size,
                    overlap=character_overlap
                )

                final_units.extend(character_units)

    chunks = create_chunks_from_units(
        units=final_units,
        chunk_size=chunk_size,
        overlap_units=overlap_units
    )

    return chunks


if __name__ == "__main__":
    sample_text = """
# RAG Notes

RAG stands for Retrieval-Augmented Generation.

A RAG system retrieves relevant chunks from an external knowledge base. The retrieved chunks are then inserted into a structured prompt. The language model uses the context to generate a grounded answer.

## ChromaDB

ChromaDB is used as a local vector database. It stores chunk IDs, chunk text, metadata, and embeddings.

During retrieval, ChromaDB compares the query embedding with stored chunk embeddings and returns the top-k most relevant chunks.

## Ollama

Ollama is used to run a local language model.
"""

    print("===== OLD CHARACTER CHUNKS =====")
    old_chunks = chunk_text_by_characters(
        sample_text,
        chunk_size=120,
        overlap=20
    )

    for chunk in old_chunks:
        print("=" * 60)
        print("chunk_id:", chunk["chunk_id"])
        print(chunk["text"])

    print("\n===== SMART CHUNKS =====")
    smart_chunks = chunk_text_smart(
        sample_text,
        chunk_size=120,
        overlap_units=1
    )

    for chunk in smart_chunks:
        print("=" * 60)
        print("chunk_id:", chunk["chunk_id"])
        print(chunk["text"])