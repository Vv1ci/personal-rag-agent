from rag.preprocessing.tokenizer import tokenize_text


def build_inverted_index(documents: list[dict]) -> dict:
    """
    Build a simple inverted index.

    Args:
        documents: a list of dictionaries.
        Each document should contain:
        - doc_id
        - text

    Returns:
        A dictionary:
        token -> list of doc_ids
    """
    index = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]

        tokens = tokenize_text(text)

        for token in tokens:
            if token not in index:
                index[token] = []

            if doc_id not in index[token]:
                index[token].append(doc_id)

    return index


def keyword_search(query: str, index: dict) -> list:
    """
    Search documents by keyword tokens.
    """
    query_tokens = tokenize_text(query)

    matched_doc_ids = set()

    for token in query_tokens:
        if token in index:
            for doc_id in index[token]:
                matched_doc_ids.add(doc_id)

    return list(matched_doc_ids)


if __name__ == "__main__":
    documents = [
        {
            "doc_id": "doc1",
            "text": "Transformer uses attention to process tokens."
        },
        {
            "doc_id": "doc2",
            "text": "Embedding converts text into vectors."
        },
        {
            "doc_id": "doc3",
            "text": "Ollama can run local language models."
        }
    ]

    index = build_inverted_index(documents)

    print("===== INDEX =====")
    print(index)

    print("\n===== SEARCH RESULT =====")
    print(keyword_search("attention model", index))