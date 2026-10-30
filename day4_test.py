from rag.preprocessing.tokenizer import tokenize_text
from rag.indexing.simple_index import build_inverted_index, keyword_search
from rag.indexing.toy_embedding import semantic_search


documents = [
    {
        "doc_id": "day1",
        "text": "Python text processing includes data cleaning, token, index, embedding, and Word2Vec."
    },
    {
        "doc_id": "day2",
        "text": "Transformer uses Q K V and attention to process tokens."
    },
    {
        "doc_id": "day3",
        "text": "LLM API supports prompt design, streaming, structured output, and function calling."
    },
    {
        "doc_id": "day4",
        "text": "Ollama can run local models. RAG uses ChromaDB, memory, tools, frontend and backend."
    }
]

sample_text = documents[0]["text"]

print("===== TOKENS =====")
print(tokenize_text(sample_text))

print("\n===== INVERTED INDEX SEARCH =====")
index = build_inverted_index(documents)
print(keyword_search("attention embedding", index))

print("\n===== TOY SEMANTIC SEARCH =====")
results = semantic_search("How does transformer attention work?", documents, top_k=2)

for result in results:
    print("=" * 50)
    print("doc_id:", result["doc_id"])
    print("score:", result["score"])
    print("text:", result["text"])