import json
from pathlib import Path

from rag.embeddings.local_embedding import load_embedding_model, embed_texts, semantic_search


chunks_path = Path("data/processed/all_chunks.json")

if not chunks_path.exists():
    raise FileNotFoundError(
        "data/processed/all_chunks.json not found. Please run: py run_ingestion.py"
    )

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded chunks: {len(chunks)}")

# To keep Day 8 fast, only test the first 20 chunks.
test_chunks = chunks[:20]

model = load_embedding_model()

texts = [chunk["text"] for chunk in test_chunks[:3]]
embeddings = embed_texts(texts, model)

print("\n===== EMBEDDING TEST =====")
print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
print("First 10 numbers of first embedding:")
print(embeddings[0][:10])

query = "What is attention in transformer models?"

results = semantic_search(
    query=query,
    documents=test_chunks,
    model=model,
    top_k=3
)

print("\n===== SEMANTIC SEARCH RESULTS =====")
print("Query:", query)

for result in results:
    print("=" * 60)
    print("score:", result["score"])
    print("chunk_id:", result["chunk_id"])
    print("metadata:", result["metadata"])
    print("text preview:", result["text"][:300])