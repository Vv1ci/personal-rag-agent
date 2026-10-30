from rag.embeddings.local_embedding import load_embedding_model
from rag.vectorstore.chroma_store import (
    build_chroma_from_chunks,
    get_chroma_collection,
    query_chroma
)


# Step 1: Build ChromaDB from chunks
build_chroma_from_chunks(
    chunks_file="data/processed/all_chunks.json",
    persist_path="chroma_db",
    collection_name="personal_rag_chunks",
    reset=True
)

# Step 2: Load collection and model
collection = get_chroma_collection(
    persist_path="chroma_db",
    collection_name="personal_rag_chunks"
)

model = load_embedding_model()

# Step 3: Query ChromaDB
query = "What is attention in transformer models?"

results = query_chroma(
    query=query,
    collection=collection,
    model=model,
    top_k=3
)

print("\n===== CHROMADB SEARCH RESULTS =====")
print("Query:", query)

ids = results["ids"][0]
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i in range(len(ids)):
    print("=" * 60)
    print("rank:", i + 1)
    print("id:", ids[i])
    print("distance:", distances[i])
    print("metadata:", metadatas[i])
    print("text preview:", documents[i][:300])