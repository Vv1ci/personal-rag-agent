from rag.ingestion import process_folder


chunks = process_folder(
    input_folder="data/raw",
    output_file="data/processed/all_chunks.json",
    chunk_size=500,
    overlap=50
)

print("\n===== FIRST 3 CHUNKS =====")

for chunk in chunks[:3]:
    print("=" * 60)
    print("chunk_id:", chunk["chunk_id"])
    print("metadata:", chunk["metadata"])
    print("text preview:", chunk["text"][:300])