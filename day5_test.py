import json

from rag.loaders.pdf_loader import load_pdf_file
from rag.preprocessing.cleaner import clean_text
from rag.preprocessing.chunker import chunk_text_by_characters


pages = load_pdf_file("data/raw/sample.pdf")

all_chunks = []

for page in pages:
    cleaned_text = clean_text(page["text"])

    page_chunks = chunk_text_by_characters(
        cleaned_text,
        chunk_size=500,
        overlap=50
    )

    for chunk in page_chunks:
        chunk_with_metadata = {
            "chunk_id": f"{page['source_file']}_page_{page['page_number']}_chunk_{chunk['chunk_id']}",
            "text": chunk["text"],
            "metadata": {
                "source_file": page["source_file"],
                "page_number": page["page_number"],
                "local_chunk_id": chunk["chunk_id"],
                "start_index": chunk["start_index"],
                "end_index": chunk["end_index"]
            }
        }

        all_chunks.append(chunk_with_metadata)


print(f"Total chunks: {len(all_chunks)}")

for chunk in all_chunks[:3]:
    print("=" * 60)
    print("chunk_id:", chunk["chunk_id"])
    print("metadata:", chunk["metadata"])
    print(chunk["text"][:300])


with open("data/processed/sample_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("\nChunks saved to data/processed/sample_chunks.json")