import json
from pathlib import Path

from rag.loaders.text_loader import load_text_file
from rag.preprocessing.cleaner import clean_text
from rag.preprocessing.chunker import chunk_text_by_characters, chunk_text_smart


file_path = "data/raw/rag_project_notes.md"

path = Path(file_path)

if not path.exists():
    raise FileNotFoundError(
        "data/raw/rag_project_notes.md not found. "
        "Please create this file first."
    )

raw_text = load_text_file(file_path)
cleaned_text = clean_text(raw_text)

old_chunks = chunk_text_by_characters(
    text=cleaned_text,
    chunk_size=300,
    overlap=50
)

smart_chunks = chunk_text_smart(
    text=cleaned_text,
    chunk_size=300,
    overlap_units=1,
    character_overlap=50
)

print("===== OLD CHARACTER CHUNKS =====")
print(f"Total old chunks: {len(old_chunks)}")

for chunk in old_chunks[:5]:
    print("=" * 60)
    print("chunk_id:", chunk["chunk_id"])
    print("start:", chunk["start_index"])
    print("end:", chunk["end_index"])
    print(chunk["text"])

print("\n===== SMART CHUNKS =====")
print(f"Total smart chunks: {len(smart_chunks)}")

for chunk in smart_chunks[:5]:
    print("=" * 60)
    print("chunk_id:", chunk["chunk_id"])
    print("start:", chunk["start_index"])
    print("end:", chunk["end_index"])
    print(chunk["text"])

comparison = {
    "old_chunk_count": len(old_chunks),
    "smart_chunk_count": len(smart_chunks),
    "old_chunks_preview": old_chunks[:5],
    "smart_chunks_preview": smart_chunks[:5]
}

output_path = Path("data/processed/day15_chunking_comparison.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print("\nComparison saved to data/processed/day15_chunking_comparison.json")