from pathlib import Path
import json
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "all_chunks.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "chat_history.jsonl"


def load_chunks(path: Path) -> list[dict]:
    """
    Load all chunks from all_chunks.json.

    all_chunks.json is a JSON file.
    In this project, it should usually be a list of dictionaries.
    Each dictionary represents one chunk.
    """
    if not path.exists():
        print(f"[WARN] Chunks file not found: {path}")
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        print("[WARN] all_chunks.json exists, but it is not a list.")
        return []

    except json.JSONDecodeError as error:
        print(f"[ERROR] Failed to parse JSON: {error}")
        return []


def count_history_lines(path: Path) -> int:
    """
    Count non-empty lines in chat_history.jsonl.

    JSONL means JSON Lines:
    each line is one JSON object.
    This format is useful for logging because we can append one record at a time.
    """
    if not path.exists():
        return 0

    count = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1

    return count


def get_chunk_source(chunk: dict) -> str:
    """
    Try to find the source file name from a chunk.

    Different projects may store source information in slightly different places:
    - chunk["metadata"]["source"]
    - chunk["metadata"]["source_file"]
    - chunk["source"]
    - chunk["file_name"]

    This function makes the script more robust.
    """
    metadata = chunk.get("metadata", {})

    possible_keys = ["source", "source_file", "file_name", "filename", "path"]

    for key in possible_keys:
        if key in metadata:
            return str(metadata[key])

        if key in chunk:
            return str(chunk[key])

    return "unknown"


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    history_count = count_history_lines(HISTORY_PATH)

    raw_files = []
    if RAW_DIR.exists():
        raw_files = [path for path in RAW_DIR.iterdir() if path.is_file()]

    source_counter = Counter()

    for chunk in chunks:
        source = get_chunk_source(chunk)
        source_counter[source] += 1

    print("=" * 60)
    print("Personal RAG Agent - Project Snapshot")
    print("=" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw documents: {len(raw_files)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chat history records: {history_count}")

    print("\nRaw files:")
    for path in raw_files:
        print(f"- {path.name}")

    print("\nChunks by source:")
    if source_counter:
        for source, count in source_counter.most_common():
            print(f"- {source}: {count}")
    else:
        print("- No chunks found.")

    print("=" * 60)


if __name__ == "__main__":
    main()