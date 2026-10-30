import json
from pathlib import Path


REQUIRED_TOP_LEVEL_FIELDS = ["chunk_id", "text", "metadata"]

REQUIRED_METADATA_FIELDS = [
    "source_file",
    "file_type",
    "page_number",
    "local_chunk_id",
    "start_index",
    "end_index"
]


def validate_chunks_file(file_path: str) -> bool:
    """
    Validate the structure of a chunks JSON file.

    Args:
        file_path: path to chunks JSON file

    Returns:
        True if validation passes, False otherwise
    """
    path = Path(file_path)

    if not path.exists():
        print(f"File not found: {file_path}")
        return False

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        print("Validation failed: JSON root must be a list.")
        return False

    if len(chunks) == 0:
        print("Validation failed: chunks list is empty.")
        return False

    chunk_ids = set()

    for index, chunk in enumerate(chunks):
        for field in REQUIRED_TOP_LEVEL_FIELDS:
            if field not in chunk:
                print(f"Validation failed: chunk {index} missing field: {field}")
                return False

        if not chunk["chunk_id"]:
            print(f"Validation failed: chunk {index} has empty chunk_id.")
            return False

        if chunk["chunk_id"] in chunk_ids:
            print(f"Validation failed: duplicate chunk_id: {chunk['chunk_id']}")
            return False

        chunk_ids.add(chunk["chunk_id"])

        if not chunk["text"]:
            print(f"Validation failed: chunk {index} has empty text.")
            return False

        metadata = chunk["metadata"]

        if not isinstance(metadata, dict):
            print(f"Validation failed: chunk {index} metadata must be a dict.")
            return False

        for field in REQUIRED_METADATA_FIELDS:
            if field not in metadata:
                print(f"Validation failed: chunk {index} metadata missing field: {field}")
                return False

    print("Validation passed.")
    print(f"Total chunks checked: {len(chunks)}")
    print(f"Unique chunk IDs: {len(chunk_ids)}")

    return True


if __name__ == "__main__":
    validate_chunks_file("data/processed/all_chunks.json")