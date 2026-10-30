import json
from pathlib import Path

from rag.loaders.text_loader import load_text_file
from rag.loaders.pdf_loader import load_pdf_file
from rag.preprocessing.cleaner import clean_text
from rag.preprocessing.chunker import chunk_text_smart


def create_chunk_id(
    source_file: str,
    page_number: int | None,
    local_chunk_id: int
) -> str:
    """
    Create a unique chunk ID.
    """
    page_part = page_number if page_number is not None else "none"
    return f"{source_file}_page_{page_part}_chunk_{local_chunk_id}"


def build_chunk_with_metadata(
    source_file: str,
    file_type: str,
    page_number: int | None,
    chunk: dict
) -> dict:
    """
    Add metadata to a raw chunk.
    """
    chunk_id = create_chunk_id(
        source_file=source_file,
        page_number=page_number,
        local_chunk_id=chunk["chunk_id"]
    )

    return {
        "chunk_id": chunk_id,
        "text": chunk["text"],
        "metadata": {
            "source_file": source_file,
            "file_type": file_type,
            "page_number": page_number,
            "local_chunk_id": chunk["chunk_id"],
            "start_index": chunk["start_index"],
            "end_index": chunk["end_index"]
        }
    }


def process_text_or_markdown_file(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    path = Path(file_path)

    raw_text = load_text_file(file_path)
    cleaned_text = clean_text(raw_text)

    chunks = chunk_text_smart(
        text=cleaned_text,
        chunk_size=chunk_size,
        overlap_units=1,
        character_overlap=overlap
    )

    chunks_with_metadata = []

    for chunk in chunks:
        chunk_with_metadata = build_chunk_with_metadata(
            source_file=path.name,
            file_type=path.suffix.lower(),
            page_number=None,
            chunk=chunk
        )

        chunks_with_metadata.append(chunk_with_metadata)

    return chunks_with_metadata


def process_pdf_file(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    path = Path(file_path)

    pages = load_pdf_file(file_path)

    chunks_with_metadata = []

    for page in pages:
        cleaned_text = clean_text(page["text"])

        page_chunks = chunk_text_smart(
            text=cleaned_text,
            chunk_size=chunk_size,
            overlap_units=1,
            character_overlap=overlap
        )

        for chunk in page_chunks:
            chunk_with_metadata = build_chunk_with_metadata(
                source_file=path.name,
                file_type=path.suffix.lower(),
                page_number=page["page_number"],
                chunk=chunk
            )

            chunks_with_metadata.append(chunk_with_metadata)

    return chunks_with_metadata


def process_single_file(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    path = Path(file_path)
    file_type = path.suffix.lower()

    if file_type in [".txt", ".md"]:
        return process_text_or_markdown_file(
            file_path=file_path,
            chunk_size=chunk_size,
            overlap=overlap
        )

    if file_type == ".pdf":
        return process_pdf_file(
            file_path=file_path,
            chunk_size=chunk_size,
            overlap=overlap
        )

    raise ValueError(f"Unsupported file type: {file_type}")


def process_folder(
    input_folder: str,
    output_file: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[dict]:
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    all_chunks = []

    supported_extensions = [".txt", ".md", ".pdf"]

    for file_path in input_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        print(f"Processing file: {file_path.name}")

        file_chunks = process_single_file(
            file_path=str(file_path),
            chunk_size=chunk_size,
            overlap=overlap
        )

        all_chunks.extend(file_chunks)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Chunks saved to: {output_file}")

    return all_chunks


if __name__ == "__main__":
    process_folder(
        input_folder="data/raw",
        output_file="data/processed/all_chunks.json",
        chunk_size=500,
        overlap=50
    )