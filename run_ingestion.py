from rag.ingestion import process_folder


def main():
    """
    Run the full ingestion pipeline.

    This script:
    - reads supported files from data/raw
    - cleans the text
    - splits text into chunks
    - adds metadata
    - saves all chunks to data/processed/all_chunks.json
    """
    chunks = process_folder(
        input_folder="data/raw",
        output_file="data/processed/all_chunks.json",
        chunk_size=500,
        overlap=50
    )

    print("\nIngestion completed successfully.")
    print(f"Total chunks created: {len(chunks)}")


if __name__ == "__main__":
    main()