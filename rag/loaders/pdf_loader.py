from pathlib import Path
from pypdf import PdfReader


def load_pdf_file(file_path: str) -> list[dict]:
    """
    Load a PDF file and return text page by page.

    Args:
        file_path: path to the PDF file

    Returns:
        A list of dictionaries.
        Each dictionary contains:
        - page_number
        - text
        - source_file
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type: {path.suffix}. Expected .pdf")

    reader = PdfReader(str(path))

    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()

        if text is None:
            text = ""

        pages.append({
            "page_number": page_index + 1,
            "text": text,
            "source_file": path.name
        })

    return pages


if __name__ == "__main__":
    pages = load_pdf_file("data/raw/sample.pdf")

    print(f"Total pages: {len(pages)}")

    for page in pages[:2]:
        print("=" * 50)
        print(f"Source: {page['source_file']}")
        print(f"Page: {page['page_number']}")
        print(page["text"][:500])