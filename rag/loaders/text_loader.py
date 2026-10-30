from pathlib import Path


def load_text_file(file_path: str) -> str:
    """
    Load a .txt or .md file and return raw text.

    Args:
        file_path: path to the text or markdown file

    Returns:
        raw text as a string
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return text


if __name__ == "__main__":
    text = load_text_file("data/raw/ai_notes.md")
    print(text[:500])