import re


def tokenize_text(text: str) -> list[str]:
    """
    Convert text into simple lowercase tokens.

    This is a simple tokenizer for learning:
    - lowercase English letters
    - keep English words and numbers
    - remove most punctuation

    Example:
    "Transformer uses Attention!" -> ["transformer", "uses", "attention"]
    """
    if text is None:
        return []

    text = text.lower()

    tokens = re.findall(r"[a-zA-Z0-9]+", text)

    return tokens


if __name__ == "__main__":
    sample = "Transformer uses Attention! Q/K/V are important in 2026."
    tokens = tokenize_text(sample)

    print(tokens)