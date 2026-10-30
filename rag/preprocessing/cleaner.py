import re


def normalize_line_endings(text: str) -> str:
    """
    Convert different line endings to Unix style '\n'.
    Windows often uses '\r\n', old Mac may use '\r'.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_extra_spaces(text: str) -> str:
    """
    Remove extra spaces and tabs inside each line.
    """
    text = re.sub(r"[ \t]+", " ", text)
    return text


def strip_each_line(text: str) -> str:
    """
    Remove leading and trailing spaces from each line.
    """
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    return "\n".join(lines)


def remove_too_many_blank_lines(text: str) -> str:
    """
    Replace 3 or more consecutive newlines with 2 newlines.
    This keeps paragraph structure but removes excessive blank lines.
    """
    return re.sub(r"\n{3,}", "\n\n", text)


def remove_simple_page_numbers(text: str) -> str:
    """
    Remove simple page number patterns.

    Examples removed:
    Page 1
    Page 12
    1 / 10
    12/30

    This function is intentionally conservative.
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Remove lines like: Page 1, page 12, PAGE 5
        if re.fullmatch(r"(?i)page\s+\d+", stripped):
            continue

        # Remove lines like: 1 / 10 or 12/30
        if re.fullmatch(r"\d+\s*/\s*\d+", stripped):
            continue

        # Remove lines that are only a number, such as: 3
        # Be careful: this may remove meaningful numbered list items if they are alone.
        if re.fullmatch(r"\d+", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_broken_lines(text: str) -> str:
    """
    Merge lines that were broken in the middle of a sentence.

    Example:
    'Transformer is a neural network
    architecture that uses attention.'

    becomes:
    'Transformer is a neural network architecture that uses attention.'

    This is a mild heuristic:
    - If a line does not end with sentence punctuation
    - and the next line starts with a lowercase letter
    - merge them with a space
    """
    lines = text.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        current_line = lines[i].strip()

        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            if (
                current_line
                and next_line
                and not current_line.endswith((".", "!", "?", ":", ";", "。", "！", "？"))
                and next_line[0].islower()
            ):
                fixed_lines.append(current_line + " " + next_line)
                i += 2
                continue

        fixed_lines.append(current_line)
        i += 1

    return "\n".join(fixed_lines)


def clean_text(text: str) -> str:
    """
    Clean raw text for RAG preprocessing.

    This cleaner:
    - normalizes line endings
    - removes extra spaces
    - strips each line
    - removes simple page numbers
    - fixes some broken lines
    - removes too many blank lines

    It tries to preserve useful document structure.
    """
    if text is None:
        return ""

    text = normalize_line_endings(text)
    text = strip_each_line(text)
    text = remove_simple_page_numbers(text)
    text = remove_extra_spaces(text)
    text = fix_broken_lines(text)
    text = remove_too_many_blank_lines(text)

    return text.strip()


if __name__ == "__main__":
    sample = """
    General AI Training


    Page 3

    Transformer is a neural network
    architecture that uses attention.

    This     sentence      has       too many spaces.

    3 / 10


    ## Attention

    Attention helps the model focus on important tokens.
    """

    cleaned = clean_text(sample)

    print("===== ORIGINAL =====")
    print(sample)

    print("===== CLEANED =====")
    print(cleaned)