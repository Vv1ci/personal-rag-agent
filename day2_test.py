from rag.loaders.pdf_loader import load_pdf_file
from rag.preprocessing.cleaner import clean_text


pages = load_pdf_file("data/raw/sample.pdf")

print(f"Total pages: {len(pages)}")

all_cleaned_pages = []

for page in pages:
    cleaned_text = clean_text(page["text"])

    page_result = {
        "source_file": page["source_file"],
        "page_number": page["page_number"],
        "text": cleaned_text
    }

    all_cleaned_pages.append(page_result)

    print("=" * 60)
    print(f"Source: {page_result['source_file']}")
    print(f"Page: {page_result['page_number']}")
    print(page_result["text"][:500])


with open("data/processed/sample_pdf_cleaned.txt", "w", encoding="utf-8") as f:
    for page in all_cleaned_pages:
        f.write(f"\n\n===== {page['source_file']} | Page {page['page_number']} =====\n\n")
        f.write(page["text"])

print("\nCleaned PDF text saved to data/processed/sample_pdf_cleaned.txt")