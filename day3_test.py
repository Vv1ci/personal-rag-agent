from rag.loaders.pdf_loader import load_pdf_file
from rag.preprocessing.cleaner import clean_text


pages = load_pdf_file("data/raw/sample.pdf")

print(f"Total pages: {len(pages)}")

with open("data/processed/sample_pdf_cleaned_day3.txt", "w", encoding="utf-8") as f:
    for page in pages:
        raw_text = page["text"]
        cleaned_text = clean_text(raw_text)

        f.write(f"\n\n===== {page['source_file']} | Page {page['page_number']} =====\n\n")
        f.write(cleaned_text)

        print("=" * 60)
        print(f"Source: {page['source_file']}")
        print(f"Page: {page['page_number']}")
        print("Raw length:", len(raw_text))
        print("Cleaned length:", len(cleaned_text))
        print(cleaned_text[:500])

print("\nDay 3 cleaned PDF text saved to data/processed/sample_pdf_cleaned_day3.txt")