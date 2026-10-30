from rag.loaders.text_loader import load_text_file
from rag.preprocessing.cleaner import clean_text


raw_text = load_text_file("data/raw/ai_notes.md")
cleaned_text = clean_text(raw_text)

print("===== RAW TEXT =====")
print(raw_text[:500])

print("\n===== CLEANED TEXT =====")
print(cleaned_text[:500])

with open("data/processed/ai_notes_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print("\nCleaned text saved to data/processed/ai_notes_cleaned.txt")