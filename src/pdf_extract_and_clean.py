import os
from text_extract import process_all_pdfs, merge_all_texts
from text_cleaner import clean_corpus_file

BASE_FOLDER = "/Users/kumarpersonal/Downloads/ScalerAssist/Context"
RAW_CORPUS_PATH = os.path.join(BASE_FOLDER, "extracted_corpus.txt")
CLEANED_CORPUS_PATH = os.path.join(BASE_FOLDER, "cleaned_text.txt")

def extract_and_clean_pipeline():
    print("Starting PDF extraction...")
    process_all_pdfs()
    
    print("Merging all extracted text files into corpus...")
    merge_all_texts()
    
    print("Cleaning merged corpus text...")
    clean_corpus_file(RAW_CORPUS_PATH, CLEANED_CORPUS_PATH)
    
    print(f"Pipeline complete. Cleaned corpus saved at: {CLEANED_CORPUS_PATH}")

if __name__ == "__main__":
    extract_and_clean_pipeline()
