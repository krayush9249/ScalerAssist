import re
import unicodedata

def clean_text(text: str) -> str:
    """
    Clean and normalize raw text.
    
    Steps:
    1. Normalize unicode characters (NFKC)
    2. Remove boilerplate (page numbers, copyrights, URLs)
    3. Remove HTML/XML tags
    4. Remove unwanted repeated special characters (---, ***, @@)
    5. Normalize whitespace
    6. Remove non-ASCII symbols
    7. Strip leading/trailing whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"Page\s*\d+\s*(of)?\s*\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}\s*.*", "", text)
    text = re.sub(r"(www\.|https?:\/\/)\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"([@#\-\*\)\(=+\|\\\/&%$^!~`{}\[\]:;\"',.?])\1{1,}", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()


def clean_corpus_file(input_path: str, output_path: str) -> None:
    """
    Read raw corpus text file, clean it, and write cleaned text to output file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    cleaned_text = clean_text(corpus)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)