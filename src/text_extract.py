import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import cv2
import os
import glob

BASE_FOLDER = "/Users/kumarpersonal/Downloads/ScalerAssist/Context"

PDF_FOLDER = os.path.join(BASE_FOLDER, "PDFs")
TEXT_FOLDER = os.path.join(BASE_FOLDER, "Text")

def extract_text_pdfplumber(pdf_path):
    """Extract all text from PDF pages using pdfplumber."""
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            all_text.append(f"\n--- Page {i+1} ---\n{text}")
    return "\n".join(all_text)

def ocr_from_pdf_images(pdf_path):
    """Convert PDF pages to images and run OCR using pytesseract."""
    pages = convert_from_path(pdf_path)
    ocr_texts = []
    for i, image in enumerate(pages):
        img_path = f"temp_page_{i+1}.png"
        image.save(img_path, "PNG")

        img = cv2.imread(img_path)
        text = pytesseract.image_to_string(img)
        ocr_texts.append(f"\n--- OCR from Page {i+1} ---\n{text}")

        os.remove(img_path)
    return "\n".join(ocr_texts)

def cleanup_old_outputs(base_name):
    """Delete previous text and OCR output files for a given PDF basename."""
    text_pattern = os.path.join(TEXT_FOLDER, f"{base_name}_text.txt")
    ocr_pattern = os.path.join(TEXT_FOLDER, f"{base_name}_ocr.txt")

    for file in glob.glob(text_pattern):
        os.remove(file)
        print(f"Deleted old file: {file}")

    for file in glob.glob(ocr_pattern):
        os.remove(file)
        print(f"Deleted old file: {file}")

def process_pdf_file(pdf_path):
    """Process a single PDF: extract text and OCR text; save outputs."""
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\nProcessing: {filename}.pdf")

    cleanup_old_outputs(filename)

    # Extract structured text
    print("Extracting structured text...")
    text_data = extract_text_pdfplumber(pdf_path)
    with open(os.path.join(TEXT_FOLDER, f"{filename}_text.txt"), "w", encoding="utf-8") as f:
        f.write(text_data)

    # OCR from images
    print("Performing OCR on images...")
    ocr_text = ocr_from_pdf_images(pdf_path)
    with open(os.path.join(TEXT_FOLDER, f"{filename}_ocr.txt"), "w", encoding="utf-8") as f:
        f.write(ocr_text)

    print(f"Finished processing {filename}.pdf")

def process_all_pdfs(pdf_folder=PDF_FOLDER):
    """Process all PDFs found in the pdf_folder."""
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return
    for pdf_path in pdf_files:
        process_pdf_file(pdf_path)
    print("\nAll PDF files processed. Clean and updated outputs are ready.")

def merge_all_texts(text_folder=TEXT_FOLDER, output_path=None):
    """
    Merge all text files in text_folder into a single corpus file.
    If output_path not provided, saves as corpus.txt inside base folder.
    """
    corpus = []
    txt_files = glob.glob(os.path.join(text_folder, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            corpus.append(f.read())

    full_text = "\n".join(corpus)
    if output_path is None:
        output_path = os.path.join(BASE_FOLDER, "corpus.txt")

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write(full_text)

    print(f"Corpus saved to {output_path}")
    return output_path