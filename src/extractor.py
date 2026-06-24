"""
PDF text extraction module.
Takes a PDF file and returns its raw extracted text, page by page.
"""

import pdfplumber


def extract_pdf_text(file) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file: a file path (str) OR a file-like object (e.g. from
              Streamlit's file_uploader), since pdfplumber supports both.

    Returns:
        A single string containing the text of all pages, joined
        with newlines. Returns an empty string if no text could be
        extracted (e.g. a scanned/image-only PDF).
    """
    text_chunks = []

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    return "\n".join(text_chunks)


def save_extracted_text(text: str, output_path: str) -> None:
    """
    Save extracted text to a .txt file, so we can inspect/cache it
    without re-running PDF extraction every time.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)