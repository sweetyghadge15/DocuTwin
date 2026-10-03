import io
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def extract_document_text(file_name: str, file_bytes: bytes, file_type: str) -> Dict[str, Any]:
    """
    Extracts text and metadata from uploaded document bytes.

    Args:
        file_name: Name of the file.
        file_bytes: Raw byte stream of the file.
        file_type: Extension or type string ('pdf', 'docx', 'txt').

    Returns:
        Dict containing:
            - name: File name
            - raw_text: Extracted raw text
            - word_count: Number of words
            - char_count: Number of characters
            - page_count: Page count (PDF) or section count
            - file_size_kb: Size in KB
            - error: Error message string if extraction failed, else None
    """
    ext = file_type.lower().strip(".")
    size_kb = len(file_bytes) / 1024.0
    text = ""
    page_count = 1
    error = None

    try:
        if ext == "txt":
            # Attempt multiple encodings for plain text files
            decoded = False
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    text = file_bytes.decode(encoding)
                    decoded = True
                    break
                except UnicodeDecodeError:
                    continue
            if not decoded:
                text = file_bytes.decode("utf-8", errors="replace")

        elif ext == "pdf":
            try:
                import fitz  # PyMuPDF
                text_parts = []
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    page_count = len(doc)
                    for page in doc:
                        extracted = page.get_text()
                        if extracted:
                            text_parts.append(extracted)
                text = "\n".join(text_parts)
                if not text.strip():
                    error = "PDF contains no readable text layer (it may be a scanned image or empty)."
            except Exception as e:
                logger.error(f"Failed to extract PDF text from {file_name}: {e}")
                error = f"Error reading PDF file: {str(e)}"

        elif ext == "docx":
            try:
                from docx import Document
                doc_stream = io.BytesIO(file_bytes)
                document = Document(doc_stream)
                paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
                # Include text from tables
                for table in document.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            paragraphs.append(row_text)
                text = "\n".join(paragraphs)
                page_count = max(1, len(document.paragraphs) // 15)
                if not text.strip():
                    error = "DOCX document appears to be empty."
            except Exception as e:
                logger.error(f"Failed to extract DOCX text from {file_name}: {e}")
                error = f"Error reading DOCX file: {str(e)}"
        else:
            error = f"Unsupported file extension: .{ext}"

    except Exception as e:
        error = f"Unexpected error during text extraction: {str(e)}"

    words = text.split() if text else []
    
    return {
        "name": file_name,
        "raw_text": text,
        "word_count": len(words),
        "char_count": len(text),
        "page_count": page_count,
        "file_size_kb": round(size_kb, 2),
        "type": ext.upper(),
        "error": error
    }
