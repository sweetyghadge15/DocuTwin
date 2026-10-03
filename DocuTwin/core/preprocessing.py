import re
from typing import List, Dict, Any

def clean_text(raw_text: str) -> str:
    """
    Cleans raw document text by stripping null bytes and normalizing whitespace,
    while preserving original words for matching.
    """
    if not raw_text:
        return ""
    # Remove null characters
    text = raw_text.replace("\x00", " ")
    # Replace tabs and multiple spaces/newlines with clean single spacing
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def split_passages(text: str, max_words: int = 120, overlap: int = 25) -> List[str]:
    """
    Splits text into overlapping passage chunks for granular semantic matching.

    Args:
        text: The text string to split.
        max_words: Maximum number of words per passage.
        overlap: Word overlap between consecutive passages.

    Returns:
        List of passage text strings.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    if len(words) <= max_words:
        return [cleaned]

    passages = []
    step = max(1, max_words - overlap)
    
    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_words]
        if chunk_words:
            passage_str = " ".join(chunk_words)
            passages.append(passage_str)
        if start + max_words >= len(words):
            break

    return passages

def preprocess_document(doc_info: Dict[str, Any], max_words: int = 120, overlap: int = 25) -> Dict[str, Any]:
    """
    Preprocesses a document object, producing clean text and passage chunks.

    Args:
        doc_info: Document dict output from extract_document_text.
        max_words: Target passage word count.
        overlap: Overlap between passages.

    Returns:
        Enriched document dictionary with cleaned text and passages.
    """
    raw_text = doc_info.get("raw_text", "")
    cleaned = clean_text(raw_text)
    passages = split_passages(cleaned, max_words=max_words, overlap=overlap)

    updated = dict(doc_info)
    updated["clean_text"] = cleaned
    updated["passages"] = passages
    updated["passage_count"] = len(passages)
    updated["avg_passage_length"] = round(sum(len(p.split()) for p in passages) / max(1, len(passages)), 1)
    
    return updated
