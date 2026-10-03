import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

_GLOBAL_MODEL = None

def load_sentence_transformer_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads and caches the SentenceTransformer model. Uses Streamlit cache if available,
    otherwise uses a module-level singleton instance.
    """
    global _GLOBAL_MODEL
    
    # Try Streamlit cache first if running inside Streamlit app
    try:
        import streamlit as st
        @st.cache_resource(show_spinner=False)
        def _get_st_cached_model(name: str):
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(name)
        return _get_st_cached_model(model_name)
    except Exception:
        # Fallback to standard Python singleton outside Streamlit
        if _GLOBAL_MODEL is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {model_name}")
            _GLOBAL_MODEL = SentenceTransformer(model_name)
        return _GLOBAL_MODEL

def generate_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Encodes a list of text strings into normalized 2D embedding vectors.

    Args:
        texts: List of passage text strings.
        model_name: SentenceTransformer model identifier.

    Returns:
        numpy.ndarray of shape (len(texts), embedding_dim) or None if texts is empty.
    """
    if not texts:
        return None
    try:
        model = load_sentence_transformer_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise RuntimeError(f"Semantic embedding generation failed: {e}")

def get_document_embeddings(docs: Dict[str, Dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Dict[str, Any]]:
    """
    Generates passage-level and mean-aggregated document embeddings for a dictionary of preprocessed documents.

    Args:
        docs: Dictionary mapping doc_name -> doc_dict (containing 'passages').
        model_name: Embedding model name.

    Returns:
        Dict mapping doc_name -> {
            'passage_embeddings': ndarray (N, D),
            'document_embedding': ndarray (1, D)
        }
    """
    result = {}
    for name, doc_info in docs.items():
        passages = doc_info.get("passages", [])
        if not passages:
            # Fallback if passages empty
            passages = [doc_info.get("clean_text", "")]
        
        passage_emb = generate_embeddings(passages, model_name=model_name)
        
        if passage_emb is not None and len(passage_emb) > 0:
            # Mean pool passage embeddings to form document-level representation
            doc_emb = np.mean(passage_emb, axis=0, keepdims=True)
            # Normalize mean vector to unit norm
            norm = np.linalg.norm(doc_emb)
            if norm > 0:
                doc_emb = doc_emb / norm
        else:
            doc_emb = None

        result[name] = {
            "passage_embeddings": passage_emb,
            "document_embedding": doc_emb
        }
        
    return result
