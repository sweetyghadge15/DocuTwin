import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.embeddings import generate_embeddings

def compare_passages(
    doc_a_data: Dict[str, Any],
    doc_b_data: Dict[str, Any],
    emb_a_data: Dict[str, Any],
    emb_b_data: Dict[str, Any],
    method: str = "Hybrid (TF-IDF + semantic)",
    alpha: float = 0.4,
    beta: float = 0.6,
    top_n: int = 5,
    threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Compares passages of Document A against Document B to find the closest matching sections.

    Args:
        doc_a_data: Document dictionary for Doc A containing 'passages'.
        doc_b_data: Document dictionary for Doc B containing 'passages'.
        emb_a_data: Embedding dictionary containing 'passage_embeddings' for Doc A.
        emb_b_data: Embedding dictionary containing 'passage_embeddings' for Doc B.
        method: Similarity calculation mode ("Hybrid (TF-IDF + semantic)", "TF-IDF only", "Semantic only").
        alpha: Lexical weight.
        beta: Semantic weight.
        top_n: Number of top matches to return.
        threshold: Minimum similarity threshold filter.

    Returns:
        List of dicts representing top passage matches:
            [
                {
                    'text_a': str,
                    'text_b': str,
                    'lexical_score': float,
                    'semantic_score': float,
                    'hybrid_score': float,
                    'display_score': float
                }, ...
            ]
    """
    passages_a = doc_a_data.get("passages", [])
    passages_b = doc_b_data.get("passages", [])

    if not passages_a or not passages_b:
        return []

    # 1. Lexical Similarity Matrix (len(A) x len(B))
    combined_texts = passages_a + passages_b
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20000,
        token_pattern=r"(?u)\b\w+\b"
    )
    
    try:
        tfidf_mat = vectorizer.fit_transform(combined_texts)
        tfidf_a = tfidf_mat[:len(passages_a)]
        tfidf_b = tfidf_mat[len(passages_a):]
        lexical_matrix = cosine_similarity(tfidf_a, tfidf_b)
    except ValueError:
        lexical_matrix = np.zeros((len(passages_a), len(passages_b)))

    lexical_matrix = np.clip(lexical_matrix, 0.0, 1.0)

    # 2. Semantic Similarity Matrix (len(A) x len(B))
    p_emb_a = emb_a_data.get("passage_embeddings")
    p_emb_b = emb_b_data.get("passage_embeddings")

    if p_emb_a is None or len(p_emb_a) == 0:
        p_emb_a = generate_embeddings(passages_a)
    if p_emb_b is None or len(p_emb_b) == 0:
        p_emb_b = generate_embeddings(passages_b)

    if p_emb_a is not None and p_emb_b is not None:
        semantic_matrix = cosine_similarity(p_emb_a, p_emb_b)
        semantic_matrix = np.maximum(0.0, semantic_matrix)
        semantic_matrix = np.clip(semantic_matrix, 0.0, 1.0)
    else:
        semantic_matrix = np.zeros((len(passages_a), len(passages_b)))

    # 3. Hybrid Score Matrix
    total_w = alpha + beta
    a_w = alpha / total_w if total_w > 0 else 0.4
    b_w = beta / total_w if total_w > 0 else 0.6
    hybrid_matrix = a_w * lexical_matrix + b_w * semantic_matrix

    # 4. Extract candidates
    candidates = []
    for i, pa in enumerate(passages_a):
        for j, pb in enumerate(passages_b):
            lex = float(lexical_matrix[i, j])
            sem = float(semantic_matrix[i, j])
            hyb = float(hybrid_matrix[i, j])

            if method == "TF-IDF only":
                score = lex
            elif method == "Semantic only":
                score = sem
            else:
                score = hyb

            if score >= threshold:
                candidates.append({
                    "passage_idx_a": i,
                    "passage_idx_b": j,
                    "text_a": pa,
                    "text_b": pb,
                    "lexical_score": lex,
                    "semantic_score": sem,
                    "hybrid_score": hyb,
                    "display_score": score
                })

    # Sort descending by primary score
    candidates.sort(key=lambda x: x["display_score"], reverse=True)

    # 5. Deduplicate overlapping passages to provide diverse, distinct match suggestions
    chosen = []
    seen_a, seen_b = set(), set()

    for cand in candidates:
        # Use first 100 characters as unique signature
        key_a = cand["text_a"][:100]
        key_b = cand["text_b"][:100]

        if key_a in seen_a or key_b in seen_b:
            continue

        chosen.append(cand)
        seen_a.add(key_a)
        seen_b.add(key_b)

        if len(chosen) >= top_n:
            break

    return chosen
