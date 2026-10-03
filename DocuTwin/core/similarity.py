import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_similarity(docs_dict: Dict[str, Dict[str, Any]]) -> Tuple[List[str], np.ndarray]:
    """
    Computes pairwise lexical similarity matrix using TF-IDF and Cosine Similarity.

    Args:
        docs_dict: Dict mapping doc_name -> doc_data (must contain 'clean_text')

    Returns:
        Tuple of (list of doc_names, 2D similarity matrix of shape (N, N))
    """
    names = list(docs_dict.keys())
    N = len(names)
    if N < 2:
        return names, np.zeros((N, N))

    texts = [docs_dict[n].get("clean_text", "") for n in names]
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
        token_pattern=r"(?u)\b\w+\b"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # Occurs if texts contain only stop words or punctuation
        sim_matrix = np.zeros((N, N))
        np.fill_diagonal(sim_matrix, 1.0)

    # Ensure matrix values are bounded in [0, 1]
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    return names, sim_matrix

def calculate_semantic_similarity(names: List[str], embeddings_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """
    Computes pairwise document semantic similarity matrix using document embedding vectors.

    Args:
        names: List of document names matching matrix order.
        embeddings_dict: Dict containing 'document_embedding' for each doc.

    Returns:
        2D similarity matrix of shape (N, N)
    """
    N = len(names)
    if N < 2:
        return np.zeros((N, N))

    doc_vectors = []
    for name in names:
        emb = embeddings_dict.get(name, {}).get("document_embedding")
        if emb is not None:
            doc_vectors.append(emb.flatten())
        else:
            # Fallback zero vector if embedding failed
            doc_vectors.append(np.zeros((384,), dtype=np.float32))

    matrix = np.array(doc_vectors)
    # Cosine similarity between normalized vectors is dot product
    sim_matrix = cosine_similarity(matrix)
    
    # Clip negative values to zero (range 0 to 1)
    sim_matrix = np.maximum(0.0, sim_matrix)
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    return sim_matrix

def calculate_hybrid_similarity(
    lexical_matrix: np.ndarray,
    semantic_matrix: np.ndarray,
    alpha: float = 0.4,
    beta: float = 0.6
) -> np.ndarray:
    """
    Calculates combined hybrid similarity matrix.
    Formula: Hybrid = alpha * Lexical + beta * Semantic

    Args:
        lexical_matrix: TF-IDF similarity matrix
        semantic_matrix: Semantic similarity matrix
        alpha: Weight for lexical similarity (default 0.4)
        beta: Weight for semantic similarity (default 0.6)

    Returns:
        2D hybrid similarity matrix
    """
    # Normalize weights if user modified sliders
    total_weight = alpha + beta
    if total_weight > 0:
        a = alpha / total_weight
        b = beta / total_weight
    else:
        a, b = 0.5, 0.5

    hybrid = a * lexical_matrix + b * semantic_matrix
    return np.clip(hybrid, 0.0, 1.0)

def classify_relationship(score: float, high_thresh: float = 0.75, mod_thresh: float = 0.45) -> str:
    """
    Classifies a numerical similarity score into descriptive relationship bands.
    Disclaimer: Indicates textual/semantic similarity, not plagiarism or authorship.
    """
    if score >= high_thresh:
        return "Highly Similar"
    elif score >= mod_thresh:
        return "Moderately Similar"
    else:
        return "Low Similarity"

def compute_all_similarities(
    docs_dict: Dict[str, Dict[str, Any]],
    embeddings_dict: Dict[str, Dict[str, Any]],
    alpha: float = 0.4,
    beta: float = 0.6,
    high_thresh: float = 0.75,
    mod_thresh: float = 0.45
) -> Dict[str, Any]:
    """
    Runs full similarity pipeline and constructs summary data structures and DataFrames.

    Returns:
        Dict containing:
            - names: list of document names
            - lexical_matrix: np.ndarray (N, N)
            - semantic_matrix: np.ndarray (N, N)
            - hybrid_matrix: np.ndarray (N, N)
            - pairs_df: pd.DataFrame with pairwise comparisons
            - lexical_df: pd.DataFrame (N, N) with named index/columns
            - semantic_df: pd.DataFrame (N, N)
            - hybrid_df: pd.DataFrame (N, N)
    """
    names, lex_mat = calculate_tfidf_similarity(docs_dict)
    sem_mat = calculate_semantic_similarity(names, embeddings_dict)
    hyb_mat = calculate_hybrid_similarity(lex_mat, sem_mat, alpha=alpha, beta=beta)

    N = len(names)
    rows = []
    
    for i in range(N):
        for j in range(i + 1, N):
            doc_a = names[i]
            doc_b = names[j]
            lex_score = float(lex_mat[i, j])
            sem_score = float(sem_mat[i, j])
            hyb_score = float(hyb_mat[i, j])
            rel_label = classify_relationship(hyb_score, high_thresh=high_thresh, mod_thresh=mod_thresh)
            
            rows.append({
                "document_a": doc_a,
                "document_b": doc_b,
                "lexical_similarity": lex_score,
                "semantic_similarity": sem_score,
                "hybrid_similarity": hyb_score,
                "relationship": rel_label
            })

    pairs_df = pd.DataFrame(rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by="hybrid_similarity", ascending=False).reset_index(drop=True)

    # Matrix DataFrames with names as index and headers
    lexical_df = pd.DataFrame(lex_mat, index=names, columns=names)
    semantic_df = pd.DataFrame(sem_mat, index=names, columns=names)
    hybrid_df = pd.DataFrame(hyb_mat, index=names, columns=names)

    return {
        "names": names,
        "lexical_matrix": lex_mat,
        "semantic_matrix": sem_mat,
        "hybrid_matrix": hyb_mat,
        "pairs_df": pairs_df,
        "lexical_df": lexical_df,
        "semantic_df": semantic_df,
        "hybrid_df": hybrid_df,
        "weights": {"alpha": alpha, "beta": beta}
    }
