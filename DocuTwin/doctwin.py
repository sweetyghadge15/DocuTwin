import io
import re
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cosine

def extract_text(doc):
    name, raw, ext = doc["name"], doc["bytes"], doc["type"]
    if ext == "txt":
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                pass
        return raw.decode("utf-8", errors="replace")
    if ext == "pdf":
        import fitz
        text_parts = []
        with fitz.open(stream=raw, filetype="pdf") as pdf:
            for page in pdf:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)
    if ext == "docx":
        from docx import Document
        document = Document(io.BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")

def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_passages(text, max_words=120, overlap=25):
    words = clean_text(text).split()
    if not words:
        return []
    passages = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = words[start:start + max_words]
        if chunk:
            passages.append(" ".join(chunk))
        if start + max_words >= len(words):
            break
    return passages

def _tfidf_matrix(texts):
    # Character n-grams can catch small edits; word n-grams capture shared terms.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
        token_pattern=r"(?u)\b\w+\b"
    )
    return vectorizer.fit_transform(texts)

def _semantic_matrix(texts):
    # SentenceTransformer model is downloaded on first use and cached by the library.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def _similarities(texts, method):
    if len(texts) < 2:
        return np.zeros((len(texts), len(texts)))
    tf = None
    sem = None
    if method in ("TF-IDF only", "Hybrid (TF-IDF + semantic)"):
        tf = cosine_similarity(_tfidf_matrix(texts))
    if method in ("Semantic only", "Hybrid (TF-IDF + semantic)"):
        sem = cosine(np.asarray(_semantic_matrix(texts)))
    if method == "TF-IDF only":
        return tf
    if method == "Semantic only":
        return sem
    # A simple baseline hybrid. Evaluate and tune this weight on a labeled dataset.
    return 0.4 * tf + 0.6 * sem

def analyze_documents(docs, method="Hybrid (TF-IDF + semantic)"):
    extracted = {}
    for doc in docs:
        text = clean_text(extract_text(doc))
        if len(text.split()) < 5:
            continue
        extracted[doc["name"]] = text
    if len(extracted) < 2:
        raise ValueError("At least two documents with readable text (5+ words) are required.")

    names = list(extracted.keys())
    matrix = _similarities([extracted[n] for n in names], method)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        score = float(np.clip(matrix[i, j], 0.0, 1.0))
        if score >= 0.75:
            label = "High similarity — review"
        elif score >= 0.45:
            label = "Moderate similarity — review"
        else:
            label = "Lower similarity"
        rows.append({
            "document_a": names[i],
            "document_b": names[j],
            "similarity": score,
            "relationship": label
        })
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values("similarity", ascending=False).reset_index(drop=True)
    return {"documents": extracted, "pairs": pairs, "method": method}

def compare_passages(text_a, text_b, method="Hybrid (TF-IDF + semantic)", top_n=3):
    passages_a = split_passages(text_a)
    passages_b = split_passages(text_b)
    if not passages_a or not passages_b:
        return []
    combined = passages_a + passages_b
    # Reuse the same comparison logic so passage scores are method-consistent.
    matrix = _similarities(combined, method)
    offset = len(passages_a)
    candidates = []
    for i, pa in enumerate(passages_a):
        for j, pb in enumerate(passages_b):
            candidates.append({
                "text_a": pa,
                "text_b": pb,
                "similarity": float(np.clip(matrix[i, offset + j], 0.0, 1.0))
            })
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    # Avoid returning many near-identical overlapping passage pairs.
    chosen = []
    seen_a, seen_b = set(), set()
    for item in candidates:
        a_key, b_key = item["text_a"][:100], item["text_b"][:100]
        if a_key in seen_a or b_key in seen_b:
            continue
        chosen.append(item)
        seen_a.add(a_key)
        seen_b.add(b_key)
        if len(chosen) >= top_n:
            break
    return chosen

def build_relationship_graph(pairs, threshold=0.55):
    import plotly.graph_objects as go
    if pairs.empty:
        fig = go.Figure()
        fig.update_layout(title="No relationships to display")
        return fig

    nodes = sorted(set(pairs["document_a"]).union(set(pairs["document_b"])))
    # Deterministic circular layout without requiring a graph package.
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    positions = {name: (float(np.cos(a)), float(np.sin(a))) for name, a in zip(nodes, angles)}
    edge_x, edge_y = [], []
    edge_text = []
    for _, row in pairs.iterrows():
        if row["similarity"] < threshold:
            continue
        x0, y0 = positions[row["document_a"]]
        x1, y1 = positions[row["document_b"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f'{row["document_a"]} ↔ {row["document_b"]}: {row["similarity"]:.1%}')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.5), hoverinfo="none", name="Relationships"
    ))
    node_x = [positions[n][0] for n in nodes]
    node_y = [positions[n][1] for n in nodes]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=nodes, textposition="top center", hovertext=nodes, hoverinfo="text",
        marker=dict(size=22), name="Documents"
    ))
    fig.update_layout(
        height=520, showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title=f"Relationships at or above {threshold:.0%} similarity"
    )
    return fig

def export_report(pairs, matches, selected_pair, threshold):
    lines = [
        "DocuTwin Analysis Report",
        "=" * 30,
        f"Selected pair: {selected_pair}",
        f"Relationship-map threshold: {threshold:.0%}",
        "",
        "Important: similarity is not proof of plagiarism, authorship, or copying.",
        "Scores depend on the chosen model, text extraction quality, and document content.",
        "",
        "Pairwise results:"
    ]
    if pairs.empty:
        lines.append("No pair results.")
    else:
        for _, row in pairs.iterrows():
            lines.append(
                f'- {row["document_a"]} vs {row["document_b"]}: '
                f'{row["similarity"]:.1%} ({row["relationship"]})'
            )
    lines += ["", "Closest passage pairs:"]
    for i, item in enumerate(matches, 1):
        lines += [
            f"\nMatch {i}: {item['similarity']:.1%}",
            f"A: {item['text_a']}",
            f"B: {item['text_b']}"
        ]
    return "\n".join(lines)
