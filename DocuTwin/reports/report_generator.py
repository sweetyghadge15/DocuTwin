import datetime
import pandas as pd
from typing import Dict, Any, List

def generate_csv_report(pairs_df: pd.DataFrame) -> bytes:
    """
    Generates downloadable CSV report bytes from the pairwise similarity DataFrame.
    """
    if pairs_df.empty:
        return b"document_a,document_b,lexical_similarity,semantic_similarity,hybrid_similarity,relationship\n"
    
    export_df = pairs_df.copy()
    # Format percentages cleanly
    export_df["lexical_pct"] = (export_df["lexical_similarity"] * 100).round(2).astype(str) + "%"
    export_df["semantic_pct"] = (export_df["semantic_similarity"] * 100).round(2).astype(str) + "%"
    export_df["hybrid_pct"] = (export_df["hybrid_similarity"] * 100).round(2).astype(str) + "%"

    cols = ["document_a", "document_b", "lexical_pct", "semantic_pct", "hybrid_pct", "relationship"]
    return export_df[cols].to_csv(index=False).encode("utf-8")

def generate_txt_report(
    summary_data: Dict[str, Any],
    doc_details: Dict[str, Dict[str, Any]],
    top_matches_by_pair: Dict[str, List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a comprehensive plain-text analysis report.

    Args:
        summary_data: Output from compute_all_similarities.
        doc_details: Extracted document data map.
        top_matches_by_pair: Dict mapping 'DocA ↔ DocB' to list of top passage matches.

    Returns:
        Formatted multi-line text report string.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    names = summary_data.get("names", [])
    pairs_df = summary_data.get("pairs_df", pd.DataFrame())
    weights = summary_data.get("weights", {"alpha": 0.4, "beta": 0.6})

    lines = [
        "================================================================================",
        "                    DOCUTWIN DOCUMENT SIMILARITY REPORT                         ",
        "================================================================================",
        f"Generated At: {timestamp}",
        f"Analyzed Documents Count: {len(names)}",
        f"Configured Weighting: {weights['alpha']:.0%} Lexical (TF-IDF) + {weights['beta']:.0%} Semantic (SentenceTransformer)",
        "",
        "IMPORTANT LEGAL & ACADEMIC DISCLAIMER:",
        "--------------------------------------------------------------------------------",
        "Similarity scores indicate textual and semantic resemblance only.",
        "They DO NOT independently establish plagiarism, copying, intent, or authorship.",
        "Scores depend on extraction quality, embedding models, and passage chunking.",
        "================================================================================",
        "",
        "1. UPLOADED DOCUMENTS SUMMARY:",
        "--------------------------------------------------------------------------------"
    ]

    for name in names:
        info = doc_details.get(name, {})
        lines.append(
            f" - {name} ({info.get('type', 'DOC')}): "
            f"{info.get('word_count', 0):,} words | "
            f"{info.get('char_count', 0):,} chars | "
            f"{info.get('passage_count', 0)} passages"
        )

    lines.extend([
        "",
        "2. PAIRWISE DOCUMENT SIMILARITY RANKINGS:",
        "--------------------------------------------------------------------------------"
    ])

    if pairs_df.empty:
        lines.append("No document pair comparisons available.")
    else:
        for idx, row in pairs_df.iterrows():
            lines.append(
                f" [{idx+1:02d}] {row['document_a']}  ↔  {row['document_b']}\n"
                f"      Hybrid Score: {row['hybrid_similarity']:.1%}  ({row['relationship']})\n"
                f"      Breakdown:  Lexical (TF-IDF) = {row['lexical_similarity']:.1%}  |  Semantic = {row['semantic_similarity']:.1%}\n"
            )

    if top_matches_by_pair:
        lines.extend([
            "",
            "3. TOP PASSAGE MATCH EXPLANATIONS:",
            "--------------------------------------------------------------------------------"
        ])
        for pair_name, matches in top_matches_by_pair.items():
            lines.append(f"\n>>> Pair: {pair_name}")
            if not matches:
                lines.append("    No matching passage pairs above threshold.")
            for m_idx, match in enumerate(matches, 1):
                lines.extend([
                    f"    Match #{m_idx} (Similarity: {match['display_score']:.1%}):",
                    f"      [Doc A]: \"{match['text_a']}\"",
                    f"      [Doc B]: \"{match['text_b']}\"",
                    ""
                ])

    lines.extend([
        "================================================================================",
        "                            END OF REPORT                                       ",
        "================================================================================"
    ])

    return "\n".join(lines)
