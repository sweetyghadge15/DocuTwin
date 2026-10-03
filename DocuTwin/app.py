import os
import io
import time
import pandas as pd
import numpy as np
import streamlit as st

# Custom modules
from core.document_parser import extract_document_text
from core.preprocessing import preprocess_document
from core.embeddings import get_document_embeddings
from core.similarity import compute_all_similarities
from core.passage_matching import compare_passages
from visualization.similarity_matrix import create_similarity_heatmap
from visualization.document_graph import build_document_dna_graph
from reports.report_generator import generate_csv_report, generate_txt_report

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuTwin — Intelligent Document Similarity & Relationship Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Global layout enhancements */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    
    /* Disclaimers & Alerts */
    .disclaimer-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 5px solid #3B82F6;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #1E3A8A;
        font-size: 0.92rem;
    }
    
    /* Card Container Styling */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center;
    }
    
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Workflow step box */
    .workflow-step {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .workflow-step h4 {
        color: #4F46E5;
        margin-bottom: 5px;
    }

    /* Side-by-side passage boxes */
    .passage-box {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 14px;
        font-size: 0.95rem;
        line-height: 1.5;
        min-height: 120px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/duotone/96/4f46e5/documents.png", width=64)
    st.title("DocuTwin")
    st.caption("Intelligent Document Similarity & Relationship Analyzer")
    st.divider()

    st.subheader("⚙️ Analysis Settings")
    
    method_option = st.selectbox(
        "Similarity Method",
        ["Hybrid (TF-IDF + semantic)", "TF-IDF only (Lexical)", "Semantic only (Embeddings)"],
        index=0,
        help="Hybrid combines lexical frequency with neural sentence transformer embeddings."
    )

    st.markdown("**Hybrid Weights Calibration**")
    lex_weight = st.slider("Lexical Weight (TF-IDF)", 0.0, 1.0, 0.4, 0.05)
    sem_weight = round(1.0 - lex_weight, 2)
    st.caption(f"Semantic Weight: **{sem_weight:.2f}** | Lexical Weight: **{lex_weight:.2f}**")

    st.divider()
    st.markdown("**Thresholds & Limits**")
    rel_threshold = st.slider("Relationship Threshold", 0.10, 0.95, 0.50, 0.05, help="Minimum similarity to show relationship connection in Document DNA.")
    passage_top_n = st.slider("Passage Matches Per Pair", 1, 10, 4)
    
    high_threshold = st.slider("High Similarity Cutoff", 0.60, 0.95, 0.75, 0.05)
    mod_threshold = st.slider("Moderate Similarity Cutoff", 0.30, 0.70, 0.45, 0.05)

    st.divider()
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.uploaded_docs = {}
        st.session_state.analysis_results = None
        st.rerun()

    st.info("💡 **Disclaimer:** Similarity indicates textual or semantic resemblance and does not independently prove plagiarism or copying.")

# -----------------------------------------------------------------------------
# 4. Main Application Layout & Tabs
# -----------------------------------------------------------------------------
st.title("📄 DocuTwin")
st.caption("Understand how your documents are related through lexical frequency and neural semantic modeling.")

# Academic & Legal Disclaimer Banner
st.markdown("""
<div class="disclaimer-box">
    <b>ℹ️ Educational & ML Project Disclaimer:</b> DocuTwin measures lexical overlap (TF-IDF) and semantic embedding similarity (SentenceTransformers). 
    Similarity scores indicate textual or conceptual resemblance and <b>do not establish plagiarism, copyright infringement, or intent</b>.
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab_home, tab_upload, tab_overview, tab_matrix, tab_passages, tab_dna, tab_search, tab_reports, tab_about = st.tabs([
    "🏠 Home",
    "📤 Upload & Process",
    "📊 Analyze & Overview",
    "🗺️ Similarity Matrix",
    "🔍 Passage Matches",
    "🧬 Document DNA",
    "🔎 Search",
    "📑 Reports & History",
    "ℹ️ About"
])

# -----------------------------------------------------------------------------
# TAB 1: HOME
# -----------------------------------------------------------------------------
with tab_home:
    st.markdown("### Discover How Your Documents Are Related")
    st.markdown("""
    DocuTwin is a machine-learning powered document analysis platform designed to compare multiple text files, 
    PDFs, and Word documents. It combines **lexical n-gram matching** with modern **neural sentence transformer embeddings** 
    to reveal hidden relationships and passage overlaps.
    """)

    st.markdown("#### ⚡ Core 5-Step ML Workflow")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="workflow-step"><h4>1. Upload</h4><p style="font-size:0.85rem;">PDF, DOCX, TXT document collection</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="workflow-step"><h4>2. Extract</h4><p style="font-size:0.85rem;">PyMuPDF & docx text parsing</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="workflow-step"><h4>3. Embed</h4><p style="font-size:0.85rem;">TF-IDF & all-MiniLM-L6-v2 vectors</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="workflow-step"><h4>4. Compare</h4><p style="font-size:0.85rem;">Hybrid scoring & passage alignment</p></div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="workflow-step"><h4>5. Visualize</h4><p style="font-size:0.85rem;">Document DNA graph & reports</p></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🎯 Target Use Cases")
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        st.markdown("##### 🎓 Students & Researchers")
        st.write("Compare research paper drafts, literature reviews, and citations across multiple documents.")
    with uc2:
        st.markdown("##### 📝 Content Editors & Reviewers")
        st.write("Identify overlapping sections, rephrased content, and common reference materials.")
    with uc3:
        st.markdown("##### 🏢 Enterprise & Knowledge Repositories")
        st.write("Detect duplicate internal documentation, standard operating procedures, and policy revisions.")

# -----------------------------------------------------------------------------
# TAB 2: UPLOAD & PROCESS
# -----------------------------------------------------------------------------
with tab_upload:
    st.subheader("📤 Document Ingestion & Processing Pipeline")
    
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Drag and drop documents here (PDF, DOCX, TXT)",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            help="Select 2 or more files to perform comparative analysis."
        )
    with col_sample:
        st.markdown("**Quick Start with Sample Data**")
        if st.button("📁 Load 4 Demo Documents", use_container_width=True):
            sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
            sample_files = ["Research_Paper_A.txt", "Research_Paper_B.txt", "Research_Paper_C.txt", "Project_Report.txt"]
            
            for fname in sample_files:
                fpath = os.path.join(sample_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, "rb") as f:
                        raw_b = f.read()
                    parsed = extract_document_text(fname, raw_b, "txt")
                    processed = preprocess_document(parsed)
                    st.session_state.uploaded_docs[fname] = processed
            st.success("Loaded 4 demo documents successfully!")
            st.rerun()

    # Process newly uploaded files
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_docs:
                with st.spinner(f"Extracting and processing {f.name}..."):
                    raw_bytes = f.getvalue()
                    ext = f.name.rsplit(".", 1)[-1].lower()
                    parsed = extract_document_text(f.name, raw_bytes, ext)
                    processed = preprocess_document(parsed)
                    st.session_state.uploaded_docs[f.name] = processed

    # Display Uploaded Documents Status Cards
    if st.session_state.uploaded_docs:
        st.markdown("---")
        st.subheader(f"Uploaded Documents Inventory ({len(st.session_state.uploaded_docs)})")
        
        for name, doc_data in list(st.session_state.uploaded_docs.items()):
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1])
                with c1:
                    st.markdown(f"**📄 {doc_data['name']}**")
                    if doc_data.get("error"):
                        st.error(f"⚠️ {doc_data['error']}")
                    else:
                        st.caption(f"Type: {doc_data['type']} | Size: {doc_data['file_size_kb']} KB")
                with c2:
                    st.write(f"📊 **{doc_data['word_count']:,}** words")
                    st.caption(f"{doc_data['char_count']:,} characters")
                with c3:
                    st.write(f"🧩 **{doc_data.get('passage_count', 0)}** passages")
                    st.caption(f"~{doc_data.get('avg_passage_length', 0)} words/passage")
                with c4:
                    if doc_data.get("error"):
                        st.warning("Skipped")
                    else:
                        st.success("✓ Ready for Analysis")
                with c5:
                    if st.button("🗑️", key=f"del_{name}"):
                        del st.session_state.uploaded_docs[name]
                        st.session_state.analysis_results = None
                        st.rerun()

        st.markdown("---")
        if len(st.session_state.uploaded_docs) < 2:
            st.warning("⚠️ Please upload at least **2 readable documents** to run comparative similarity analysis.")
        else:
            if st.button("🚀 Run Comprehensive ML Similarity Analysis", type="primary", use_container_width=True):
                with st.spinner("Generating TF-IDF matrices and computing SentenceTransformer neural embeddings..."):
                    valid_docs = {
                        k: v for k, v in st.session_state.uploaded_docs.items()
                        if not v.get("error") and v.get("word_count", 0) >= 5
                    }
                    if len(valid_docs) < 2:
                        st.error("At least 2 documents with readable text (5+ words) are required.")
                    else:
                        try:
                            emb_dict = get_document_embeddings(valid_docs)
                            sim_results = compute_all_similarities(
                                valid_docs,
                                emb_dict,
                                alpha=lex_weight,
                                beta=sem_weight,
                                high_thresh=high_threshold,
                                mod_thresh=mod_threshold
                            )
                            st.session_state.analysis_results = {
                                "docs": valid_docs,
                                "embeddings": emb_dict,
                                "similarity": sim_results
                            }
                            # Log history
                            st.session_state.analysis_history.append({
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "doc_count": len(valid_docs),
                                "top_similarity": float(sim_results["pairs_df"].iloc[0]["hybrid_similarity"]) if not sim_results["pairs_df"].empty else 0.0
                            })
                            st.success("Analysis complete! Switch to the **📊 Analyze & Overview** tab to view results.")
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")

# -----------------------------------------------------------------------------
# TAB 3: ANALYZE & OVERVIEW DASHBOARD
# -----------------------------------------------------------------------------
with tab_overview:
    if not st.session_state.analysis_results:
        st.info("👈 Please upload at least 2 documents and click **Run Comprehensive ML Similarity Analysis** in the Upload tab.")
    else:
        results = st.session_state.analysis_results
        sim_data = results["similarity"]
        pairs_df = sim_data["pairs_df"]
        names = sim_data["names"]

        st.subheader("📊 Document Similarity Dashboard")

        # Top Executive Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Total Documents</div><div class="metric-value">{len(names)}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Pairwise Comparisons</div><div class="metric-value">{len(pairs_df)}</div></div>', unsafe_allow_html=True)
        with m3:
            top_val = f"{pairs_df.iloc[0]['hybrid_similarity']:.1%}" if not pairs_df.empty else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-title">Highest Similarity</div><div class="metric-value" style="color:#4F46E5;">{top_val}</div></div>', unsafe_allow_html=True)
        with m4:
            avg_val = f"{pairs_df['hybrid_similarity'].mean():.1%}" if not pairs_df.empty else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-title">Average Similarity</div><div class="metric-value">{avg_val}</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🏆 Pairwise Document Similarity Rankings")
        
        if pairs_df.empty:
            st.info("No pairwise comparisons available.")
        else:
            view_df = pairs_df.copy()
            view_df["Hybrid Score"] = (view_df["hybrid_similarity"] * 100).round(1).astype(str) + "%"
            view_df["Lexical (TF-IDF)"] = (view_df["lexical_similarity"] * 100).round(1).astype(str) + "%"
            view_df["Semantic (Embedding)"] = (view_df["semantic_similarity"] * 100).round(1).astype(str) + "%"

            st.dataframe(
                view_df[["document_a", "document_b", "Hybrid Score", "Lexical (TF-IDF)", "Semantic (Embedding)", "relationship"]],
                column_config={
                    "document_a": "Document A",
                    "document_b": "Document B",
                    "relationship": "Relationship Band"
                },
                use_container_width=True,
                hide_index=True
            )

            # CSV Download Button
            csv_bytes = generate_csv_report(pairs_df)
            st.download_button(
                "📥 Download Pair Rankings (CSV)",
                csv_bytes,
                "doctwin_similarity_rankings.csv",
                "text/csv",
                type="secondary"
            )

# -----------------------------------------------------------------------------
# TAB 4: SIMILARITY MATRIX
# -----------------------------------------------------------------------------
with tab_matrix:
    if not st.session_state.analysis_results:
        st.info("👈 Run analysis in the Upload tab first.")
    else:
        results = st.session_state.analysis_results
        sim_data = results["similarity"]

        st.subheader("🗺️ Document Similarity Heatmap Matrices")
        
        matrix_type = st.radio(
            "Select Similarity Matrix view:",
            ["Hybrid Similarity Matrix", "Lexical (TF-IDF) Matrix", "Semantic (SentenceTransformer) Matrix"],
            horizontal=True
        )

        if matrix_type == "Hybrid Similarity Matrix":
            fig_mat = create_similarity_heatmap(sim_data["hybrid_df"], title=f"Hybrid Matrix ({lex_weight:.0%} Lexical + {sem_weight:.0%} Semantic)", colorscale="Viridis")
        elif matrix_type == "Lexical (TF-IDF) Matrix":
            fig_mat = create_similarity_heatmap(sim_data["lexical_df"], title="Lexical (TF-IDF) Similarity Matrix", colorscale="Blues")
        else:
            fig_mat = create_similarity_heatmap(sim_data["semantic_df"], title="Semantic (SentenceTransformer) Similarity Matrix", colorscale="Teal")

        st.plotly_chart(fig_mat, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: PASSAGE MATCHES EXPLORER
# -----------------------------------------------------------------------------
with tab_passages:
    if not st.session_state.analysis_results:
        st.info("👈 Run analysis in the Upload tab first.")
    else:
        results = st.session_state.analysis_results
        sim_data = results["similarity"]
        pairs_df = sim_data["pairs_df"]
        docs = results["docs"]
        embeddings = results["embeddings"]

        st.subheader("🔍 Matching Passage Explorer")
        st.caption("Inspect side-by-side passages with highest textual and semantic similarity between any pair of documents.")

        if pairs_df.empty:
            st.info("No document pairs available.")
        else:
            pair_options = [f"{r['document_a']}  ↔  {r['document_b']}" for _, r in pairs_df.iterrows()]
            selected_pair_str = st.selectbox("Select a Document Pair to Inspect:", options=pair_options)
            
            selected_idx = pair_options.index(selected_pair_str)
            chosen_pair = pairs_df.iloc[selected_idx]

            doc_a_name = chosen_pair["document_a"]
            doc_b_name = chosen_pair["document_b"]

            st.write(f"Pair Score: **Hybrid {chosen_pair['hybrid_similarity']:.1%}** (Lexical: {chosen_pair['lexical_similarity']:.1%} | Semantic: {chosen_pair['semantic_similarity']:.1%})")

            # Passage Matches calculation
            matches = compare_passages(
                docs[doc_a_name],
                docs[doc_b_name],
                embeddings[doc_a_name],
                embeddings[doc_b_name],
                method=method_option,
                alpha=lex_weight,
                beta=sem_weight,
                top_n=passage_top_n,
                threshold=0.1
            )

            if not matches:
                st.info("No passage matches found for this document pair.")
            else:
                st.markdown(f"#### Top {len(matches)} Matching Passages")
                for i, match in enumerate(matches, 1):
                    with st.expander(f"Match #{i} — Similarity: {match['display_score']:.1%} (Hybrid: {match['hybrid_score']:.1%} | Lexical: {match['lexical_score']:.1%} | Semantic: {match['semantic_score']:.1%})", expanded=(i == 1)):
                        col_l, col_r = st.columns(2)
                        with col_l:
                            st.markdown(f"**{doc_a_name} (Passage #{match['passage_idx_a']+1})**")
                            st.markdown(f'<div class="passage-box">{match["text_a"]}</div>', unsafe_allow_html=True)
                        with col_r:
                            st.markdown(f"**{doc_b_name} (Passage #{match['passage_idx_b']+1})**")
                            st.markdown(f'<div class="passage-box">{match["text_b"]}</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 6: DOCUMENT DNA
# -----------------------------------------------------------------------------
with tab_dna:
    if not st.session_state.analysis_results:
        st.info("👈 Run analysis in the Upload tab first.")
    else:
        results = st.session_state.analysis_results
        sim_data = results["similarity"]
        pairs_df = sim_data["pairs_df"]
        names = sim_data["names"]

        st.subheader("🧬 Document DNA — Interactive Relationship Graph")
        st.caption("Visualizing inter-document similarity networks. Nodes represent documents; connecting lines indicate similarity above the configured threshold.")

        graph_fig = build_document_dna_graph(pairs_df, names, threshold=rel_threshold)
        st.plotly_chart(graph_fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 7: SEARCH
# -----------------------------------------------------------------------------
with tab_search:
    if not st.session_state.uploaded_docs:
        st.info("👈 Upload documents first to enable content search.")
    else:
        st.subheader("🔎 Document Content Search")
        query = st.text_input("Search terms or keywords across all uploaded documents:", placeholder="e.g. machine learning, cloud, diagnosis")
        
        if query.strip():
            st.markdown(f"#### Search Results for: *'{query}'*")
            found_count = 0
            
            for doc_name, doc_info in st.session_state.uploaded_docs.items():
                text = doc_info.get("clean_text", "")
                matches = list(re.finditer(re.escape(query), text, re.IGNORECASE))
                if matches:
                    found_count += 1
                    with st.expander(f"📄 {doc_name} — {len(matches)} occurrences found", expanded=True):
                        # Display snippets
                        for m in matches[:5]:
                            start = max(0, m.start() - 60)
                            end = min(len(text), m.end() + 60)
                            snippet = text[start:end].replace("\n", " ")
                            highlighted = snippet.replace(m.group(0), f"**:blue[{m.group(0)}]**")
                            st.markdown(f"- ...{highlighted}...")
                        if len(matches) > 5:
                            st.caption(f"+ {len(matches)-5} more occurrences")

            if found_count == 0:
                st.warning(f"No occurrences of '{query}' were found in any uploaded document.")

# -----------------------------------------------------------------------------
# TAB 8: REPORTS & HISTORY
# -----------------------------------------------------------------------------
with tab_reports:
    st.subheader("📑 Reports & Analysis History")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        sim_data = results["similarity"]
        pairs_df = sim_data["pairs_df"]
        docs = results["docs"]
        embeddings = results["embeddings"]

        st.markdown("#### Downloadable Executive Reports")
        col_rep1, col_rep2 = st.columns(2)
        
        with col_rep1:
            csv_bytes = generate_csv_report(pairs_df)
            st.download_button(
                "📥 Download Similarity Rankings (CSV)",
                csv_bytes,
                "doctwin_results.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col_rep2:
            # Generate top matches dictionary for report
            top_matches_dict = {}
            if not pairs_df.empty:
                for _, row in pairs_df.iterrows():
                    p_name = f"{row['document_a']} ↔ {row['document_b']}"
                    pm = compare_passages(
                        docs[row['document_a']],
                        docs[row['document_b']],
                        embeddings[row['document_a']],
                        embeddings[row['document_b']],
                        method=method_option,
                        alpha=lex_weight,
                        beta=sem_weight,
                        top_n=2
                    )
                    top_matches_dict[p_name] = pm

            txt_report_str = generate_txt_report(sim_data, docs, top_matches_dict)
            st.download_button(
                "📄 Download Executive Report (TXT)",
                txt_report_str.encode("utf-8"),
                "doctwin_executive_report.txt",
                "text/plain",
                use_container_width=True
            )

    st.divider()
    st.markdown("#### 📜 Session Analysis History")
    if not st.session_state.analysis_history:
        st.info("No session analysis runs logged yet.")
    else:
        hist_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 9: ABOUT & METHODOLOGY
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ About DocuTwin & Machine Learning Methodology")
    st.markdown("""
    ### Machine Learning Architecture & Technical Pipeline

    DocuTwin combines classical lexical statistics with modern transformer-based neural representations:

    1. **Lexical Matching (TF-IDF + Cosine Similarity)**
       - Extracts sublinear word and character n-grams $(1, 2)$.
       - Constructs high-dimensional term-frequency vectors and computes cosine similarity.
       - Highly sensitive to exact phrasing, shared terminology, and verbatim duplication.

    2. **Semantic Representation (SentenceTransformers)**
       - Utilizes the `all-MiniLM-L6-v2` dense embedding model (384 dimensions).
       - Maps document passages into a unified dense vector space where conceptually similar sentences map to nearby points.
       - Detects rephrasing, synonym substitutions, and conceptual alignment even when vocabulary differs.

    3. **Hybrid Scoring Engine**
       - Evaluates $\\text{Hybrid Score} = \\alpha \\cdot \\text{Lexical} + \\beta \\cdot \\text{Semantic}$.
       - Default weights: $\\alpha = 0.40$ (Lexical) and $\\beta = 0.60$ (Semantic).

    ### Important Limitation & Ethics Disclaimer
    - **Similarity ≠ Plagiarism:** Textual and semantic resemblance occur naturally in academic citations, literature reviews, template agreements, and standard technical terminology.
    - Similarity scores must be interpreted by human reviewers and cannot independently establish intent, authorship, or academic misconduct.
    """)
