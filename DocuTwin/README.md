# DocuTwin — Intelligent Document Similarity & Relationship Analyzer

DocuTwin is a machine-learning-based web application that allows users to upload multiple documents (PDF, DOCX, TXT) and analyze their textual and semantic similarity. 

The system identifies document-to-document relationships, passage matches, similarity scores, and presents them in an interactive dashboard with similarity matrix heatmaps, a **Document DNA** relationship graph, and downloadable reports.

---

## 📌 Important Educational & ML Disclaimer

> **Similarity does not prove plagiarism or copyright infringement.**
>
> DocuTwin measures lexical overlap (TF-IDF) and semantic embedding proximity (SentenceTransformers). High similarity scores indicate textual or conceptual resemblance and **do not independently establish plagiarism, copying, intent, or authorship**. Similarity values can result from standard citations, template structures, shared technical vocabulary, or literature reviews.

---

## 🏗️ Architecture & Project Structure

```text
DocuTwin/
├── app.py                      # Main Streamlit web application & multi-tab dashboard UI
├── requirements.txt            # Python dependency requirements
├── README.md                   # Complete documentation and setup guide
├── .gitignore                  # Git ignore file
│
├── core/                       # Machine Learning & Core Processing Engine
│   ├── __init__.py
│   ├── document_parser.py      # Multi-format text extraction (PDF via PyMuPDF, DOCX via python-docx, TXT)
│   ├── preprocessing.py        # Text cleaning and passage chunking
│   ├── embeddings.py           # Cached SentenceTransformer (all-MiniLM-L6-v2) embedding engine
│   ├── similarity.py           # TF-IDF, Cosine Similarity, and Hybrid Scoring matrices
│   └── passage_matching.py     # Side-by-side passage alignment and score comparison
│
├── visualization/              # Data Visualization Modules
│   ├── __init__.py
│   ├── similarity_matrix.py    # Interactive Plotly similarity matrix heatmaps
│   └── document_graph.py       # Document DNA interactive Plotly relationship network graph
│
├── reports/                    # Executive Report Generation
│   ├── __init__.py
│   └── report_generator.py     # Downloadable CSV rankings & plain-text executive reports
│
└── sample_data/                # Demonstration Datasets
    ├── README.md
    ├── Research_Paper_A.txt    # Original healthcare AI paper abstract & intro
    ├── Research_Paper_B.txt    # Paraphrased version (high semantic similarity)
    ├── Research_Paper_C.txt    # Near-duplicate version (high lexical & semantic similarity)
    └── Project_Report.txt      # Unrelated cloud computing report (low similarity)
```

---

## 🔬 Machine Learning Methodology

DocuTwin combines classical lexical statistics with modern transformer-based neural representations:

1. **Lexical Matching (TF-IDF + Cosine Similarity)**
   - Computes term frequency-inverse document frequency using word and character $n$-grams $(1, 2)$.
   - Sensitive to exact word matches, verbatim phrases, and identical terminology.

2. **Semantic Representation (SentenceTransformers)**
   - Utilizes the `all-MiniLM-L6-v2` dense embedding model (384 dimensions).
   - Encodes passages into a continuous vector space where semantically similar sentences map to nearby points.
   - Detects rephrased ideas, synonym substitutions, and conceptual alignment even when different words are used.

3. **Hybrid Similarity Formula**
   - Combines lexical and semantic scores using configurable weights:
     $$\text{Hybrid Score} = \alpha \cdot \text{Lexical Score} + \beta \cdot \text{Semantic Score}$$
   - Default weights: $\alpha = 0.40$ (Lexical) and $\beta = 0.60$ (Semantic).

4. **Relationship Classification Bands**
   - **Highly Similar**: $\ge 75\%$
   - **Moderately Similar**: $45\% \text{ to } 74\%$
   - **Low Similarity**: $< 45\%$

---

## 💻 Setup & Installation Guide

### Prerequisites
- Python **3.9** or higher installed on your computer.

### Step 1: Open Terminal in the Project Directory
Navigate to the project root directory:
```bash
cd DocuTwin
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv .venv

# macOS / Linux
python3 -m venv .venv
```

### Step 3: Activate Virtual Environment
```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Web Application

To launch the web app, run the following command in your terminal with the virtual environment activated:

```bash
streamlit run app.py
```

After running the command, your default browser will automatically open:
```text
Local URL: http://localhost:8501
```

---

## 🧪 Demonstration & Testing Walkthrough

1. **Launch App**: Execute `streamlit run app.py`.
2. **Load Sample Data**: On the **Upload & Process** tab, click **"📁 Load 4 Demo Documents"**.
3. **Run Analysis**: Click **"🚀 Run Comprehensive ML Similarity Analysis"**.
4. **Explore Dashboard**:
   - **Analyze & Overview**: View global similarity rankings and executive metrics.
   - **Similarity Matrix**: View interactive Heatmaps for Hybrid, Lexical, and Semantic scores.
   - **Passage Matches**: Inspect side-by-side passage alignments between Document A and Document B.
   - **Document DNA**: Adjust the similarity threshold slider to view the network connection graph.
   - **Search**: Perform full-text term searches across all uploaded files.
   - **Reports**: Download CSV rankings and TXT executive reports.

---

## 🔮 Future Enhancements

- OCR engine integration (Tesseract) for scanned image-based PDFs.
- Multilingual sentence transformer models for cross-lingual similarity.
- Persistent database storage (SQLite / PostgreSQL) for historical analysis tracking.
- Contradiction and NLI stance detection.
