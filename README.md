# ResearchLens AI 🔬
> **Intelligent Research-Paper Analysis & Comparative Literature Review System**

ResearchLens AI is a college-level, fully reproducible document intelligence system for academic researchers. Users upload multiple research paper PDFs to extract structured components, execute hybrid semantic search, perform grounded question answering with verifiable page-level citations, compare papers across methodology/datasets, synthesize literature reviews, and discover author-declared research gaps.

---

## 🌟 Core Features

- **Provider-Agnostic LLM Layer:** Seamlessly switch between Google Gemini API, local Ollama (100% offline privacy mode), or any OpenAI-compatible endpoint.
- **Section-Aware Chunking:** Intelligently preserves academic paper sections (`Abstract`, `Methodology`, `Results`, `Limitations`, `Conclusion`), handling two-column layouts and stripping headers/footers.
- **Evidence-Grounded Extraction:** Extracts 9 key research components (`objective`, `methodology`, `datasets`, `algorithms`, `metrics`, `results`, `limitations`, `future work`) with exact page citations and fuzzy quote verification.
- **Hybrid Search with RRF:** Fuses dense vector embeddings (FAISS) with lexical BM25 search via Reciprocal Rank Fusion ($k=60$) and Cross-Encoder reranking.
- **Verifiable RAG Chat:** Conversational QA with strict inline citations (`[Paper N, p. X]`) and an automated claim verification judge.
- **Faithful Cross-Paper Comparison:** Side-by-side matrices with automatic **Comparability Warnings** when papers evaluate on distinct benchmarks.
- **Literature Review & Gap Discovery:** Synthesizes structured reviews in IEEE/APA/BibTeX formats and clusters author-declared limitations to highlight unaddressed research gaps.
- **Academic Export:** One-click export to Markdown, DOCX, and PDF.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- (Optional) [Ollama](https://ollama.com/) for fully offline local operation
- (Optional) Tesseract OCR for scanned PDF processing

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-repo/ResearchLens_AI.git
cd ResearchLens_AI

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and set your preferred LLM provider:
```bash
cp .env.example .env
```
For Google Gemini (Free tier):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```
For Ollama (Local offline):
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

### 4. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Tests & Evaluation

```bash
# Run automated unit tests
pytest -v

# Run test suite with coverage (>= 70% requirement)
pytest --cov=core --cov=modules --cov=evaluation tests/

# Run reproducible evaluation harness & generate evaluation/RESULTS.md
python -m evaluation.report

# Generate human ground-truth labelling templates (Markdown & JSON) for papers
python -m evaluation.generate_label_template --all-db

# Run code linter
ruff check .
```

---

## 📐 System Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full architectural specifications, data flows, and query sequences.
See [docs/DECISIONS.md](docs/DECISIONS.md) for engineering design decisions and rationale.
