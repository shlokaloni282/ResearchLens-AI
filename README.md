# ResearchLens AI

**ResearchLens AI** is an AI research assistant designed for scientific papers. It extracts page-level text from dense academic PDFs, applies specialized chunking strategies (Fixed, Recursive, and Semantic), and provides a verifiable RAG evaluation framework with strict provenance tracking.

---

## Features

- **Multi-Page PDF Ingestion**: Page-by-page extraction preserving exact page numbering.
- **Academic Domain Cleaning**: Strips IEEE Xplore watermark footers, trims trailing references/bibliographies, and protects scientific abbreviations (`Fig.`, `Eq.`, Roman numerals).
- **Three Chunking Strategies**:
  - `fixed`: Sliding character window.
  - `recursive`: Hierarchical character/paragraph/sentence splitting.
  - `semantic`: Embedding-based sentence similarity clustering using `sentence-transformers/all-MiniLM-L6-v2` and `pysbd`.
- **In-Memory Vector Search**: Normalized dense embeddings, cosine similarity, cross-paper search, and paper-specific filtering.
- **LLM-Powered Grounded Q&A**:
  - Dedicated answer generation service (`backend/app/services/qa_service.py`).
  - Google Gemini API integration (modular direct REST implementation).
  - Strict grounding constraints preventing hallucinations.
  - Provenance verification: citations with paper filename and page number are cross-checked against retrieved chunks.
  - Explicit abstention when context lacks sufficient evidence.
- **RAG Evaluation Framework**:
  - Ground truth dataset with factoid, methodology, and negative/unanswerable test questions.
  - Automated chunking scorecard benchmarking Fact Retention %, Boundary Coherence, and Size Distribution.
  - Automated retrieval evaluation calculating Hit Rate@1, @3, @5 and MRR.

---

## Project Structure

```text
ResearchLens-AI/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application entrypoint
│   │   ├── routes/
│   │   │   ├── papers.py               # POST /papers/upload
│   │   │   └── qa.py                   # POST /qa/ask
│   │   ├── schemas/
│   │   │   └── qa.py                   # QuestionRequest & AnswerResponse schemas
│   │   └── services/
│   │       ├── chunking.py             # Fixed, recursive, and semantic chunking
│   │       ├── retrieval.py            # InMemoryVectorIndex with cosine similarity
│   │       └── qa_service.py           # Grounded Q&A with Gemini & provenance verification
│   └── tests/
│       ├── test_chunking.py            # Unit tests for chunking algorithms
│       ├── test_retrieval.py           # Unit tests for vector search & filtering
│       └── test_qa.py                  # Unit tests for grounded Q&A, citations & abstention
├── data/
│   └── benchmark_papers/               # Curated evaluation PDFs
├── evaluation/
│   ├── data/
│   │   └── ground_truth.json           # Golden benchmark questions & facts
│   ├── reports/                        # Saved evaluation run JSON reports
│   ├── eval_chunking.py                # Automated chunking evaluation scorecard
│   └── eval_retrieval.py               # Automated retrieval evaluation (Hit Rate & MRR)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Environment Setup

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backend API

```powershell
uvicorn backend.app.main:app --reload
```
API Documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Configure Gemini API Key (Optional for production)

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
# Optional: customize model (defaults to gemini-1.5-flash)
$env:GEMINI_MODEL="gemini-1.5-flash"
```

### 4. Ask Questions via API

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/qa/ask" `
  -ContentType "application/json" `
  -Body '{"question": "What is the power dissipation of the RNG at 100 MHz?", "top_k": 3}'
```

### 5. Run Unit Tests

```powershell
$env:PYTHONPATH="."
pytest -v
```

---

## RAG Evaluation

### Run Chunking Evaluation

Benchmark Fixed vs. Recursive vs. Semantic chunking against the ground-truth facts:

```powershell
$env:PYTHONPATH="."
python evaluation/eval_chunking.py --pdf data/benchmark_papers/rng_paper.pdf
```

The script evaluates:
- **Fact Retention Rate (%)**: Percentage of critical domain facts that survive unbroken within a single chunk.
- **Boundary Coherence**: Detection of dangling abbreviations (`Fig.`, `Eq.`) or broken sentences.
- **Size Distribution**: Chunk count, median, min, max, and fragment counts (< 50 chars).
- Output is displayed in a console scorecard and saved to `evaluation/reports/`.
