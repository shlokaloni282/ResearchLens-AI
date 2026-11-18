# ResearchLens AI — Knowledge Transfer

**Purpose:** Give Antigravity (or another coding IDE) enough verified context to continue ResearchLens AI without confusing ideas with completed work.

**Status key**

- **Confirmed implemented/tested:** supported directly by the referenced conversation history.
- **Confirmed implemented, test status unclear:** described or listed as present, but no successful test result was captured.
- **Planned:** discussed as a future milestone, not shown as built.
- **Verify in repository:** conversation context does not establish the precise implementation or current code state.

> The source checkout was not present in this handoff workspace when this document was assembled. Treat repository structure, exact dependency versions, and exact function signatures as items to inspect in the real project before editing. This document does not claim that an endpoint or feature exists unless the history confirms it.

## 1. Project overview

**ResearchLens AI** is an AI research assistant for scientific papers. The initial vertical slice is a Python backend that accepts a research-paper PDF, extracts page text, and divides it into chunks using selectable strategies. The broader product direction is to make scientific papers easier to search, understand, cite, and compare across papers.

The project was being developed in **Antigravity**. The known implemented work is intentionally a small foundation: PDF upload/extraction plus chunking. The conversation described embeddings, vector storage, retrieval, multi-paper Q&A, and citations as the next milestone; these must be treated as unbuilt until verified in source.

## 2. Problem statement and vision

Researchers and students need to work across long, dense scientific papers. Finding a claim, its context, and the page that supports it can take time. Comparing methods and results across papers adds further overhead.

**Vision:** provide a paper-grounded research assistant that can ingest papers, retrieve relevant passages, answer questions with traceable page citations, and eventually compare evidence across multiple papers.

**Problem being addressed:** turn uploaded paper PDFs into structured, retrievable, citation-aware material, then build useful research workflows on top of it.

## 3. Goals

### Initial engineering goals

1. Accept a PDF through an HTTP API.
2. Extract text page by page and retain page numbering.
3. Produce chunks using fixed, recursive, and semantic strategies.
4. Expose chunking settings to callers and permit strategy experiments.
5. Establish tests around chunking and paper upload behavior.

### Product goals (planned)

1. Search a paper collection with natural-language questions.
2. Answer from retrieved paper passages and cite paper/page.
3. Support multiple papers and comparison questions.
4. Make retrieval and answer quality measurable and improvable.

## 4. Scope and boundaries

### Confirmed foundation

- Backend identifies itself in the observed API documentation as **ResearchLens AI 0.1.0**, described as “An AI research assistant for scientific papers.”
- API documentation was served through FastAPI-style OpenAPI/Swagger UI.
- A PDF upload route supports a chunking strategy and chunk sizing options.
- The upload response includes the filename, total page count, and page-level extracted text.
- Fixed-size, recursive, and semantic chunking are represented in the API schema.
- Conversation history reports successful fixed and recursive upload/chunking experiments.

### Planned or not confirmed

- Embedding generation and embedding model/provider.
- Vector database/index and persistence.
- Semantic question answering or chat.
- Multi-paper ingestion/search and comparison.
- Citation generation in model answers (distinct from retaining page numbers in extracted content).
- Frontend/user accounts, authentication, cloud deployment, or production operations.
- A database schema or durable paper/chunk storage.

Do not implement assumptions from this section as though they were existing behavior. First inspect the source and current Git state.

## 5. Features and user flow

### Current API flow (confirmed)

1. Caller sends a PDF as multipart form data to `POST /papers/upload`.
2. Caller may choose `strategy`, `chunk_size`, and `overlap`.
3. The service extracts text by page; observed example returned four pages.
4. The selected chunking strategy processes extracted text.
5. The API returns paper/page data and, according to conversation-reported checks, chunk records with identifiers, paper association, strategy and page metadata. Inspect actual response model and code before relying on exact field names.

### Intended researcher flow (planned)

1. Upload one or more papers.
2. Review paper metadata and processing status.
3. Ask a question or search for a concept.
4. Inspect retrieved excerpts with paper and page references.
5. Ask for synthesis/comparison across selected papers.
6. Follow citations back to the supporting passages/pages.

Only the upload/extraction/chunking portion is confirmed as implemented.

## 6. Architecture and backend structure

### Confirmed architectural shape

- Python backend with a FastAPI-style HTTP API and OpenAPI 3.1 docs.
- Separation into route, service, and test modules was reflected in the staged-file list in the conversation.
- PDF extraction and chunking are separate services.
- A chunk-comparison utility was listed in the staged files.

### Conversation-reported file layout

The following paths appeared in a Git staged-file listing. Verify these still exist and inspect their contents:

```text
.gitignore
backend/
  __init__.py
  app/
    main.py
    routes/
      papers.py
    services/
      chunking.py
      compare_chunking.py
      pdf_service.py
    tests/
      test_chunking.py
      test_papers.py
requirements.txt
```

There was also a modified `backend/app/.gitignore`. A root `.gitignore` was added to ignore local environments, bytecode, environment secrets, and pytest cache. Actual ignore rules should be checked.

### Proposed logical pipeline (partially built)

```text
PDF upload
  -> PDF/page text extraction
  -> chunk strategy
  -> page-aware chunk records
  -> [planned] embeddings
  -> [planned] vector index
  -> [planned] retrieval
  -> [planned] grounded answer with citations
```

The bracketed steps were described as upcoming work, not completed implementation.

## 7. Tech stack

| Area | Evidence/status |
|---|---|
| Language | Python backend, confirmed by project paths and setup context |
| HTTP/API | FastAPI-style app with OpenAPI/Swagger UI; observed local URL was `http://127.0.0.1:8000` |
| PDF parsing | A `pdf_service.py` module was listed; exact parser/library is not established here |
| Chunking | Fixed, recursive, semantic strategies exposed by upload API |
| Tests | `backend/tests/test_chunking.py` and `backend/tests/test_papers.py` were listed |
| Dependencies | `requirements.txt` was listed; exact packages and versions must be read from repository |
| Embeddings/vector store/LLM | Planned; provider and model not established |
| Frontend/database/deployment | Not established |

Do not select a provider or add a paid dependency without checking existing project decisions and constraints.

## 8. PDF extraction, chunks, and page citations

### PDF extraction

The observed upload response for the example paper reported:

- Filename: `100_MHz_Random_Number_Generator_Design_Using_Interleaved_Metastable_NAND_NOR_Latches.pdf`
- Total pages: `4`
- Page objects containing `page_number` and extracted `text`

This confirms page-oriented text extraction for that example. The conversation notes that basic extraction included figure captions, references, and IEEE footer information. OCR for scanned PDFs, layout/table preservation, and extraction quality handling were not confirmed.

### Chunking strategies

The API schema showed:

- `strategy`: `fixed | recursive | semantic` (observed default in separate Swagger interactions included recursive and semantic; verify actual default from code).
- `chunk_size`: integer, min 1, max 5000; observed value 500.
- `overlap`: integer, min 0, max 4999; observed value 50.

**Fixed-size — confirmed tested:** upload was tested. Exact tokenizer/character semantics are not preserved in the conversation; inspect code.

**Recursive — confirmed tested:** same PDF uploaded with `chunk_size=500`, `overlap=50`; history reported HTTP 200, 4 extracted pages, 35 chunks, page numbers preserved, unique chunk IDs, paper ID, and strategy recorded. The observed sample boundary split a phrase between adjacent chunks; paragraph/sentence boundary handling could be refined.

**Semantic — implementation reported, test result not captured:** conversation said current implementation uses sentence similarity. The same chunk size/overlap behavior does not apply in the same way as fixed and recursive. A Swagger upload was prepared/run in the history, but there is no clear success result for semantic chunking. Confirm by running/reviewing the real project only if appropriate.

### Metadata and citations

Page numbers were reported preserved in recursive chunks. A page-aware chunk record is essential for source citations. Verify which metadata is actually returned/stored (paper ID, chunk ID, filename, page number, strategy, offsets, etc.). Do not infer that conversational answers already cite pages: answer generation/retrieval was a later milestone.

Recommended citation shape for future retrieval: paper identifier/title + page number + quoted or displayed supporting chunk. Preserve page provenance through every transform and never cite a page that cannot be tied to the retrieved text.

## 9. Multi-paper retrieval and comparison (planned)

The stated next milestone was:

**Embeddings → Vector database → Retrieval → Multi-paper Q&A → Citations**

No embedding, vector DB, retrieval, Q&A, or cross-paper comparison endpoint is confirmed built.

Suggested staged design, subject to repository conventions:

1. Add persistent paper/chunk records and stable identifiers.
2. Create embeddings for chunks while retaining paper/page metadata.
3. Index vectors with a filterable paper identifier.
4. Retrieve top passages for a question; optionally filter selected papers.
5. Build answers only from supplied passages and return source references.
6. Add comparison mode that groups evidence by paper and distinguishes agreement, disagreement, and missing evidence.
7. Return retrieved evidence with the answer so the UI can display verifiable citations.

## 10. Evaluation and experiments

### Experiments recorded

- Fixed chunking: conversation marked tested.
- Recursive chunking: tested successfully with the sample four-page PDF, size 500, overlap 50; 35 chunks were reported.
- Semantic chunking: identified as the next test; no confirmed successful result recorded.
- A `compare_chunking.py` utility was among files listed for staging, but its implemented metrics and outputs are unknown.

### Evaluation plan (planned)

Keep extraction/chunking evaluation separate from answer-quality evaluation.

**Chunking experiments**
- Compare fixed, recursive, semantic on the same PDFs and size settings where comparable.
- Track chunk count, size distribution, boundary quality, sentence/paragraph splits, page provenance completeness, processing time, and failure rate.
- Record reproducible inputs and configuration. Semantic chunking may not use size/overlap identically; document the actual algorithm rather than forcing false parity.

**Retrieval experiments (future)**
- Create a small question set with expected supporting paper/page passages.
- Measure Recall@k, Precision@k, and MRR (or nDCG where graded relevance is available).
- Compare embedding/model/index settings on the same dataset.

**Answer/citation experiments (future)**
- Score factual support/faithfulness, answer relevance, citation correctness (paper and page), and abstention when evidence is absent.
- Manually review disagreements and multi-paper synthesis.
- Keep benchmark questions, expected evidence, and run settings versioned.

These metrics are recommendations for planned work, not metrics already computed.

## 11. API endpoints

Observed in the OpenAPI/Swagger history:

| Method | Path | Status | Observed behavior |
|---|---|---|---|
| GET | `/` | Confirmed listed | Home endpoint; exact response not captured |
| GET | `/health` | Confirmed listed | Health check; exact response not captured |
| POST | `/papers/upload` | Confirmed listed and exercised | Multipart PDF upload; query parameters include strategy, chunk_size, overlap; response includes filename, total_pages, pages/page text |

Example request captured in Swagger:

```bash
curl -X POST \
  'http://127.0.0.1:8000/papers/upload?strategy=recursive&chunk_size=500&overlap=50' \
  -H 'accept: */*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@100_MHz_Random_Number_Generator_Design_Using_Interleaved_Metastable_NAND_NOR_Latches.pdf;type=application/pdf'
```

The exact upload response was truncated in the conversation. Do not assume response fields beyond filename, page count, page text, and the separately reported chunk metadata checks without inspecting implementation.

**Planned endpoints:** none were specified by path. Search/ask, paper listing/detail, comparison, and retrieval endpoints remain design tasks unless they exist in the repository.

## 12. Setup and run commands

The history confirms a local development server at port 8000 and use of a Python virtual environment. Exact install/start commands were not captured. These are conventional commands to check against the repository's README and dependency list before use:

```powershell
# From the repository root
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

If the import path differs, inspect `backend/app/main.py` and use the project's documented command. Swagger was available at `http://127.0.0.1:8000/docs`; health endpoint at `http://127.0.0.1:8000/health`.

Possible test invocation, inferred from the listed pytest cache and test files; verify pytest is installed and repo instructions first:

```powershell
python -m pytest
```

These are setup suggestions, not commands documented as having been run successfully in the referenced history.

## 13. Tests and verification state

- Test modules `backend/tests/test_chunking.py` and `backend/tests/test_papers.py` were present in the staged-file list.
- A successful recursive API interaction was reported, including extraction and chunk metadata checks.
- Fixed chunking was marked tested.
- The conversation does not provide test names, a test-run transcript, coverage, or pass/fail totals. Do not claim a full test suite passed.
- Semantic result is not confirmed.
- No retrieval or answer-generation tests exist in confirmed context.

## 14. Git workflow recorded

The conversation described this workflow from the ResearchLens repository root:

```powershell
git status
git add .gitignore backend requirements.txt
git status
git commit -m "Implement PDF extraction and chunking strategies"
git push origin main
git status
```

A root `.gitignore` was added because the local `.venv/` and `__pycache__/` appeared as untracked. The suggested rules included `.venv/`, `__pycache__/`, `*.py[cod]`, `.env`, and `.pytest_cache/`. The conversation recommended checking that secrets and the virtual environment were not staged.

**Important status distinction:** commands to commit and push were recommended, but no successful commit/push result is included in the available history. Confirm current branch, working tree, remotes, and commit log before doing any Git operation.

## 15. Known limitations and risks

Confirmed or directly noted:

- Basic PDF extraction may retain headers, footers, figure captions, and reference text.
- Recursive chunk boundaries can split phrases; boundary handling may need improvement.
- Semantic chunking's size/overlap behavior differs from fixed/recursive.
- The confirmed sample is a single four-page paper; broad document robustness is unproven.
- No OCR/scanned-document support is confirmed.
- No embeddings, vector search, RAG answers, or multi-paper comparison are confirmed.
- No answer-level citation behavior is confirmed.
- No persistence, UI, authentication, or deployment behavior is established.
- No full test-suite result is available.

Engineering concerns to address as the product grows: malformed/oversized PDF handling, empty extraction, duplicate uploads, temporary-file cleanup, parser errors, page/chunk metadata integrity, request limits, and explicit errors. These are review items, not known existing defects.

## 16. Phased end-to-end roadmap

### Phase 0 — Repository handover

- Open the actual ResearchLens repository in Antigravity.
- Read README, requirements, source, tests, Git status, and branch history.
- Reconcile this document with the code; update any stale status statements.
- Preserve user work and secrets; do not overwrite unrelated changes.

### Phase 1 — Stabilize ingestion (current foundation)

- Confirm upload validation, extraction behavior, response schema, and chunk metadata.
- Run existing tests and add coverage for invalid PDFs, empty pages, page numbering, and each strategy.
- Verify fixed/recursive/semantic behavior and document parameter semantics.
- Improve PDF text cleanup and boundary handling only with regression coverage.

### Phase 2 — Make chunking comparison reproducible

- Define comparable input corpus and settings.
- Emit measurable chunk statistics and boundary diagnostics.
- Explain semantic strategy differences and retain experiment outputs.
- Select a default strategy from evidence rather than assumption.

### Phase 3 — Paper/chunk persistence

- Agree data models and stable paper/chunk IDs.
- Persist source metadata and page provenance.
- Add list/detail/delete or reprocess behavior only as required by product scope.
- Define duplicate-upload and failure/retry behavior.

### Phase 4 — Embeddings and vector retrieval

- Choose embedding provider/model and vector store after reviewing constraints.
- Generate embeddings in an explicit, repeatable pipeline.
- Store metadata filters and support per-paper collection selection.
- Build a retrieval endpoint/service with evidence payloads.
- Evaluate retrieval against labeled questions using Recall@k/Precision@k/MRR.

### Phase 5 — Grounded Q&A with citations

- Add question answering over retrieved passages.
- Require answers to cite source paper and page for material claims.
- Return citations as structured data, not just prompt-formatted prose.
- Abstain or state evidence gaps when retrieved material is insufficient.
- Evaluate factual support and citation accuracy.

### Phase 6 — Multi-paper comparison

- Support user-selected paper sets.
- Retrieve evidence separately and jointly across papers.
- Present comparison by topic/claim, with per-paper citations and explicit disagreement/uncertainty.
- Test questions whose papers agree, conflict, or omit the requested evidence.

### Phase 7 — Usability and operational readiness

- Build or connect a user-facing interface if in scope.
- Add observability, configuration/secrets handling, upload limits, and clear error messages.
- Document local setup, deployment, and rollback.
- Revisit security, privacy, retention, and licensing before handling real research libraries.

## 17. Instructions for the next coding agent

1. Start by inspecting the repository and its instructions (README, AGENTS.md, tests, Git status); this document is context, not a source of truth over code.
2. Preserve confirmed functionality and any user changes.
3. Separate implemented behavior from proposed work in commits and documentation.
4. Before adding RAG, first trace page provenance from extraction through chunks; citations depend on this.
5. Keep evaluation data and settings reproducible.
6. Report what was actually run and what remains unverified. Do not mark planned features complete based on this roadmap.

## 18. Provenance of this handoff

This document is based on the available “NEW PROJS” conversation history. The available history included: the project PDF upload/chunking API documentation, conversation summaries of fixed and recursive chunk tests, the reported staged file list, and a stated future milestone for embeddings/vector DB/retrieval/multi-paper Q&A/citations. The original four-page sample paper discussed there concerns a 100 MHz random-number-generator design and is an ingestion test input, not the ResearchLens product specification.

Details absent from the available conversation have been explicitly marked for verification rather than filled in by guesswork.

