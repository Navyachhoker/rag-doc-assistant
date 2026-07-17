# RAG Document Assistant

A multimodal Retrieval-Augmented Generation (RAG) system that answers natural language questions over long, image-and-table-heavy technical PDF manuals. It retrieves relevant text passages, tables, and images from ingested documents and generates grounded answers with inline citations, embedded images, and auto-generated charts where appropriate.

---

## Features

- Question answering over multi-hundred-page PDF manuals with page-level source citations
- Inline retrieval and rendering of relevant images (screenshots, diagrams) alongside text answers
- Automatic markdown table formatting for structured/comparative data
- On-demand chart generation from numeric data found in retrieved context
- Resumable, checkpointed ingestion that survives API failures without redoing completed work
- OCR fallback for scanned pages

---

## Architecture

```
Upload PDF
    |
    v
[Extraction]  --  PyMuPDF (text + embedded images) + pdfplumber (tables -> markdown)
    |              Scanned-page detection with OCR fallback (Tesseract)
    v
[Chunking]  --  Recursive character splitting, page-bounded, ~1000 chars/chunk
    |
    v
[Text Embedding]  --  Embedding model, stored in pgvector
    |
    v
[Image Captioning]  --  Vision model generates a searchable text description
    |                    per image; the caption is embedded (not the raw pixels)
    v
[PostgreSQL + pgvector]  --  chunks and image captions share one vector space
    |
    v
[Query] --> [Vector similarity search: text + images] --> [LLM]
    |                                                          |
    |                                                generates answer, tags
    |                                                [IMAGE:n] / [CHART:{...}]
    v
[Post-processing]  --  tags replaced with real inline markdown images/charts
    |
    v
[Chat UI]  --  renders markdown, images inline, source citations
```

### Design notes

**Images are made retrievable by captioning, not by embedding pixels directly.** Rather than using a joint image/text embedding model, every extracted image is described in natural language by a vision model at ingestion time. That caption is embedded using the same text embedding model as the document chunks, so both live in one shared vector space. A single similarity search against the query can then surface either a relevant paragraph or a relevant screenshot, ranked on the same scale, with no separate image index to maintain.

**The LLM decides when an image or chart is actually needed**, rather than the system always attaching every retrieved image regardless of relevance. The generation prompt instructs the model to emit an explicit `[IMAGE:n]` or `[CHART:{...}]` tag only when a specific image or numeric visualization would materially help answer the question. A post-processing step converts these tags into real inline markdown, rendered natively by the frontend.

**Ingestion is staged and resumable rather than atomic.** Large source documents can exceed 900 pages and 1,000+ embedded images, which makes a single all-or-nothing ingestion transaction fragile against rate-limited free-tier APIs. Ingestion is split into three checkpointed stages — extract, embed text, caption images — each of which commits progress per item rather than per document. A failure partway through resumes exactly where it stopped, without re-spending API calls on already-completed work.

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| Backend API | FastAPI | Async, typed, auto-generates OpenAPI docs for testing |
| Database | PostgreSQL + pgvector | Native vector similarity search alongside relational metadata |
| ORM | SQLAlchemy 2.0 | Type-safe models, clean migration path |
| PDF extraction | PyMuPDF (fitz) | Fast text + embedded image extraction, page rasterization for OCR fallback |
| Table extraction | pdfplumber | Structure-aware table detection, converted to markdown |
| OCR | Tesseract (via pytesseract) | Fallback for scanned pages |
| Text embeddings | Gemini text-embedding-004 | Free tier, generous token/minute quota |
| Image captioning | Gemini Vision | Multimodal, converts images to searchable text descriptions |
| Answer generation | Groq (Llama 3.3 70B) | Fast inference, separate provider from embeddings to diversify quota risk |
| Charting | Matplotlib | Server-side chart generation from LLM-identified numeric data |
| Frontend | Streamlit | Functional chat UI with native markdown/image rendering |

---

## Engineering Challenges and Solutions

This section documents real issues encountered during development and how they were resolved.

### Environment and dependency issues

- **PATH not recognizing locally installed binaries (Tesseract, psql) on Windows.** Resolved by adding the install directory to system PATH with a full terminal restart, or by hardcoding the binary path directly in code as a PATH-independent fallback.
- **pgvector unavailable on a local Windows PostgreSQL install.** Compiling pgvector from source on Windows requires a full C++ build toolchain. Migrated to a managed PostgreSQL provider with pgvector pre-installed, which removed the local-compilation problem entirely and is a legitimate production pattern, not just a workaround.
- **Malformed database connection string in the environment file** (duplicated key prefix, unescaped characters). Diagnosed by printing the raw parsed connection string via the settings loader rather than visually inspecting the file, which surfaced a duplicated key substring that was invisible on casual inspection.
- **HTTP client SDK incompatibility.** An LLM provider SDK internally passed an argument to `httpx` that a newer `httpx` version had removed, causing a startup crash. Resolved by upgrading the SDK to a version compatible with the current `httpx` release.

### Third-party model availability

A vision model used for image captioning was deprecated mid-project — all requests began returning `404`. This was diagnosed by programmatically listing available models against the live API key rather than relying on documentation, which is unreliable for fast-moving model catalogs. A second model choice also failed, this time due to an access-tier restriction despite appearing in the listing. The system was reconfigured to use a stable, non-preview model, confirmed working via a direct test call before resuming ingestion.

Design takeaway: the vision model name is a single configuration value, not hardcoded throughout the codebase, specifically because model availability on free-tier API access has proven to be a recurring maintenance concern rather than a one-time setup step.

### Rate and quota limits

Real ingestion runs against large documents surfaced quota-exhaustion errors from the vision API, both from per-minute rate limits and from the daily request cap. Three fixes were made in response:

1. A sliding-window rate limiter was added around image captioning calls specifically (not text embeddings, which run on a separate and much larger quota), keeping the system under the vision API's per-minute limit.
2. Retry logic was configured to not retry on quota-exhaustion errors specifically, since a quota-exhausted call is guaranteed to fail identically on every retry — retrying it only wastes time and risks client-side timeouts upstream.
3. The captioning stage now stops the entire batch immediately on a quota error rather than continuing to attempt every remaining item in the queue, preserving already-completed work and failing fast instead of hanging.

### Silent failure masking a real bug

The most consequential bug in the project: the original image captioning method caught all exceptions broadly and returned an empty string on any failure. When the vision model name became invalid, every captioning call failed — but the pipeline interpreted each empty string as "this image is non-informative, skip it," committed successfully, and marked the entire document ready. The failure was invisible until manually cross-checked against the database.

Fixed by narrowing exception handling to distinguish a genuine skip (image too small to be meaningful) from a real API failure, which now propagates, is logged explicitly, and correctly flips the document to a failed status for resumption. Broad exception swallowing that converts failure into an indistinguishable success state is avoided throughout the codebase as a general principle.

### Retrieval crash on partially-processed data

With resumable ingestion, some images legitimately exist in the database without an embedding yet (not yet processed, or explicitly marked non-informative). The retrieval layer's similarity query did not originally account for this, causing a type error whenever such a row was returned. Fixed by explicitly filtering out rows with no embedding in both the chunk and image retrieval queries.

### Resume logic gap

The initial resume implementation checked only the document's status field to decide what to do, but a "failed" status is not itself a stage the pipeline recognizes — it's an end state. Calling resume on a failed document silently did nothing, while the log misleadingly reported success. Fixed by having resume inspect the actual database state (which rows still lack embeddings or captions) to determine the correct stage to resume into, rather than trusting the status field alone.

### Frontend image rendering

Backend answers correctly included image references as relative paths, which resolve correctly when fetched directly from the backend, but silently fail in the frontend since the browser resolves them relative to the frontend's own origin (a different port). Fixed by rewriting relative image URLs to absolute backend URLs before rendering.

---

## File Structure

```
rag-doc-assistant/
├── app/
│   ├── main.py                        FastAPI app entrypoint, lifespan/startup
│   ├── config.py                      Centralized settings
│   │
│   ├── core/
│   │   ├── database.py                SQLAlchemy models, pgvector columns, session
│   │   └── logging_config.py          Structured logging setup
│   │
│   ├── api/
│   │   ├── dependencies.py            Dependency providers
│   │   └── routers/
│   │       ├── documents.py           Upload, list, delete, resume endpoints
│   │       ├── query.py               Query endpoint
│   │       └── health.py              Health check
│   │
│   ├── services/
│   │   ├── document_service.py        Upload validation, resume-stage detection
│   │   └── query_service.py           Retrieval + generation orchestration
│   │
│   ├── ingestion/
│   │   ├── extractors/
│   │   │   ├── base.py                Extractor interface
│   │   │   └── pdf_extractor.py       PyMuPDF + pdfplumber, OCR fallback
│   │   ├── ocr.py                     OCR wrapper
│   │   ├── chunker.py                 Text splitting, page-bounded
│   │   └── pipeline.py                Staged, resumable ingestion orchestrator
│   │
│   ├── embeddings/
│   │   ├── text_embedder.py           Text embedding wrapper
│   │   └── image_captioner.py         Vision captioning wrapper
│   │
│   ├── retrieval/
│   │   ├── retriever.py               Vector similarity search
│   │   └── relevance.py               Image relevance thresholding
│   │
│   ├── generation/
│   │   ├── prompts.py                 System/user prompt templates
│   │   ├── answer_generator.py        LLM call, tag injection, chart injection
│   │   └── chart_generator.py         Chart rendering from LLM-specified data
│   │
│   ├── models/
│   │   └── schemas.py                 Request/response contracts
│   │
│   ├── storage/
│   │   └── image_store.py             Disk persistence for images
│   │
│   └── utils/
│       └── rate_limiter.py            Sliding-window rate limiter
│
├── frontend/
│   └── streamlit_app.py               Chat UI, document upload, inline rendering
│
├── sample_docs/                       Reference PDFs used for testing
├── storage/
│   ├── uploads/                       Transient upload staging
│   └── images/                        Persisted extracted/generated images
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- A PostgreSQL database with the pgvector extension available
- A Gemini API key
- A Groq API key
- Tesseract OCR binary (optional for born-digital PDFs, recommended as a fallback for scanned content)

### Installation

```bash
python -m venv venv
venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` and populate:

```env
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
IMAGE_STORAGE_PATH=./storage/images
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K_TEXT=5
TOP_K_IMAGES=3
```

### Running

Two processes, run concurrently in separate terminals:

```bash
# Terminal 1 — backend
uvicorn app.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

Backend API docs: `http://localhost:8000/docs`
Frontend chat UI: `http://localhost:8501`

---

## Known Limitations

- Table extraction relies on visible ruling lines and may miss tables that rely purely on whitespace alignment.
- Chart generation trusts the LLM's numeric extraction from retrieved text without a schema-constrained extraction step; figures should be spot-checked against source pages.
- Free-tier API quotas mean large-scale ingestion across many documents is a multi-session process rather than a single batch run.