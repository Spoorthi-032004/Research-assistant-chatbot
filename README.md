# AI Research Assistant Backend

A production-ready, CPU-only FastAPI backend for uploading, chatting with, searching,
comparing, and analyzing research papers -- built to minimize LLM API usage through
caching, duplicate detection, and consolidated (single-call) analysis prompts.

A Streamlit frontend is included in streamlit_app.py for interacting with the
backend through a web interface. Every capability is exposed through **Swagger UI** at `/docs`.

---

## 1. Architecture

```
app/
  api/            # FastAPI routers -- request/response wiring only, no business logic
  services/       # Business logic (PDF parsing, embeddings, LLM calls, search, analytics)
  database/       # MongoDB (Motor) and FAISS persistence
  guardrails/     # Regex -> Presidio -> NeMo Guardrails input/output pipelines
  models/         # Pydantic schemas
  utils/          # Hashing, token counting, chunking
  core/           # Settings and logging configuration
  main.py         # FastAPI app + lifespan (Mongo connect/disconnect)
```

**Data separation (per spec):**
- **MongoDB** stores only AI-generated results: `paper_name`, `sha256`, `summary`,
  `structured_information`, `research_gap`, `timestamp`. It never stores embeddings
  or chunk text.
- **FAISS** stores a single global on-disk index (`data/faiss_indexes/global.index`)
  shared across all uploaded papers, plus a JSON sidecar (`data/faiss_indexes/metadata.json`)
  mapping every FAISS vector ID to its `paper_id`, `paper_name`, `page_number` (approximate),
  `chunk_index`, and `chunk_text`. On retrieval, FAISS returns vector IDs + similarity
  scores, and the metadata file is used to recover the original chunk text and paper
  info to build the LLM context. `/chat` filters global search results down to a single
  paper by `paper_id`.

## 2. Multi-paper upload

`/upload` accepts **1 to 3** PDF files (`MAX_UPLOAD_FILES` in `.env`, default 3) in a
single multipart request. Each PDF is processed **fully and independently** -- hash
check, text extraction, chunking, embeddings, FAISS storage, and summary/structured-
profile/research-gap generation -- exactly as a single upload would be.

If 2 or more PDFs are uploaded, an additional relatedness step runs after all papers
are processed:
- Each paper's generated `summary` is embedded (via the same local `all-MiniLM-L6-v2`
  model already loaded for chunk embeddings -- no extra LLM call).
- The average pairwise cosine similarity across all uploaded papers' summary
  embeddings is computed.
- If that average exceeds `PAPER_SIMILARITY_THRESHOLD` (default `0.80`), the papers
  are treated as **related**: one additional LLM call produces a combined summary
  (common themes / key differences / overall findings) on top of each paper's
  individual summary.
- Otherwise the papers are treated as **unrelated**: no combined summary is
  generated, and the response says so explicitly (`relatedness: "unrelated"`,
  `message: "The uploaded papers belong to different topics."`).

The response (`MultiUploadResponse`) always includes each paper's individual
`UploadResponse` in `papers`, plus `relatedness`, `average_similarity`, and
`combined_summary` (null when not applicable).

## 3. LLM usage minimization strategy

1. **Duplicate detection** -- SHA-256 hash of every upload is checked before any
   processing. Duplicate uploads return the cached MongoDB document with **zero**
   LLM calls.
2. **Single consolidated analysis call** -- `/upload` generates the summary,
   structured profile, and initial research gap in **one** Groq call that returns
   structured JSON, instead of three separate calls.
3. **RAG scoping** -- `/chat` only sends the top-k FAISS-retrieved chunks to the
   LLM, never the full paper.
4. **Cached comparisons** -- `/compare` reuses the structured profiles already
   cached in MongoDB; it never re-parses PDFs or re-embeds text.
5. **Bounded analysis prompt** -- the single upload analysis call is built from a
   capped excerpt of the raw extracted text (start + end of the document, see
   `paper_service._build_analysis_context`), not the entire paper, to bound prompt
   size. There is no section-detection step.
6. **Fixed-size RAG chunking** -- the full RAG pipeline is: **PDF -> extracted text
   -> fixed-size chunks (`CHUNK_SIZE_TOKENS`=500, `CHUNK_OVERLAP_TOKENS`=60, no
   section detection) -> embeddings -> FAISS -> top-`TOP_K_RETRIEVAL` (default 5)
   chunks matching the question -> LLM answer**. Only those top-k chunks (not the
   full paper) are ever sent to the LLM for `/chat`. Tune chunk size/overlap/top-k
   via `.env`.
7. **Token/cost analytics** -- every LLM call is logged (`GET /analytics`) so usage
   and spend are fully observable.

## 4. Guardrails pipeline

```
Input:  Regex validation -> Microsoft Presidio -> NeMo Guardrails -> Groq
Output: Groq -> NeMo Guardrails -> Microsoft Presidio -> Regex validation
```

- **Regex stage** blocks API keys, passwords, credit card numbers (Luhn-validated),
  and common prompt-injection phrasing.
- **Presidio stage** detects PII (emails, phone numbers, credit cards, SSNs, IBANs,
  crypto addresses). Input PII is **hard-blocked**; output PII is **masked/redacted**
  so the user still receives a useful answer.
- **NeMo Guardrails stage** applies topical/jailbreak rails (config in
  `app/guardrails/nemo_config/`). It fails **open** (logs a warning, does not break
  the app) if the `nemoguardrails` package or its dependent LLM call is unavailable,
  so the regex + Presidio stages remain the enforced baseline in that case.

## 5. Session-based conversational memory

`/chat` supports ChatGPT-style multi-turn conversations scoped to a single uploaded
paper:

- Omit `session_id` to start a new conversation -- a session is created and its ID is
  returned in the response (`ChatResponse.session_id`). Pass that ID back on
  follow-up questions to continue the same conversation.
- Conversation history is persisted in MongoDB, collection `chat_sessions`:
  ```json
  {
    "session_id": "...",
    "paper_id": "...",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "created_at": "...",
    "updated_at": "..."
  }
  ```
  A session is bound to exactly one `paper_id` at creation; reusing a `session_id`
  with a different `paper_id` returns `400`.
- Every LLM call is assembled as: **System Prompt -> Conversation History -> Retrieved
  FAISS Chunks (current question only) -> Current Question**. History lets the model
  resolve references like "they"/"it" to what was discussed earlier; RAG grounding
  still comes only from the current question's retrieved excerpts (chunks/embeddings
  are always read from FAISS, never from MongoDB).
- History is trimmed to the last `MAX_CHAT_HISTORY_EXCHANGES` user+assistant pairs
  (default 12, i.e. within the 10-15 range) before being sent to the LLM, to control
  token usage as conversations grow. What's stored in Mongo is the guardrail-sanitized
  text (PII already masked), never raw PII.
- Token accounting (`GET /analytics`) naturally includes history tokens, since the
  full assembled prompt (history + context + question) is what's actually sent to Groq.

## 6. Setup

### Prerequisites
- Python 3.11+
- A running MongoDB instance (local or Atlas)
- A [Groq API key](https://console.groq.com)

### MongoDB Setup
The application uses MongoDB to store AI-generated paper results and chat session
history.MongoDB can be run locally or through MongoDB Atlas.
-For a local MongoDB installation, make sure the MongoDB service is running and use:
'''env
MONGO_URI=mongodb://localhost:27017

-For MongoDB Atlas, create a cluster, create a database user, allow the required
network access, and use the Atlas connection string as MONGO_URI in the .env
file.

### Install

```bash
python -m venv .venv
Windows: .venv\Scripts\activate          # macOSor Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # required by Presidio's NLP engine
```

### Configure

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY, MONGO_URI, etc.
```
Create a .env file in the project root and configure the required values:
```env
GROQ_API_KEY=your_groq_api_key
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=research_assistant
ENABLE_NEMO_GUARDRAILS=false
```
For MongoDB Atlas, replace MONGO_URI with the Atlas connection string.


### Run the Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI -- every endpoint
listed below is fully usable from there, including file upload.

### Run the Streamlit Frontend
Keep the FastAPI backend running on port 8000.
Open a new terminal, activate the same virtual environment, and run:
```bash
streamlit run streamlit_app.py
```
The Streamlit frontend will be available at: **http://localhost:8501**

## 7. Environment Variables

The main configuration values are loaded from .env. Important settings include:

- GROQ_API_KEY -- API key used for LLM requests.
- MONGO_URI -- MongoDB connection string.
- MONGO_DB_NAME -- MongoDB database name.
- ENABLE_NEMO_GUARDRAILS -- Enables or disables NeMo Guardrails.
- MAX_UPLOAD_FILES -- Maximum number of PDFs accepted per upload (default: 3).
- PAPER_SIMILARITY_THRESHOLD -- Threshold used to determine whether uploaded papers are related (default: 0.80).
- CHUNK_SIZE_TOKENS -- RAG chunk size (default: 500).
- CHUNK_OVERLAP_TOKENS -- Chunk overlap (default: 60).
- TOP_K_RETRIEVAL -- Number of relevant chunks retrieved for chat (default: 5).
- MAX_CHAT_HISTORY_EXCHANGES -- Maximum conversation history retained for LLM requests (default: 12).

## 8. Notes on optional heavy dependencies

- **PaddleOCR** is included in `requirements.txt` for future image/scanned-PDF
  support as specified, but the current `/upload` pipeline uses PyMuPDF text
  extraction only; OCR is not yet wired into the extraction path.
- **NeMo Guardrails** requires its own LLM-backed self-check calls (configured to
  use Groq in `app/guardrails/nemo_config/config.yml`). If you want to disable it
  (e.g. to save API calls in development), set `ENABLE_NEMO_GUARDRAILS=false` in
  `.env` -- the regex and Presidio stages still run.
- **Presidio** requires a spaCy model (`en_core_web_lg` recommended, `en_core_web_sm`
  also works with lower accuracy). Install per the command above.

## 9. Project conventions

- All business logic lives in `services/`; `api/` routers only validate input,
  call a service, and map exceptions to HTTP responses.
- All I/O (Mongo, FAISS, HTTP calls to Groq/search APIs) is async.
- Every service raises typed exceptions (`PaperProcessingError`, `ChatError`,
  `CompareError`, `GapAnalysisError`, `GuardrailBlockedException`) that the API
  layer translates into appropriate HTTP status codes.
