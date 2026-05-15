# Plant Operator Q&A System

A local AI assistant that helps industrial plant floor supervisors get instant, accurate answers from the right documentation — automatically routing every question to the correct manual and facility without any manual searching.

---

## The Problem

A floor supervisor at a plant needs to know the H2S alarm threshold. They have 15 manuals covering 5 different facilities across safety, maintenance, and quality control. Finding the right answer means knowing which manual to open, which section to read, and whether the information applies to their specific site. Under time pressure, this is slow and error-prone.

## The Solution

Ask a plain English question. The system figures out which document category and facility applies, pulls the most relevant excerpts, and generates a precise, cited answer — all in seconds.

---

## Features

- **Intelligent routing** — Classifies every question by document type (safety, maintenance, quality control) and facility before searching, so retrieval is always focused
- **Hybrid search** — Combines semantic vector search with BM25 keyword search, then fuses results using Reciprocal Rank Fusion (RRF) for better accuracy on technical terms like part numbers and procedure codes
- **Distance fallback** — If no strong match is found at a specific facility, automatically widens the search across all facilities
- **Streaming answers** — Responses start appearing immediately, word by word, rather than waiting for the full answer to generate
- **Multi-turn chat** — Ask follow-up questions naturally; the system remembers the conversation context
- **Source citations** — Every answer cites exactly which document it came from, with `[Source N]` inline markers
- **PDF page preview** — Expand any source to see the actual rendered page from the original PDF
- **Feedback loop** — Rate each answer with 👍 or 👎; ratings are logged to a local database
- **Analytics dashboard** — Track which questions are asked most, which documents are referenced, and overall satisfaction
- **Prompt caching** — System prompts and document context are cached, reducing latency and API costs on repeated queries
- **Fully local** — All data stays on your machine; only the answer generation calls the Anthropic API

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│    router.py    │  Claude classifies question →  { doc_type, equipment }
│  (Claude API)   │  e.g. { "safety_procedures", "APR" }
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│                retriever.py                 │
│                                             │
│  1. Semantic search  →  ChromaDB cosine     │
│  2. Distance fallback (widens if needed)    │
│  3. BM25 keyword search  →  rank_bm25       │
│  4. RRF fusion  →  merged, re-ranked chunks │
└────────┬────────────────────────────────────┘
         │  top-k chunks with metadata
         ▼
┌─────────────────┐
│     qa.py       │  Claude generates answer with [Source N] citations
│  (Claude API)   │  streamed token by token
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    app.py       │  Streamlit chat UI
│   (Streamlit)   │  Sources, PDF preview, feedback buttons
└─────────────────┘
```

---

## Supported Facilities & Document Types

| Code | Facility |
|------|----------|
| **APR** | Aurora Petrochemical Refinery |
| **BDP** | Brookhaven Dairy Processing |
| **HLX** | Helix Pharmaceuticals Building 4 |
| **NXS** | Nexus Semiconductor Fab 3 |
| **TM7** | Tide Motors Assembly Plant 7 |

Each facility has three document categories:
- **Safety Procedures** — hazards, PPE, permits, emergency response, LOTO, confined space
- **Maintenance Manuals** — PM schedules, inspections, vibration analysis, work orders
- **Quality Control Standards** — product specs, QC tests, hold/release procedures, HACCP

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM (routing + answers) | [Anthropic Claude Sonnet](https://www.anthropic.com/) via `anthropic` SDK |
| Vector database | [ChromaDB](https://www.trychroma.com/) (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` via ChromaDB's default ONNX embedding function |
| Keyword search | [rank-bm25](https://github.com/dorianbrown/rank_bm25) (BM25Okapi) |
| PDF parsing | [pdfplumber](https://github.com/jsvine/pdfplumber) |
| PDF preview rendering | [PyMuPDF](https://pymupdf.readthedocs.io/) (fitz) |
| Feedback storage | SQLite (Python built-in `sqlite3`) |
| Web UI | [Streamlit](https://streamlit.io/) |

---

## Project Structure

```
prototype/
│
├── data/                        # Source PDF documents (15 files)
│   ├── APR_safety_procedures.pdf
│   ├── APR_maintenance_manual.pdf
│   ├── APR_quality_control.pdf
│   └── ...                      # Pattern: {FACILITY}_{doc_type}.pdf
│
├── chroma_db/                   # Persisted vector store (auto-created by ingest.py)
│
├── app.py                       # Streamlit web UI — entry point
├── router.py                    # Claude-based question classifier
├── retriever.py                 # Hybrid BM25 + semantic search with RRF
├── qa.py                        # Streaming answer generation with citations
├── feedback.py                  # SQLite feedback logging and analytics
├── pdf_preview.py               # PDF page rendering via PyMuPDF
├── ingest.py                    # One-time PDF → ChromaDB ingestion pipeline
│
├── requirements.txt
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/plant-operator-qa.git
cd plant-operator-qa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> The `.env` file is listed in `.gitignore` and will never be committed.

### 4. Ingest the documents (optional — auto-runs on first launch)

This step parses all PDFs, splits them into chunks, generates embeddings, and stores everything in a local ChromaDB database. The app **auto-runs ingestion the first time you launch it** if `chroma_db/` doesn't exist, so you can usually skip this step. Run it manually only when you want to pre-build the index, or after changing PDFs:

```bash
python ingest.py
```

Expected output:
```
Ingesting 15 PDFs into 'plant_docs'...
  APR_maintenance_manual.pdf: 8 chunks
  APR_quality_control.pdf: 6 chunks
  APR_safety_procedures.pdf: 7 chunks
  ...
Done. 15 PDFs, 107 total chunks.
ChromaDB persisted to ./chroma_db/
```

### 5. Launch the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## How to Use

### Asking questions

Type any operational question in the chat input at the bottom of the screen. Be as specific or as general as you like:

| Example question | What happens |
|---|---|
| `What are the H2S alarm thresholds at APR?` | Routes to safety / APR |
| `How often should vibration checks be done on rotating equipment?` | Routes to maintenance / APR |
| `What is the phosphatase test procedure after pasteurization?` | Routes to quality control / BDP |
| `What PPE is required for confined space entry at the semiconductor fab?` | Routes to safety / NXS |
| `What are the PM schedules across all our facilities?` | Routes to maintenance / all facilities |

### Reading the answer

Each answer includes:
- **Inline citations** — `[Source 1]`, `[Source 2]` pointing to specific document chunks
- **Routing decision** (expandable) — shows which doc type and facility was selected
- **Sources** (expandable) — lists the source files, with a "View page N" option to see the actual PDF page
- **Retrieved chunks** (expandable, debug) — shows the raw text excerpts that were used

### Follow-up questions

You can ask follow-up questions naturally without repeating context:

```
You:       What are the H2S alarm thresholds at APR?
Assistant: The IDLH level is 50 ppm...

You:       What should I do if that alarm activates?
Assistant: Based on the APR safety procedures... [knows you mean APR H2S]
```

### Giving feedback

After each answer, click 👍 or 👎. Ratings are stored locally in `feedback.db` and displayed in the **Analytics** tab.

### Clearing the conversation

Click **Clear conversation** in the sidebar to start fresh.

---

## How It Works — Under the Hood

### Ingestion (`ingest.py`)

Each PDF filename encodes its metadata: `APR_safety_procedures.pdf` maps to facility=`APR`, type=`safety_procedures`. The pipeline:

1. Extracts text page by page with `pdfplumber`
2. Splits into 600-character chunks with 100-character overlap (to preserve context at chunk boundaries)
3. Stores each chunk in ChromaDB with metadata: `{equipment, doc_type, source, chunk_index, page_num}`
4. Uses idempotent `upsert` — safe to re-run without creating duplicates

### Routing (`router.py`)

A fast Claude call (`max_tokens=100`) classifies the question into one of three document types and one of five facility codes (or "all"). The system prompt is cached by the Anthropic API, so repeated calls skip re-processing those tokens. If Claude returns malformed output, the system falls back to `safety_procedures / all` rather than erroring.

### Retrieval (`retriever.py`)

Three steps run for every query:

**Step 1 — Semantic search**: ChromaDB embeds the question and finds the top-k most similar chunks by cosine distance, filtered to only the relevant doc type and facility.

**Step 2 — Distance fallback**: If the best chunk found has a cosine distance above 0.6 (meaning it's still fairly dissimilar), the search widens to all facilities. This handles questions that don't explicitly name a site.

**Step 3 — BM25 + RRF fusion**: A BM25 keyword search runs on the same filtered subset of chunks. Results from semantic and keyword search are merged using Reciprocal Rank Fusion:

```
score(chunk) = 1 / (60 + semantic_rank) + 1 / (60 + bm25_rank)
```

Chunks that rank highly in both searches score the highest. This hybrid approach is particularly valuable for exact technical terms (part numbers, chemical names, regulation codes) that semantic search alone might miss.

### Answer generation (`qa.py`)

The top-k chunks are formatted as numbered source blocks and sent to Claude alongside the question. The system prompt is cached with `cache_control: ephemeral`. The document context block is also marked for caching — when the same documents are retrieved for a follow-up question, that portion is served from cache rather than re-processed.

Claude is instructed to answer using only the provided excerpts and to cite `[Source N]` inline. If the answer isn't in the excerpts, it says so rather than guessing.

Streaming (`client.messages.stream()`) pushes each token to the Streamlit UI as it's generated, so the answer begins appearing within half a second.

### Multi-turn conversation (`app.py`)

Prior conversation turns are stored in `st.session_state.messages` and passed as message history to Claude on each new question. The routing step always uses only the current question (routing doesn't benefit from history), while the answer generation step receives the full prior context. On Streamlit reruns triggered by button clicks (e.g., feedback), past messages are replayed from session state without calling Claude again.

---

## Adding New Documents

The system accepts any PDF following the naming convention `{FACILITY}_{doc_type}.pdf`.

**Supported facility codes:** `APR`, `BDP`, `HLX`, `NXS`, `TM7`

**Supported doc types:** `safety_procedures`, `maintenance_manual`, `quality_control`

To add a new document:

1. Place the PDF in the `data/` folder following the naming convention
2. Re-run `python ingest.py` — existing chunks are updated via upsert, new chunks are added

To add a **new facility or document type**, update the `EQUIPMENT_CODES` / `DOC_TYPES` sets in `ingest.py`, the routing system prompt in `router.py`, and the facility legend in the `app.py` sidebar.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key from [console.anthropic.com](https://console.anthropic.com/) |

---

## License

MIT
