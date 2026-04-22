# Local Business Review Summarizer

A RAG-based chatbot that fetches real customer reviews from the Yelp dataset and delivers structured pros and cons summaries for local businesses. Built as an end-to-end AI project using semantic search, LLM generation, and automated data pipelines.

---

## What it does

- Takes a natural language query about a local business or location
- Retrieves the most relevant review chunks from a vector database using hybrid search
- Generates a structured summary with pros, cons, and a verdict using an LLM
- Supports follow-up questions with multi-turn conversational memory
- Automatically refreshes data weekly via n8n automation

---

## Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| Dataset | Yelp Open Dataset | 6.9M real reviews, free |
| Vector DB | Qdrant | Self-hostable, fast, great Python client |
| Embeddings | all-MiniLM-L6-v2 | Free, local, 384-dim, fast |
| LLM | LLaMA 3.3 70B via Groq | Fastest inference, free tier, high availability |
| Search | Hybrid (Semantic + BM25) | Better retrieval than semantic alone |
| UI | Streamlit | Simple, Python-native, no frontend needed |
| Automation | n8n | Visual workflow automation for scheduled ingestion |
| Containerization | Docker | Runs Qdrant and n8n consistently |

---

## Prompting Framework

This project uses a **RISEN prompting framework** combined with RAG:

| Component | Description |
|-----------|-------------|
| **Role** | Defines the assistant as a local business review analyst |
| **Instructions** | Rules — only use provided context, never hallucinate |
| **Steps** | Structured output format — pros, cons, verdict |
| **End goal** | Help users evaluate local businesses through real reviews |
| **Narrowing** | Restrict model strictly to retrieved review context |

On top of RISEN, two additional techniques are layered:

- **RAG (Retrieval Augmented Generation)** — real review chunks from Qdrant are injected into every prompt so the model responds based on actual customer data rather than its own training knowledge
- **Multi-turn conversational memory** — last 6 conversation turns are passed with every request so follow-up questions like "what about their service?" work naturally

---

## Architecture

```
Yelp Dataset (JSON)
        |
        v
ingest.py — clean, chunk, embed, store in Qdrant
        |
        v
    Qdrant DB (persisted in qdrant-data/)
        |
        v
User query in Streamlit UI
        |
        v
search.py — hybrid search (semantic + BM25) — top 8 chunks
        |
        v
llm.py — RISEN prompt + RAG context — Groq LLaMA 3.3 70B
        |
        v
Structured pros/cons summary displayed in UI

Background automation (n8n):
n8n scheduler — trigger.py — ingest.py (every Monday 9am)
```

---

## Folder Structure

```
local-business-review-summarizer/
├── docker-compose.yml        # Qdrant + n8n services
├── .env                      # API keys (never commit)
├── .env.example              # Template for required keys
├── .gitignore
├── README.md
├── start.sh                  # One command startup script
├── app/
│   ├── ingest.py             # Load, chunk, embed, store reviews
│   ├── search.py             # Hybrid semantic + BM25 search
│   ├── llm.py                # RISEN prompt + Groq LLM call
│   ├── ui.py                 # Streamlit chatbot interface
│   ├── trigger.py            # Flask endpoint for n8n automation
│   └── requirements.txt
├── n8n-workflows/
│   └── pipeline.json         # Exported n8n automation workflow
└── logs/
    └── ingest.log            # Ingestion run history
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker Engine
- WSL2 (if on Windows)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/local-business-review-summarizer.git
cd local-business-review-summarizer
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Fill in your API keys:

```env
GROQ_API_KEY=your_groq_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
BUSINESS_FILE=data/yelp_academic_dataset_business.json
REVIEW_FILE=data/yelp_academic_dataset_review.json
MAX_BUSINESSES=500
MAX_REVIEWS_PER_BUSINESS=10
```

### 4. Add Yelp dataset

Download from [kaggle.com/datasets/yelp-dataset/yelp-dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) and place both files in `data/`:

```
data/yelp_academic_dataset_business.json
data/yelp_academic_dataset_review.json
```

### 5. Start Docker services

```bash
docker compose up -d
```

### 6. Run ingestion (first time only)

```bash
python3 app/ingest.py
```

### 7. Start the app

```bash
./start.sh
```

Open [http://localhost:8501](http://localhost:8501)

## Evaluation Results

| Metric | Score |
|--------|-------|
| Avg Precision@5 | 0.92 |
| Avg MRR | 0.90 |
| Avg Faithfulness | 0.68 |

Run evaluation: `python3 app/evaluate.py`

---

## Usage

**Specific business query:**
```
Pros and cons of Papa Murphy's in Tucson
```

**Location-based query:**
```
Best restaurants in Nashville
```

**Follow-up questions:**
```
What do people say about the service?
Is it good for families?
What are the most common complaints?
```

Toggle **"Show retrieved reviews"** in the sidebar to see exactly which review chunks were fetched from Qdrant — this makes the RAG pipeline visible in real time.

---

## File Responsibilities

| File | Runs when | Does what |
|------|-----------|-----------|
| `ingest.py` | Once on setup, then weekly via n8n | Loads Yelp data, chunks reviews, embeds and stores in Qdrant |
| `search.py` | Every user query | Converts query to vector, runs hybrid search, returns top chunks |
| `llm.py` | Every user query | Builds RISEN prompt with retrieved chunks, calls Groq, returns summary |
| `ui.py` | Always (main app) | Streamlit interface, calls search.py and llm.py |
| `trigger.py` | Called by n8n | Flask endpoint that runs ingest.py in background |

---

## n8n Automation

The ingestion pipeline is automated using n8n:

1. **Schedule Trigger** — fires every Monday at 9am
2. **HTTP Request** — calls `trigger.py` at `/ingest`
3. `trigger.py` runs `ingest.py` in a background thread and responds immediately
4. Ingestion output is written to `logs/ingest.log`

To import the workflow: open n8n at [http://localhost:5678](http://localhost:5678), go to Workflows, click the menu, and import `n8n-workflows/pipeline.json`.

---

## Search Strategy

Hybrid search combining two approaches:

- **Semantic search (70% weight)** — vector similarity via Qdrant and MiniLM embeddings. Finds conceptually related reviews even without exact keyword matches
- **BM25 keyword search (30% weight)** — term frequency reranking via rank-bm25. Boosts results that contain the exact words from the query
- **City filtering** — detects city names in the query and pre-filters Qdrant results to that location before reranking

---

## Chunking Strategy

Reviews are chunked using `RecursiveCharacterTextSplitter`:

- Chunk size: 256 tokens
- Chunk overlap: 32 tokens
- Each chunk stored with metadata: business name, city, state, rating, date
- Reviews under 5 words are filtered out before chunking

---

## API Keys Required

| Service | Where to get | Cost |
|---------|-------------|------|
| Groq | [console.groq.com](https://console.groq.com) | Free |
| Yelp dataset | [kaggle.com](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) | Free download |

---

## Acknowledgements

- Yelp for the open dataset
- Groq for fast LLM inference
- Qdrant for the vector database
- n8n for workflow automation
- Hugging Face for the sentence-transformers library
