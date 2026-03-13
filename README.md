# FinSight

A RAG-powered assistant for financial document analysis. Upload earnings reports, 10-Ks, analyst notes, or any internal financial document and ask questions in plain language.

Built this after spending a lot of time watching analysts copy-paste between PDFs and spreadsheets. The goal was simple: drop in a document and start asking questions without any manual prep.

---

## What it does

- Ingests PDF and text-based financial documents
- Chunks and embeds them using OpenAI's `text-embedding-3-small`
- Stores everything in an in-memory ChromaDB vector store
- Runs a ReAct-style agent (tool-calling loop) that retrieves context before answering
- Shows source attribution for every answer so you can verify what it pulled

The agent is intentionally designed to retrieve before it responds — it won't hallucinate an answer when the context isn't there. It'll just tell you it doesn't have enough information.

---

## Architecture

```
User Query
    │
    ▼
FinancialAgent (ReAct loop)
    │
    ├─── search_documents() ──► VectorStore (ChromaDB)
    │         │                      │
    │         │◄─── top-k chunks ────┘
    │
    ├─── [multi-step if needed]
    │
    └─── Final Answer + Sources
```

**Ingestion pipeline:**
```
PDF / TXT ──► DocumentProcessor ──► text cleaning ──► chunking ──► VectorStore.add()
```

---

## Stack

| Component | Tool |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB (in-memory) |
| Agent | OpenAI function-calling loop |
| UI | Streamlit |
| PDF parsing | pypdf |

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/finsight.git
cd finsight
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your environment**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

**5. Run the app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. Paste your OpenAI API key in the sidebar (it's never stored anywhere)
2. Upload one or more financial documents (PDF or TXT)
3. Hit **Ingest Documents** — this chunks and embeds everything
4. Start asking questions

**Some things worth trying:**
- "What was the revenue growth compared to last year?"
- "Summarize the main risk factors"
- "What guidance did management give for next quarter?"
- "What are the gross and operating margins?"

If you upload multiple documents, you can ask cross-document questions and it'll pull from both.

---

## Project structure

```
finsight/
├── app.py                  # Streamlit UI
├── src/
│   ├── agent.py            # ReAct agent with tool-calling
│   ├── vector_store.py     # ChromaDB wrapper
│   ├── ingestion.py        # Document loading and chunking
│   └── config.py           # Settings and env vars
├── requirements.txt
└── .env.example
```

---

## Configuration

You can tweak the following in `.env` or through the UI:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | Model used for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `TOP_K_RETRIEVAL` | `6` | Chunks retrieved per query |

Chunk size and overlap are adjustable in the sidebar.

---

## Known limitations

- The vector store is in-memory, so documents are gone when you close the session. Persistence with ChromaDB's disk backend is straightforward to add if needed.
- Very large PDFs (100+ pages) may be slow to ingest depending on your machine.
- Table-heavy PDFs sometimes don't parse well — the text extracted from complex tables can be messy. Works best with text-heavy reports.
- No streaming yet. Answers appear all at once after the agent finishes reasoning.

---

## Possible extensions

A few things that would make this more production-ready:

- **Persistent storage** — swap the in-memory ChromaDB client for a persistent one or a managed vector DB (Pinecone, Qdrant, Weaviate)
- **Hybrid search** — combine dense retrieval with BM25 for better recall on financial terminology
- **Structured extraction** — add a pipeline that extracts key metrics (revenue, EBITDA, margins) into structured form and stores them separately
- **Multi-tenant support** — namespace collections per user or document set
- **Evaluation** — add a RAGAS-based eval pipeline to track answer faithfulness and relevance over time

---

## Requirements

- Python 3.10+
- OpenAI API key
- ~500MB disk space for dependencies

---

## License

MIT
