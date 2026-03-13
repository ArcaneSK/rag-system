# Local PDF RAG System

This project is a fully working retrieval-augmented generation system that:

- ingests PDFs from a local folder
- parses PDFs with PyMuPDF and OCR fallback for image-heavy pages
- chunks and embeds them with OpenAI embeddings
- uses the official OpenAI Python SDK
- stores vectors in local ChromaDB
- uses hybrid retrieval (vector + keyword BM25-style scoring + reranking)
- answers questions in a Gradio chat interface with grounded source citations

## What you get

- `data/pdfs/`: drop your PDFs here
- `data/chroma/`: persistent Chroma vector store
- `data/state/`: manifest and chunk catalog for incremental indexing
- `data/models/`: local reranker model cache
- `uv run rag-ingest`: CLI indexing entrypoint
- `uv run rag-reset`: CLI reset entrypoint
- `app.py`: Gradio chat UI

## Setup

1. Create the local virtual environment with `uv`:

```bash
uv venv --python 3.13
```

2. Install and lock dependencies:

```bash
uv sync
```

3. Copy the env template and set your API key:

```bash
cp .env.example .env
```

4. Add PDF files into:

```bash
data/pdfs/
```

## Run indexing

Incremental indexing only embeds new or changed PDFs.

```bash
uv run rag-ingest
```

## Reset the local DB

This clears the local Chroma store and indexing state, but keeps your PDFs.

```bash
uv run rag-reset
```

For non-interactive use:

```bash
uv run rag-reset --yes
```

## Run the chat app

```bash
uv run rag-ui
```

The Gradio UI will open a local chat interface where you can:

- refresh the PDF index
- inspect indexing status
- ask grounded questions against the indexed PDFs
- see which PDF pages were used to answer

## Cost and accuracy choices

- Default embedding model: `text-embedding-3-small`
  - chosen for strong price/performance
- Default chat model: `gpt-4o-mini`
  - chosen for low cost and solid grounded answer quality
- Default reranker:
  - `jinaai/jina-reranker-v1-turbo-en` via FastEmbed cross-encoder
  - improves precision after initial retrieval without another OpenAI call
- PDF parsing strategy:
  - PyMuPDF sorted block extraction first
  - OCR fallback for pages with little native text or heavy image coverage
- Retrieval strategy:
  - OpenAI embedding similarity in ChromaDB
  - local BM25-style keyword retrieval
  - reciprocal rank fusion
  - neighbor expansion for adjacent chunk context
  - local cross-encoder reranking on the candidate set

This keeps retrieval accurate while remaining cost-efficient: OpenAI is used for embeddings and generation, while OCR fallback and reranking run locally.

## Optional tuning

Relevant `.env` settings:

```bash
MODEL_CACHE_DIR=data/models
RERANK_ENABLED=true
RERANK_CANDIDATES=12
RERANK_MODEL_NAME=jinaai/jina-reranker-v1-turbo-en
RERANK_MAX_CHARS=1800
OCR_ENABLED=true
OCR_MIN_TEXT_CHARS=120
OCR_IMAGE_AREA_THRESHOLD=0.55
OCR_RENDER_DPI=144
```

## Project layout

```text
rag_system/
  config.py
  ocr.py
  pdf_ingest.py
  retrieval.py
  rerank.py
  service.py
  store.py
app.py
scripts/ingest.py
data/
  pdfs/
  chroma/
  state/
  models/
pyproject.toml
```

## Notes

- The reranker model may download locally on first use and then be cached under `data/models/`.
- RapidOCR uses packaged OCR model assets from the installed Python environment.
- Chroma data persists locally in `data/chroma/`.
- If you change chunking or model settings significantly, run a forced rebuild.
- If you change OCR settings or the PDF parser strategy, run `uv run rag-ingest --force`.
- To force a rebuild with `uv`, run `uv run rag-ingest --force`.
