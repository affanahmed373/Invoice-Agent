# Invoice Agent

AI-powered invoice analysis agent that extracts text from PDF invoices, stores embeddings in a vector database, and answers natural-language questions about the invoices.

## Tech Stack

- **LangChain** — Agent orchestration & text splitting
- **Google Gemini** (`gemini-2.5-flash`) — LLM backbone
- **Qdrant** — Vector database for semantic search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — Embedding model
- **pdfplumber** — PDF text + table extraction

## Pipeline

1. **PDF extraction** — Extract page text and line-item tables with pdfplumber
2. **Chunking** — Store both page text chunks and structured `line_item` chunks
3. **Embedding** — Generate 384-dim embeddings via `all-MiniLM-L6-v2`
4. **Vector storage** — Store embeddings in Qdrant for similarity search
5. **Agent** — LangChain agent backed by Gemini queries the vector store to answer questions

## Setup

The notebooks are still available as experiments, but the maintained implementation is now in `vector_storage.py`, `invoice_agent.py`, and `app.py`. Install dependencies with:

```bash
uv sync
```

Set these environment variables locally (or as Space secrets on Hugging Face):

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI / Gemini API key |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_HOST` | Qdrant cluster URL |

## Usage

Ingest all PDFs in the current directory into Qdrant:

```bash
uv run python vector_storage.py
```

Ask questions from the command line:

```bash
uv run python invoice_agent.py
```

Run the Gradio app:

```bash
uv run python app.py
```

## Hugging Face Spaces

Yes. This project can run as a **Gradio Space**. Create a Space with the Gradio SDK, upload `app.py`, `invoice_agent.py`, `vector_storage.py`, `pyproject.toml`, `uv.lock`, and any indexed data/configuration you need, then add `GOOGLE_API_KEY`, `QDRANT_API_KEY`, and `QDRANT_HOST` as Space secrets. The Space connects to the existing Qdrant collection; it does not expose the API keys to users.

For a public demo, avoid committing invoice PDFs containing personal or commercial information. The current UI queries an already-indexed Qdrant collection. PDF upload and per-user ingestion would require an additional upload flow and storage policy.

## Project Layout

```
Invoice_Agent/
├── app.py                  # Hugging Face Spaces / Gradio entrypoint
├── invoice_agent.py        # Retrieval tools and Gemini agent
├── vector_storage.py       # PDF extraction, chunking, embeddings, and upsert
├── invoice_agent.ipynb     # Original exploratory notebook
├── vector_storage.ipynb    # Original exploratory notebook
├── *.pdf                   # Sample invoice documents
```

The original notebooks remain useful for Colab exploration, but they are no longer required to run the project.

Drop any vendor's PDF into the folder (prefer filenames without spaces). Ingestion auto-discovers `*.pdf` files and extracts vendor-agnostic metadata on every chunk: `vendor_name`, `invoice_number`, `invoice_date`, and `source`.

Each invoice produces:
- `page_text` chunks for general Q&A
- `line_item` chunks (from tables when available, otherwise text heuristics) for product/price comparison

Vendor names come from legal-entity patterns, supplier labels, or email/website domains—not from a hardcoded vendor list. If a PDF has no usable vendor text, set an optional entry in `VENDOR_OVERRIDES`.

Example questions:

```
"What is the total amount on invoice RE-202659666?"
"Which vendor has the cheaper price for PG Tips tea bags?"
"Compare atta prices across vendors."
```
