# Invoice Agent

AI-powered invoice analysis agent that extracts text from PDF invoices, stores embeddings in a vector database, and answers natural-language questions about the invoices.

## Tech Stack

- **LangChain** — Agent orchestration & text splitting
- **Google Gemini** (`gemini-2.5-flash`) — LLM backbone
- **Qdrant** — Vector database for semantic search
- **sentence-transformers** (`all-MiniLM-L6-v2`) — Embedding model
- **pypdf** — PDF text extraction

## Pipeline

1. **PDF extraction** — Extract text from invoice PDFs using pypdf
2. **Chunking** — Split text into manageable chunks with LangChain text splitters
3. **Embedding** — Generate 384-dim embeddings via `all-MiniLM-L6-v2`
4. **Vector storage** — Store embeddings in Qdrant for similarity search
5. **Agent** — LangChain agent backed by Gemini queries the vector store to answer questions

## Setup

The notebooks are designed to run on **Google Colab**. You will need the following secrets configured in Colab's `userdata`:

| Secret | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI / Gemini API key |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `QDRANT_HOST` | Qdrant cluster URL |

## Project Layout

```
Invoice_Agent/
├── invoice_agent.ipynb     # Main notebook: full extraction → agent pipeline
├── vector_storage.ipynb    # Vector storage experiments
├── *.pdf                   # Sample invoice documents
```

## Usage

Open `invoice_agent.ipynb` in Google Colab, configure your API keys, and run all cells. The agent can then answer questions like:

```
"What is the total amount on invoice RE-202659666?"
"List all line items from the sales invoice."
```
