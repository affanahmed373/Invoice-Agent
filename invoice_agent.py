"""Invoice question-answering agent backed by Qdrant and Gemini."""

from __future__ import annotations

import re

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models as qmodels

from vector_storage import COLLECTION_NAME, EMBEDDING_MODEL_NAME, get_qdrant_client

SYSTEM_MESSAGE = (
    "You are an invoice analysis assistant for multi-vendor invoices. "
    "Use search_qdrant for general invoice questions. Use compare_prices_across_vendors "
    "for price comparisons. Always cite vendor, invoice number, product, and unit price."
)


def _fill_metadata(document: Document) -> Document:
    metadata = dict(document.metadata or {})
    text = document.page_content or ""
    prefix = re.search(r"Vendor:\s*(.+?)\.\s*Invoice:\s*(.+?)\.\s*Date:\s*(.+?)\.\s*File:\s*(.+?)\.", text)
    if prefix:
        metadata.setdefault("vendor_name", prefix.group(1).strip())
        metadata.setdefault("invoice_number", prefix.group(2).strip())
        metadata.setdefault("invoice_date", prefix.group(3).strip())
        metadata.setdefault("source", prefix.group(4).strip())
    if "LINE ITEM" in text:
        metadata.setdefault("chunk_type", "line_item")
        for key in ("product", "unit_price", "amount"):
            match = re.search(rf"{key}:\s*([^|]+)", text)
            if match:
                metadata.setdefault(key, match.group(1).strip())
    document.metadata = metadata
    return document


def _vendor_key(document: Document) -> str:
    vendor = (document.metadata.get("vendor_name") or "").strip()
    source = (document.metadata.get("source") or "").strip()
    return vendor if vendor.lower() not in {"", "unknown", "unknown vendor"} else source or "unknown"


class InvoiceAgent:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = QdrantVectorStore(client=get_qdrant_client(), collection_name=COLLECTION_NAME, embedding=self.embeddings, content_payload_key="text")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        @tool
        def search_qdrant(query: str) -> str:
            """Search invoice text and line items for a question."""
            return self._format(self._retrieve(query))

        @tool
        def compare_prices_across_vendors(product_query: str) -> str:
            """Compare products across vendors using line-item chunks."""
            queries = [part.strip() for part in product_query.split(",") if part.strip()] or [product_query]
            documents = []
            seen = set()
            for query in queries:
                for document in self._retrieve(query, line_items_only=True, fetch_k=40):
                    document = _fill_metadata(document)
                    key = (_vendor_key(document), document.metadata.get("product"), document.page_content)
                    if key not in seen:
                        seen.add(key)
                        documents.append(document)
            return self._format(documents)

        self.graph = create_agent(model=self.llm, tools=[search_qdrant, compare_prices_across_vendors], system_prompt=SYSTEM_MESSAGE)

    def _retrieve(self, query: str, *, line_items_only: bool = False, fetch_k: int = 40) -> list[Document]:
        search_filter = None
        if line_items_only:
            search_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="chunk_type", match=qmodels.MatchValue(value="line_item"))])
        try:
            documents = self.vectorstore.similarity_search(query, k=fetch_k, filter=search_filter)
        except TypeError:
            documents = self.vectorstore.similarity_search(query, k=fetch_k)
        selected = []
        vendor_counts: dict[str, int] = {}
        for document in documents:
            document = _fill_metadata(document)
            vendor = _vendor_key(document)
            if vendor_counts.get(vendor, 0) >= 5:
                continue
            vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
            selected.append(document)
            if len(selected) >= 20:
                break
        return selected

    @staticmethod
    def _format(documents: list[Document]) -> str:
        if not documents:
            return "No relevant documents found in Qdrant."
        results = []
        for index, document in enumerate(documents, 1):
            metadata = document.metadata
            results.append(f"[Result {index} | vendor: {metadata.get('vendor_name', 'unknown')} | invoice: {metadata.get('invoice_number', '?')} | product: {metadata.get('product', '?')} | unit_price: {metadata.get('unit_price', '?')}]\n{document.page_content}")
        return "\n\n---\n\n".join(results)

    def ask(self, question: str) -> str:
        result = self.graph.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content


if __name__ == "__main__":
    print(InvoiceAgent().ask("Which products appear on multiple vendor invoices, and who has the cheaper unit price?"))