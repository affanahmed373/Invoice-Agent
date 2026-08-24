"""Extract invoice PDFs and store searchable chunks in Qdrant."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pdf_document_embeddings")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

LEGAL_SUFFIX = r"(?:GmbH|S\.?R\.?L\.?|Ltd\.?|Limited|Inc\.?|A\.?G\.?|B\.?V\.?|SAS|plc|LLC|LLP|OHG|e\.?U\.?|Group)"
COMMON_EMAIL_ROOTS = {"gmail", "yahoo", "hotmail", "outlook", "icloud", "gmx", "web", "aol", "mail", "protonmail", "live", "msn", "example", "google"}
BUYER_OR_NOISE = re.compile(r"zaki|asian foods|billing address|bill[- ]?to|lieferadresse|kunde|customer|debitor|universit|lagerort|warehouse|^(pos|qty|amount|total|page|invoice|rechnung|facture|date|datum|tel|phone|email|iban|vat|mwst|brutto|netto|summe|betrag)\b", re.I)
PRICE_HEADER_HINTS = re.compile(r"preis|price|amount|betrag|e\.-?\s*preis|g\.-?\s*preis|unit", re.I)
DESC_HEADER_HINTS = re.compile(r"bezeichnung|description|artikel|item|product|marque|marke|designation", re.I)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .:;-|,")


def _looks_like_company(name: str) -> bool:
    return bool(name and 3 <= len(name) <= 80 and not BUYER_OR_NOISE.search(name) and not re.search(r"\d{5}|strasse|straße|avenue|street", name, re.I))


def _domain_to_vendor(domain: str) -> str | None:
    tlds = {"com", "net", "org", "be", "de", "fr", "uk", "co", "io", "eu", "info", "biz"}
    parts = [part for part in domain.lower().split(".") if part not in tlds]
    if not parts or parts[0] in COMMON_EMAIL_ROOTS:
        return None
    return " ".join(part.upper() if len(part) <= 3 else part.title() for part in parts)


def extract_vendor_name(full_text: str, pdf_name: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and pdf_name in overrides:
        return overrides[pdf_name]
    for chunk in (full_text[:2000], full_text):
        for match in re.finditer(rf"([A-Z][A-Za-z0-9&.,\-][A-Za-z0-9&.,\- ]{{1,50}}\s+{LEGAL_SUFFIX})\b", chunk):
            parts = _clean(match.group(1)).split()
            for start in range(len(parts)):
                candidate = " ".join(parts[start:])
                if re.search(LEGAL_SUFFIX + r"$", candidate, re.I) and _looks_like_company(candidate):
                    return candidate
    labels = r"Supplier|Vendor|Seller|Sold by|From|Lieferant|Verkäufer|Facturé par|Issuer|Geliefert durch"
    for match in re.finditer(rf"(?:{labels})\s*[:\-]?\s*\n?\s*([^\n]{{3,80}})", full_text, re.I):
        candidate = _clean(match.group(1))
        if _looks_like_company(candidate):
            return candidate
    for domain in re.findall(r"[\w.+\-]+@([\w\-]+(?:\.[\w\-]+)+)", full_text):
        if vendor := _domain_to_vendor(domain):
            return vendor
    for host in re.findall(r"https?://(?:www\.)?([\w.\-]+)", full_text, re.I):
        if vendor := _domain_to_vendor(host):
            return vendor
    return "Unknown Vendor"


def extract_invoice_number(full_text: str, pdf_name: str) -> str:
    patterns = (r"\b(RE-\d+)\b", r"Invoice\s*No\.?\s*:?\s*\n?\s*([A-Z]{0,5}\d[\w\-]*)", r"Invoive\s*No\.?\s*:?\s*\n?\s*([A-Z]{0,5}\d[\w\-]*)", r"Rechnungsnummer\s*:?\s*\n?\s*([A-Z]{0,5}\d[\w\-]*)", r"Facture\s*N[o°º.]*\s*:?\s*\n?\s*([A-Z]{0,5}\d[\w\-]*)", r"Rechn\.?\s*Nr\.?\s*:?\s*(?:\n[^\n]*){0,8}?\n\s*(\d{6,})", r"Rechn\.?\s*Nr\.?\s*:?\s*([A-Z]{0,5}\d[\w\-]*)")
    for pattern in patterns:
        if match := re.search(pattern, full_text, re.I):
            return match.group(1)
    if match := re.search(r"(RE-\d+|\d{6,})", Path(pdf_name).stem, re.I):
        return match.group(1)
    return "Unknown"


def extract_invoice_date(full_text: str) -> str:
    patterns = (r"(?:Document Date|Invoice Date|Rechnungsdatum|Lieferdatum)\s*:?\s*\n?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})", r"Datum\s*:?\s*(?:\n[^\n]*){0,8}?\n\s*(\d{2}\.\d{2}\.\d{4})", r"Datum\s*:?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})", r"\b(\d{2}\.\d{2}\.\d{4})\b", r"\b(\d{2}-\d{2}-\d{4})\b")
    for pattern in patterns:
        if match := re.search(pattern, full_text, re.I):
            return match.group(1)
    return "Unknown"


def extract_invoice_metadata(full_text: str, pdf_name: str, overrides: dict[str, str] | None = None) -> dict[str, str]:
    return {"source": pdf_name, "vendor_name": extract_vendor_name(full_text, pdf_name, overrides), "invoice_number": extract_invoice_number(full_text, pdf_name), "invoice_date": extract_invoice_date(full_text)}


def build_context_prefix(metadata: dict[str, str]) -> str:
    return f"Vendor: {metadata['vendor_name']}. Invoice: {metadata['invoice_number']}. Date: {metadata['invoice_date']}. File: {metadata['source']}.\n"


def _cell(value: object) -> str:
    return "" if value is None else _clean(str(value).replace("\n", " "))


def is_line_item_table(header_row: list[object] | None) -> bool:
    if not header_row or len(header_row) < 3:
        return False
    joined = " ".join(_cell(cell) for cell in header_row)
    return bool(PRICE_HEADER_HINTS.search(joined) and (DESC_HEADER_HINTS.search(joined) or re.search(r"\bpos\b", joined, re.I)))


def format_line_item_text(fields: dict[str, str]) -> str:
    parts = ["LINE ITEM"]
    for key in ("product", "brand", "description", "qty", "uom", "unit_price", "amount", "article_no", "raw"):
        if fields.get(key):
            parts.append(f"{key}: {fields[key]}")
    return " | ".join(parts)


def line_items_from_table(table: list[list[object]], page_number: int) -> list[dict[str, object]]:
    if not table or len(table) < 2 or not is_line_item_table(table[0]):
        return []
    headers = [_cell(header).lower() for header in table[0]]

    def find_col(*hints: str) -> int | None:
        return next((index for index, header in enumerate(headers) if any(hint in header for hint in hints)), None)

    indexes = {"brand": find_col("marke", "brand", "marque"), "description": find_col("bezeichnung", "description", "artikel", "item", "product"), "qty": find_col("menge", "qty", "quantity", "anzahl"), "unit_price": find_col("e.-preis", "e-preis", "unit", "einzel"), "amount": find_col("g.-preis", "g-preis", "amount", "betrag", "total"), "article_no": find_col("art.-nr", "art-nr", "sku", "code")}
    items = []
    for row_index, row in enumerate(table[1:], start=1):
        cells = [_cell(cell) for cell in row]
        if not any(cells):
            continue
        get = lambda key: cells[indexes[key]] if indexes[key] is not None and indexes[key] < len(cells) else ""
        product = " ".join(part for part in (get("brand"), get("description")) if part).strip() or " ".join(cells)
        if not re.search(r"[A-Za-z]{3,}", product):
            continue
        fields = {**{key: get(key) for key in indexes}, "product": product, "raw": " | ".join(cell for cell in cells if cell)}
        items.append({"text": format_line_item_text(fields), "page_number": page_number, "item_index": row_index, "product": product, "unit_price": fields["unit_price"], "amount": fields["amount"]})
    return items


def line_items_from_text(page_text: str, page_number: int) -> list[dict[str, object]]:
    items = []
    patterns = (re.compile(r"^(\d+)\s+(\d+)\s+(\d+)\s+(.+?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s+\d+\s*$"), re.compile(r"^([A-Z]?\d+[A-Z0-9]*)\s+(.+?)\s+(\d+)\s+(BAG|BOX|TRAY|PCS|CTN|KG)\s+(-?\d+[.,]\d{2})\s+(-?\d+[.,]\d{2})\b", re.I))
    for line in (_clean(value) for value in page_text.splitlines()):
        match = patterns[0].match(line) or patterns[1].match(line)
        if not match:
            continue
        groups = match.groups()
        fields = {"article_no": groups[1] if len(groups) == 7 else groups[0], "qty": groups[2], "product": groups[3] if len(groups) == 7 else groups[1], "unit_price": groups[4], "amount": groups[6] if len(groups) == 7 else groups[5], "raw": line}
        items.append({"text": format_line_item_text(fields), "page_number": page_number, "item_index": len(items) + 1, "product": fields["product"], "unit_price": fields["unit_price"], "amount": fields["amount"]})
    return items


def extract_documents(pdf_paths: list[str | Path], vendor_overrides: dict[str, str] | None = None) -> tuple[list[Document], list[dict[str, str]]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True)
    documents: list[Document] = []
    summaries = []
    for pdf_path in pdf_paths:
        pdf_name = Path(pdf_path).name
        with pdfplumber.open(pdf_path) as pdf:
            pages = [(page.extract_text() or "", page.extract_tables() or []) for page in pdf.pages]
        metadata = extract_invoice_metadata("\n".join(text for text, _ in pages), pdf_name, vendor_overrides)
        summaries.append(metadata)
        prefix = build_context_prefix(metadata)
        for page_number, (page_text, tables) in enumerate(pages, 1):
            if page_text.strip():
                for document in splitter.create_documents([page_text], metadatas=[{**metadata, "page_number": page_number, "chunk_type": "page_text"}]):
                    document.page_content = prefix + document.page_content
                    documents.append(document)
            line_items = [item for table in tables for item in line_items_from_table(table, page_number)] or line_items_from_text(page_text, page_number)
            for item in line_items:
                documents.append(Document(page_content=prefix + str(item["text"]), metadata={**metadata, "page_number": item["page_number"], "chunk_type": "line_item", "item_index": item["item_index"], "product": item["product"], "unit_price": item["unit_price"], "amount": item["amount"]}))
    return documents, summaries


def get_qdrant_client() -> QdrantClient:
    host = os.environ.get("QDRANT_HOST")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not host:
        raise RuntimeError("QDRANT_HOST is required")
    return QdrantClient(url=host, api_key=api_key)


def ingest_pdfs(pdf_paths: list[str | Path], vendor_overrides: dict[str, str] | None = None) -> int:
    documents, _ = extract_documents(pdf_paths, vendor_overrides)
    if not documents:
        return 0
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode([document.page_content for document in documents], show_progress_bar=True)
    points = []
    for vector, document in zip(vectors, documents):
        key = f"{document.metadata.get('source')}|{document.metadata.get('page_number')}|{document.metadata.get('chunk_type')}|{document.page_content}"
        point_id = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10**18
        points.append(PointStruct(id=point_id, vector=vector.tolist(), payload={"text": document.page_content, **document.metadata}))
    get_qdrant_client().upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
    return len(points)


if __name__ == "__main__":
    paths = sorted(Path.cwd().glob("*.pdf"))
    print(f"Ingested {ingest_pdfs(paths)} chunks into {COLLECTION_NAME}.")