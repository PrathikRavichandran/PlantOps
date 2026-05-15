"""
ingest.py — Parse all PDFs, chunk text, embed, and persist to ChromaDB.
Run once before launching app.py. Re-run whenever PDFs change.
"""
import os
import pdfplumber
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

DATA_DIR = "./data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "plant_docs"
CHUNK_SIZE = 600
CHUNK_OVL = 100

EQUIPMENT_CODES = {"APR", "BDP", "HLX", "NXS", "TM7"}
DOC_TYPES = {"safety_procedures", "maintenance_manual", "quality_control"}


def extract_pages(pdf_path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_num": i, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVL) -> list[str]:
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start: start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def parse_filename(filename: str) -> tuple[str, str]:
    name = filename.replace(".pdf", "")
    parts = name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    equipment, doc_type = parts[0], parts[1]
    if equipment not in EQUIPMENT_CODES:
        raise ValueError(f"Unknown equipment code '{equipment}' in {filename}")
    if doc_type not in DOC_TYPES:
        raise ValueError(f"Unknown doc type '{doc_type}' in {filename}")
    return equipment, doc_type


def ingest_pdf(collection, pdf_path: str) -> int:
    filename = os.path.basename(pdf_path)
    equipment, doc_type = parse_filename(filename)

    pages = extract_pages(pdf_path)
    if not pages:
        return 0

    # Build cumulative offset map for page_num tracking
    page_offsets = []
    combined = ""
    for p in pages:
        page_offsets.append((len(combined), p["page_num"]))
        combined += p["text"] + " "

    def get_page_num(char_offset: int) -> int:
        page_num = page_offsets[0][1]
        for offset, pnum in page_offsets:
            if char_offset >= offset:
                page_num = pnum
            else:
                break
        return page_num

    chunks = chunk_text(combined)

    documents, metadatas, ids = [], [], []
    step = CHUNK_SIZE - CHUNK_OVL
    for i, chunk in enumerate(chunks):
        char_start = i * step
        documents.append(chunk)
        metadatas.append({
            "equipment": equipment,
            "doc_type": doc_type,
            "source": filename,
            "chunk_index": i,
            "page_num": get_page_num(char_start),
        })
        ids.append(f"{equipment}_{doc_type}_{i}")

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(chunks)


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=DefaultEmbeddingFunction(),
    )

    total_chunks = 0
    pdf_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".pdf")]

    print(f"Ingesting {len(pdf_files)} PDFs into '{COLLECTION_NAME}'...")
    for filename in pdf_files:
        pdf_path = os.path.join(DATA_DIR, filename)
        n = ingest_pdf(collection, pdf_path)
        print(f"  {filename}: {n} chunks")
        total_chunks += n

    print(f"\nDone. {len(pdf_files)} PDFs, {total_chunks} total chunks.")
    print(f"ChromaDB persisted to {CHROMA_PATH}/")


if __name__ == "__main__":
    main()
