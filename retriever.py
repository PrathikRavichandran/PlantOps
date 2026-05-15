"""
retriever.py — Hybrid BM25 + semantic search with RRF fusion and distance fallback.
"""
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "plant_docs"
N_RESULTS_SPECIFIC = 5
N_RESULTS_ALL = 8
DISTANCE_FALLBACK_THRESHOLD = 0.6
RRF_K = 60

_collection = None
_all_chunks: list[dict] = []


def get_collection():
    global _collection, _all_chunks
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
        )
        _all_chunks = _load_all_chunks(_collection)
    return _collection


def _load_all_chunks(collection) -> list[dict]:
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta, cid in zip(result["documents"], result["metadatas"], result["ids"]):
        chunks.append({
            "id":          cid,
            "text":        doc,
            "source":      meta["source"],
            "equipment":   meta["equipment"],
            "doc_type":    meta["doc_type"],
            "chunk_index": meta["chunk_index"],
            "page_num":    meta.get("page_num", 1),
        })
    return chunks


def build_where_filter(doc_type: str, equipment: str) -> dict:
    if equipment == "all":
        return {"doc_type": {"$eq": doc_type}}
    return {
        "$and": [
            {"equipment": {"$eq": equipment}},
            {"doc_type":  {"$eq": doc_type}},
        ]
    }


def _semantic_search(
    question: str, doc_type: str, equipment: str, n: int
) -> tuple[list[dict], float]:
    collection = get_collection()
    where = build_where_filter(doc_type, equipment)
    results = collection.query(
        query_texts=[question],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "id":          f"{meta['equipment']}_{meta['doc_type']}_{meta['chunk_index']}",
            "text":        doc,
            "source":      meta["source"],
            "equipment":   meta["equipment"],
            "doc_type":    meta["doc_type"],
            "chunk_index": meta["chunk_index"],
            "page_num":    meta.get("page_num", 1),
            "distance":    dist,
        })
    chunks.sort(key=lambda x: x["distance"])
    best_dist = chunks[0]["distance"] if chunks else 1.0
    return chunks, best_dist


def _bm25_search(question: str, doc_type: str, equipment: str) -> list[str]:
    if equipment == "all":
        subset = [c for c in _all_chunks if c["doc_type"] == doc_type]
    else:
        subset = [
            c for c in _all_chunks
            if c["doc_type"] == doc_type and c["equipment"] == equipment
        ]
    if not subset:
        return []

    tokenized = [c["text"].lower().split() for c in subset]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(question.lower().split())
    ranked = sorted(
        zip([c["id"] for c in subset], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [cid for cid, _ in ranked]


def _rrf_merge(
    semantic_chunks: list[dict],
    bm25_ranked_ids: list[str],
    k: int = RRF_K,
) -> list[dict]:
    id_to_chunk: dict[str, dict] = {c["id"]: c for c in semantic_chunks}

    # Include BM25-only chunks not already in semantic results
    bm25_id_set = set(bm25_ranked_ids)
    for c in _all_chunks:
        if c["id"] in bm25_id_set and c["id"] not in id_to_chunk:
            id_to_chunk[c["id"]] = {**c, "distance": 1.0}

    scores: dict[str, float] = {}
    for rank, chunk in enumerate(semantic_chunks):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(bm25_ranked_ids):
        if cid in id_to_chunk:
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [id_to_chunk[cid] for cid in sorted_ids if cid in id_to_chunk]


def retrieve(question: str, doc_type: str, equipment: str) -> list[dict]:
    n = N_RESULTS_ALL if equipment == "all" else N_RESULTS_SPECIFIC

    semantic_chunks, best_distance = _semantic_search(question, doc_type, equipment, n)

    # Widen search if closest chunk is still quite dissimilar
    effective_equipment = equipment
    if best_distance > DISTANCE_FALLBACK_THRESHOLD and equipment != "all":
        semantic_chunks, best_distance = _semantic_search(
            question, doc_type, "all", N_RESULTS_ALL
        )
        effective_equipment = "all"
        n = N_RESULTS_ALL

    bm25_ids = _bm25_search(question, doc_type, effective_equipment)
    fused = _rrf_merge(semantic_chunks, bm25_ids)
    return fused[:n]
