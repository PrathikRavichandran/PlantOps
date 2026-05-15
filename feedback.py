"""
feedback.py — SQLite-backed feedback logging and analytics.
"""
import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "./feedback.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                question       TEXT NOT NULL,
                doc_type       TEXT NOT NULL,
                equipment      TEXT NOT NULL,
                sources_json   TEXT NOT NULL,
                answer_snippet TEXT NOT NULL,
                rating         TEXT NOT NULL
            )
        """)
        conn.commit()


def log_feedback(
    question: str,
    routing: dict,
    sources: list,
    answer_snippet: str,
    rating: str,
) -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO feedback
                (timestamp, question, doc_type, equipment,
                 sources_json, answer_snippet, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                routing.get("doc_type", "unknown"),
                routing.get("equipment", "unknown"),
                json.dumps(sources),
                answer_snippet[:200],
                rating,
            ),
        )
        conn.commit()


def get_analytics() -> dict:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY timestamp DESC"
        ).fetchall()

    if not rows:
        return {
            "total_queries":    0,
            "pct_positive":     0.0,
            "top_doc_types":    {},
            "top_equipment":    {},
            "top_sources":      {},
            "recent_questions": [],
        }

    total = len(rows)
    positives = sum(1 for r in rows if r["rating"] == "up")

    doc_type_counts: dict = {}
    equipment_counts: dict = {}
    source_counts: dict = {}

    for r in rows:
        doc_type_counts[r["doc_type"]] = doc_type_counts.get(r["doc_type"], 0) + 1
        equipment_counts[r["equipment"]] = equipment_counts.get(r["equipment"], 0) + 1
        for src in json.loads(r["sources_json"]):
            source_counts[src] = source_counts.get(src, 0) + 1

    return {
        "total_queries":    total,
        "pct_positive":     round(100.0 * positives / total, 1),
        "top_doc_types":    dict(sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True)),
        "top_equipment":    dict(sorted(equipment_counts.items(), key=lambda x: x[1], reverse=True)),
        "top_sources":      dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "recent_questions": [r["question"] for r in rows[:10]],
    }
