"""
app.py — Plant Operator Q&A: multi-turn chat, hybrid retrieval, feedback, PDF preview.
Run: streamlit run app.py
"""
import os
import streamlit as st
import anthropic
from dotenv import load_dotenv

from router import classify_question
from retriever import get_collection, retrieve
from qa import answer_question_stream, get_sources
from feedback import init_db, log_feedback, get_analytics
from pdf_preview import render_page

load_dotenv()


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Create a .env file with your key.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def get_chroma_collection():
    return get_collection()


@st.cache_data(ttl=3600)
def cached_render_page(source_filename: str, page_num: int) -> bytes | None:
    return render_page(source_filename, page_num)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_page_num_for_source(source_filename: str, chunks: list[dict]) -> int:
    for chunk in (chunks or []):
        if chunk.get("source") == source_filename:
            return chunk.get("page_num", 1)
    return 1


def _handle_feedback(msg_index: int, rating: str) -> None:
    msg = st.session_state.messages[msg_index]
    question_text = st.session_state.messages[msg_index - 1]["content"]
    log_feedback(
        question=question_text,
        routing=msg["routing"],
        sources=msg["sources"] or [],
        answer_snippet=msg["content"],
        rating=rating,
    )
    st.session_state.messages[msg_index]["feedback"] = rating
    st.rerun()


def _render_assistant_message(i: int, msg: dict) -> None:
    st.markdown(msg["content"])

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Routing decision", expanded=False):
            st.json(msg["routing"])
    with col2:
        if msg["sources"]:
            with st.expander("Sources", expanded=True):
                for j, src in enumerate(msg["sources"], 1):
                    st.write(f"**[{j}]** `{src}`")
                    page_num = _get_page_num_for_source(src, msg["chunks"])
                    with st.expander(f"View page {page_num}", expanded=False):
                        png = cached_render_page(src, page_num)
                        if png:
                            st.image(png, use_container_width=True)
                        else:
                            st.caption("Preview unavailable.")

    with st.expander("Retrieved chunks (debug)", expanded=False):
        for k, chunk in enumerate(msg["chunks"] or []):
            dist = chunk.get("distance", "N/A")
            dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
            st.markdown(f"**Chunk {k+1}** | `{chunk['source']}` | distance: `{dist_str}`")
            st.text(chunk["text"])
            st.divider()

    if msg["feedback"] is None:
        col_up, col_down, _ = st.columns([1, 1, 8])
        with col_up:
            if st.button("👍", key=f"up_{i}"):
                _handle_feedback(i, "up")
        with col_down:
            if st.button("👎", key=f"down_{i}"):
                _handle_feedback(i, "down")
    else:
        rated = "👍" if msg["feedback"] == "up" else "👎"
        st.caption(f"You rated this answer: {rated}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Plant Operator Q&A",
        page_icon="🏭",
        layout="wide",
    )

    init_db()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Facility Reference")
        st.markdown("""
| Code | Facility |
|------|----------|
| **APR** | Aurora Petrochemical Refinery |
| **BDP** | Brookhaven Dairy Processing |
| **HLX** | Helix Pharmaceuticals Bldg 4 |
| **NXS** | Nexus Semiconductor Fab 3 |
| **TM7** | Tide Motors Assembly Plant 7 |
""")
        st.divider()
        st.markdown("""
**Document categories:**
- Safety Procedures
- Maintenance Manuals
- Quality Control Standards

Questions are automatically routed to the right category and facility.
""")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    st.title("Plant Operator Q&A System")
    st.caption("Intelligent document routing — Claude Sonnet + ChromaDB + BM25 hybrid search")

    tab_qa, tab_analytics = st.tabs(["Q&A", "Analytics"])

    # ── Q&A Tab ───────────────────────────────────────────────────────────────
    with tab_qa:
        client = get_anthropic_client()
        _ = get_chroma_collection()

        # Replay all past messages from session state (no Claude calls)
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown(msg["content"])
                else:
                    _render_assistant_message(i, msg)

        # New user input — only fires on explicit submission, not on button-click reruns
        if prompt := st.chat_input("Ask a question about your plant documents..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            # Snapshot history BEFORE appending the new user message
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Routing question to the right documentation..."):
                routing = classify_question(prompt, client)

            with st.spinner(f"Retrieving from {routing['doc_type'].replace('_', ' ')} docs..."):
                chunks = retrieve(prompt, routing["doc_type"], routing["equipment"])

            with st.chat_message("assistant"):
                full_answer = st.write_stream(
                    answer_question_stream(prompt, chunks, client, history=history)
                )

            st.session_state.messages.append({
                "role":     "assistant",
                "content":  full_answer,
                "routing":  routing,
                "sources":  get_sources(chunks),
                "chunks":   chunks,
                "feedback": None,
            })
            st.rerun()

    # ── Analytics Tab ─────────────────────────────────────────────────────────
    with tab_analytics:
        st.subheader("Usage Analytics")
        analytics = get_analytics()

        if analytics["total_queries"] == 0:
            st.info(
                "No feedback recorded yet. "
                "Answer questions and rate them with 👍 / 👎 to populate this tab."
            )
        else:
            col1, col2 = st.columns(2)
            col1.metric("Total Rated Queries", analytics["total_queries"])
            col2.metric("Positive Rating", f"{analytics['pct_positive']}%")

            st.divider()
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Top Document Types**")
                for dt, count in analytics["top_doc_types"].items():
                    st.write(f"- `{dt}`: {count}")
            with col4:
                st.markdown("**Top Facilities**")
                for eq, count in analytics["top_equipment"].items():
                    st.write(f"- `{eq}`: {count}")

            st.divider()
            st.markdown("**Top Referenced Sources**")
            for src, count in analytics["top_sources"].items():
                st.write(f"- `{src}`: {count} references")

            st.divider()
            st.markdown("**Recent Questions (last 10)**")
            for q in analytics["recent_questions"]:
                st.write(f"- {q}")

        if st.button("Refresh Analytics"):
            st.rerun()


if __name__ == "__main__":
    main()
