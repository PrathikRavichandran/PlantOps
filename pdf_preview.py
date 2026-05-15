"""
pdf_preview.py — Render a single PDF page as PNG bytes using PyMuPDF.
Cache renders in app.py with @st.cache_data(ttl=3600).
"""
import os
import fitz  # PyMuPDF

DATA_DIR = "./data"


def render_page(source_filename: str, page_num: int = 1) -> bytes | None:
    pdf_path = os.path.join(DATA_DIR, source_filename)
    if not os.path.isfile(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        page_index = min(max(0, page_num - 1), len(doc) - 1)
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        png = pix.tobytes("png")
        doc.close()
        return png
    except Exception:
        return None
