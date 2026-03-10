"""Streamlit app for real-time RAG Q/A over uploaded documents.

Run:
    streamlit run src/rag_realtime_app.py --server.port 8502
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from src.rag_core import (
    ChunkMetadata,
    answer_question,
    build_index,
    split_text_into_chunks,
)
from src.rag_core import _read_pdf_file as read_pdf_file  # reuse internal readers
from src.rag_core import _read_docx_file as read_docx_file


st.set_page_config(page_title="RAG Q/A Demo", page_icon="🧠", layout="wide")

st.title("🧠 RAG Q/A Demo (Local, Extractive)")
st.caption(
    "Upload documents, build an index, and ask questions. "
    "This demo uses TF‑IDF retrieval with extractive answer synthesis—no external LLM required."
)


def _read_uploaded_text(file) -> str:
    return file.getvalue().decode("utf-8", errors="ignore")


def _read_uploaded_pdf(file) -> str:
    # Save to a temporary BytesIO for PyPDF2 reader
    # Use the rag_core reader by writing to a temporary file-like path
    data = BytesIO(file.getvalue())
    try:
        # PyPDF2's PdfReader supports file-like objects
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyPDF2 is required to read PDF files.") from exc
    reader = PdfReader(data)
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _read_uploaded_docx(file) -> str:
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise RuntimeError("python-docx is required to read DOCX files.") from exc
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (tokens)", min_value=80, max_value=1000, value=280, step=20)
    overlap = st.number_input("Overlap (tokens)", min_value=0, max_value=400, value=40, step=10)
    top_k = st.number_input("Top‑k retrieval", min_value=1, max_value=20, value=5, step=1)
    max_contexts = st.number_input("Contexts used", min_value=1, max_value=10, value=3, step=1)
    max_sentences = st.number_input("Answer sentences", min_value=1, max_value=10, value=5, step=1)

    st.markdown("---")
    st.caption("Supported: .txt, .md, .pdf, .docx")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    build_btn = st.button("Build / Rebuild Index", use_container_width=True)


def _build_index_from_uploads(
    uploads,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
):
    docs: List[Tuple[str, str]] = []
    for f in uploads:
        name = f.name
        suffix = Path(name).suffix.lower()
        if suffix in (".txt", ".md"):
            text = _read_uploaded_text(f)
        elif suffix == ".pdf":
            text = _read_uploaded_pdf(f)
        elif suffix == ".docx":
            text = _read_uploaded_docx(f)
        else:
            st.warning(f"Unsupported file type: {name}")
            continue
        if text.strip():
            docs.append((name, text))
    if not docs:
        raise ValueError("No readable content from uploads.")
    return build_index(
        docs,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
    )


if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if uploaded_files and build_btn:
    with st.spinner("Building index..."):
        try:
            vectorizer, matrix, metadata, chunk_texts = _build_index_from_uploads(
                uploaded_files,
                chunk_size_tokens=int(chunk_size),
                overlap_tokens=int(overlap),
            )
            st.session_state.rag_index = {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "metadata": metadata,
                "chunk_texts": chunk_texts,
            }
            st.success(f"Index ready ({len(chunk_texts)} chunks).")
        except Exception as exc:
            st.error(f"Failed to build index: {exc}")


st.markdown("### Ask a question")
question = st.text_input("Your question", placeholder="e.g., What are the key eligibility requirements?")
go = st.button("Get Answer", type="primary")

if go:
    idx = st.session_state.get("rag_index")
    if not idx:
        st.warning("Please upload documents and build the index first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and synthesizing..."):
            try:
                answer, results = answer_question(
                    question,
                    vectorizer=idx["vectorizer"],
                    matrix=idx["matrix"],
                    metadata=idx["metadata"],
                    chunk_texts=idx["chunk_texts"],
                    top_k=int(top_k),
                    max_contexts=int(max_contexts),
                    max_sentences=int(max_sentences),
                )
                st.subheader("Answer")
                st.write(answer if answer else "No relevant answer found.")

                st.subheader("Retrieved Contexts")
                for score, md, text in results:
                    with st.expander(f"[{score:.3f}] {md.source_path}  (tokens {md.start_token_index}-{md.end_token_index})"):
                        st.write(text)
            except Exception as exc:
                st.error(f"Failed to answer: {exc}")


