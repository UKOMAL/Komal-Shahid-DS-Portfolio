"""Core RAG utilities: indexing, querying, and simple extractive answering.

This module provides a minimal, dependency-light Retrieval-Augmented Generation
pipeline using TF-IDF for retrieval and extractive synthesis for answering.

Key capabilities:
- Load documents from a directory (.txt, .md, .pdf, .docx)
- Chunk documents with overlap for better retrieval recall
- Build and persist a TF-IDF index
- Query the index and return top-k contexts
- Synthesize an extractive answer from retrieved contexts

No external LLM is required; this is designed to run locally with packages
already present in the repository requirements.
"""
from __future__ import annotations

import io
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # Optional readers (guarded imports)
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None  # type: ignore

try:
    import docx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    docx = None  # type: ignore


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata describing an individual text chunk."""

    chunk_id: int
    source_path: str
    start_token_index: int
    end_token_index: int


def _read_text_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "PyPDF2 is not available but a .pdf file was provided. "
            "Install PyPDF2 or remove .pdf files."
        )
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


def _read_docx_file(path: Path) -> str:
    if docx is None:
        raise RuntimeError(
            "python-docx is not available but a .docx file was provided. "
            "Install python-docx or remove .docx files."
        )
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_texts_from_directory(input_dir: str, allowed_exts: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    """Load texts from a directory with supported extensions.

    Args:
        input_dir: Directory containing source documents.
        allowed_exts: Optional list of file extensions to include; defaults to SUPPORTED_EXTENSIONS.

    Returns:
        List of tuples (source_path, text).
    """
    src_dir = Path(input_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")

    exts = set(e.lower() for e in (allowed_exts or SUPPORTED_EXTENSIONS))
    results: List[Tuple[str, str]] = []

    for path in sorted(src_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        text: str
        if path.suffix.lower() in {".txt", ".md"}:
            text = _read_text_file(path)
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf_file(path)
        elif path.suffix.lower() == ".docx":
            text = _read_docx_file(path)
        else:
            # Should not happen due to extension filter
            continue
        if text.strip():
            results.append((str(path), text))
    return results


def split_text_into_chunks(
    text: str,
    chunk_size_tokens: int = 280,
    overlap_tokens: int = 40,
) -> List[Tuple[int, int, str]]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Full document text.
        chunk_size_tokens: Target number of whitespace-delimited tokens per chunk.
        overlap_tokens: Number of tokens to overlap between consecutive chunks.

    Returns:
        List of tuples (start_token_index, end_token_index, chunk_text).
    """
    tokens = text.split()
    if not tokens:
        return []

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < chunk_size_tokens.")

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    step = chunk_size_tokens - overlap_tokens
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_text = " ".join(tokens[start:end])
        chunks.append((start, end, chunk_text))
        if end == len(tokens):
            break
        start += step
    return chunks


def build_index(
    docs: Sequence[Tuple[str, str]],
    *,
    chunk_size_tokens: int = 280,
    overlap_tokens: int = 40,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> Tuple[TfidfVectorizer, np.ndarray, List[ChunkMetadata], List[str]]:
    """Build a TF-IDF index over chunked documents.

    Args:
        docs: Sequence of (source_path, text).
        chunk_size_tokens: Words per chunk.
        overlap_tokens: Overlap in words between chunks.
        max_features: Cap vocabulary size for memory efficiency.
        ngram_range: N-gram range for vectorizer.
        min_df: Minimum document frequency for terms.

    Returns:
        (vectorizer, matrix, chunk_metadata, chunk_texts)
    """
    chunk_texts: List[str] = []
    metadata: List[ChunkMetadata] = []
    next_chunk_id = 0

    for source_path, text in docs:
        chunks = split_text_into_chunks(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        for start_idx, end_idx, chunk_text in chunks:
            metadata.append(
                ChunkMetadata(
                    chunk_id=next_chunk_id,
                    source_path=source_path,
                    start_token_index=start_idx,
                    end_token_index=end_idx,
                )
            )
            chunk_texts.append(chunk_text)
            next_chunk_id += 1

    if not chunk_texts:
        raise ValueError("No text content found to index. Check your input directory and filters.")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        # Basic cleanup; leave stop_words=None for domain specificity
        strip_accents="unicode",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(chunk_texts)
    return vectorizer, matrix, metadata, chunk_texts


def save_index(
    index_path: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    metadata: Sequence[ChunkMetadata],
    chunk_texts: Sequence[str],
) -> None:
    """Persist index artifacts to disk using pickle."""
    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": metadata,
        "chunk_texts": list(chunk_texts),
    }
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_index(index_path: str) -> Tuple[TfidfVectorizer, np.ndarray, List[ChunkMetadata], List[str]]:
    """Load a previously saved index from disk."""
    path = Path(index_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    # Basic payload validation
    required = {"vectorizer", "matrix", "metadata", "chunk_texts"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Corrupt index file. Missing keys: {sorted(missing)}")
    return (
        payload["vectorizer"],
        payload["matrix"],
        payload["metadata"],
        payload["chunk_texts"],
    )


def search(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    metadata: Sequence[ChunkMetadata],
    chunk_texts: Sequence[str],
    *,
    top_k: int = 5,
) -> List[Tuple[float, ChunkMetadata, str]]:
    """Search the index and return top_k results.

    Returns:
        List of tuples (score, chunk_metadata, chunk_text) sorted by descending score.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    # argsort descending
    top_indices = np.argsort(-sims)[: max(top_k, 1)]
    results: List[Tuple[float, ChunkMetadata, str]] = []
    for idx in top_indices:
        score = float(sims[idx])
        md = metadata[int(idx)]
        text = chunk_texts[int(idx)]
        results.append((score, md, text))
    return results


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def synthesize_answer_extractive(
    question: str,
    contexts: Sequence[str],
    *,
    max_sentences: int = 5,
) -> str:
    """Simple extractive 'answer' by selecting sentences with highest token overlap.

    This is a pragmatic approach when no LLM is available. It surfaces the most
    relevant sentences from the retrieved contexts.
    """
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return ""

    candidate_sentences: List[Tuple[float, str]] = []
    for ctx in contexts:
        for sent in _SENT_SPLIT_RE.split(ctx.strip()):
            if not sent:
                continue
            s_tokens = set(_tokenize(sent))
            if not s_tokens:
                continue
            # Jaccard similarity as a simple relevance proxy
            inter = len(q_tokens & s_tokens)
            union = len(q_tokens | s_tokens)
            score = inter / union if union else 0.0
            if score > 0:
                candidate_sentences.append((score, sent.strip()))

    if not candidate_sentences:
        # fallback: return first few sentences from top contexts
        fallback = []
        for ctx in contexts:
            parts = [p.strip() for p in _SENT_SPLIT_RE.split(ctx) if p.strip()]
            fallback.extend(parts[: max(1, max_sentences // 2)])
            if len(fallback) >= max_sentences:
                break
        return " ".join(fallback[:max_sentences])

    candidate_sentences.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in candidate_sentences[:max_sentences]]
    return " ".join(best)


def answer_question(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: np.ndarray,
    metadata: Sequence[ChunkMetadata],
    chunk_texts: Sequence[str],
    *,
    top_k: int = 5,
    max_contexts: int = 3,
    max_sentences: int = 5,
) -> Tuple[str, List[Tuple[float, ChunkMetadata, str]]]:
    """Retrieve top_k contexts and synthesize an extractive answer."""
    results = search(
        query=query,
        vectorizer=vectorizer,
        matrix=matrix,
        metadata=metadata,
        chunk_texts=chunk_texts,
        top_k=top_k,
    )
    contexts = [t for _, _, t in results[:max_contexts]]
    answer = synthesize_answer_extractive(query, contexts, max_sentences=max_sentences)
    return answer, results



