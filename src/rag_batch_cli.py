"""Batch CLI for RAG indexing and querying.

Usage examples:
1) Build an index from a directory of documents:
   python -m src.rag_batch_cli index --input-dir ./my_docs --index-out ./rag_index.pkl

2) Ask a single question using a saved index:
   python -m src.rag_batch_cli query --index ./rag_index.pkl --question "What is the refund policy?"

3) Ask many questions from a text file (one per line) and write results to CSV:
   python -m src.rag_batch_cli query --index ./rag_index.pkl --questions-file ./questions.txt --out ./answers.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from src.rag_core import (
    answer_question,
    build_index,
    load_index,
    load_texts_from_directory,
    save_index,
)


def _read_questions(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Questions file not found: {path}")
    if path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        return [q for q in lines if q]
    if path.suffix.lower() == ".csv":
        questions: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "question" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'question' column.")
            for row in reader:
                q = (row.get("question") or "").strip()
                if q:
                    questions.append(q)
        return questions
    raise ValueError("Unsupported questions file type. Use .txt (one per line) or .csv with 'question' column.")


def _write_answers_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        for q, a in rows:
            writer.writerow([q, a])


def cmd_index(args: argparse.Namespace) -> None:
    docs = load_texts_from_directory(args.input_dir)
    vectorizer, matrix, metadata, chunk_texts = build_index(
        docs,
        chunk_size_tokens=args.chunk_size,
        overlap_tokens=args.overlap,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
    )
    save_index(args.index_out, vectorizer, matrix, metadata, chunk_texts)
    print(f"Index written to: {args.index_out} ({len(chunk_texts)} chunks)")


def cmd_query(args: argparse.Namespace) -> None:
    vectorizer, matrix, metadata, chunk_texts = load_index(args.index)
    outputs: List[Tuple[str, str]] = []

    if args.question:
        questions = [args.question]
    elif args.questions_file:
        questions = _read_questions(Path(args.questions_file))
    else:
        raise ValueError("Provide either --question or --questions-file.")

    for q in questions:
        answer, _ = answer_question(
            q,
            vectorizer=vectorizer,
            matrix=matrix,
            metadata=metadata,
            chunk_texts=chunk_texts,
            top_k=args.top_k,
            max_contexts=args.max_contexts,
            max_sentences=args.max_sentences,
        )
        outputs.append((q, answer))
        if not args.out:
            print(f"\nQ: {q}\nA: {answer}\n" + ("-" * 40))

    if args.out:
        _write_answers_csv(Path(args.out), outputs)
        print(f"Wrote {len(outputs)} answers to: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch CLI for RAG indexing and querying (TF-IDF + extractive QA)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build a TF-IDF index from documents.")
    p_index.add_argument("--input-dir", required=True, help="Directory containing documents (.txt, .md, .pdf, .docx).")
    p_index.add_argument("--index-out", required=False, default="./rag_index.pkl", help="Output path for the index file.")
    p_index.add_argument("--chunk-size", type=int, default=280, help="Tokens per chunk (words).")
    p_index.add_argument("--overlap", type=int, default=40, help="Token overlap between chunks.")
    p_index.add_argument("--max-features", type=int, default=50000, help="Max vocabulary size for TF-IDF.")
    p_index.add_argument("--ngram-min", type=int, default=1, help="Minimum n in n-gram range.")
    p_index.add_argument("--ngram-max", type=int, default=2, help="Maximum n in n-gram range.")
    p_index.add_argument("--min-df", type=int, default=1, help="Minimum document frequency.")
    p_index.set_defaults(func=cmd_index)

    p_query = sub.add_parser("query", help="Ask questions using an existing index.")
    p_query.add_argument("--index", required=True, help="Path to an index file created by the 'index' command.")
    group = p_query.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", help="Single question to ask.")
    group.add_argument("--questions-file", help="Path to .txt (one per line) or .csv with a 'question' column.")
    p_query.add_argument("--top-k", type=int, default=5, help="Top-k chunks to retrieve.")
    p_query.add_argument("--max-contexts", type=int, default=3, help="Number of retrieved chunks to use for answer.")
    p_query.add_argument("--max-sentences", type=int, default=5, help="Max sentences in the synthesized answer.")
    p_query.add_argument("--out", help="Optional CSV path to write answers. Prints to stdout if omitted.")
    p_query.set_defaults(func=cmd_query)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.func(args)
        return 0
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


