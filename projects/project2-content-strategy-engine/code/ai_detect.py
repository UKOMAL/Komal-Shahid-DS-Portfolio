"""
Heuristic AI-detection risk analyzer for the M2 and M3 whitepapers.

This is not a substitute for GPTZero / Turnitin AI / Originality.ai — those
are blackbox classifiers that update constantly. What we can do is measure
the well-known surface signals that AI detectors lean on:

1. AI-tell vocabulary frequency (delve, tapestry, navigate, leverage, etc.)
2. Em-dash density (LLMs overuse em-dashes vs. typical academic prose)
3. Sentence-length burstiness (LLMs produce more uniform sentence lengths;
   humans alternate short and long sentences more)
4. Parallel three-item list density ("X, Y, and Z" patterns)
5. Hedging-without-specifics density ("may", "could", "various", "several")
6. Sentence-starter monotony (% of sentences starting with The/This/It)

Each paragraph gets a 0-100 risk score combining these signals. Paragraphs
scoring > 55 are flagged as high-risk; > 70 as very-high-risk.

Usage:
    python ai_detect.py path/to/whitepaper.docx
"""
from __future__ import annotations

import re
import statistics
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# Vocabulary known to be over-represented in LLM output relative to
# professional academic writing. Curated from published AI-detection studies
# and well-documented "AI tell" lists (2024-2026).
AI_TELL_WORDS = {
    "delve", "delves", "delving",
    "tapestry", "tapestries",
    "navigate", "navigates", "navigating", "navigation",
    "leverage", "leverages", "leveraging",
    "robust", "robustness",
    "comprehensive", "comprehensively",
    "intricate", "intricacies",
    "pivotal",
    "myriad",
    "underscore", "underscores", "underscored",
    "showcase", "showcases", "showcasing",
    "moreover",
    "furthermore",
    "in conclusion",
    "in summary",
    "it is important to note",
    "it's important to note",
    "it is worth noting",
    "in today's", "in today's fast-paced",
    "fast-paced",
    "ever-evolving", "ever-changing",
    "landscape",
    "realm",
    "embark",
    "harness", "harnesses", "harnessing",
    "facilitate", "facilitates", "facilitating",
    "utilize", "utilizes", "utilizing", "utilization",
    "endeavor",
    "in the realm of",
    "a plethora of",
    "paradigm",
    "synergy", "synergies",
    "cutting-edge",
    "groundbreaking",
    "revolutionize",
    "elevate",
    "foster", "fostering",
    "garner", "garnered",
    "holistic",
    "unparalleled",
    "vibrant",
    "seamless", "seamlessly",
}

# These start tokens, when they dominate, signal LLM cadence
SENTENCE_STARTERS = ("The ", "This ", "It ", "These ", "Those ", "There ")


def read_docx_paragraphs(path: Path) -> list[str]:
    """Return non-empty paragraph texts from a .docx file."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for p in root.iter(f"{{{NS['w']}}}p"):
        runs = [
            (t.text or "")
            for t in p.iter(f"{{{NS['w']}}}t")
        ]
        text = "".join(runs).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — fine for risk-scoring purposes."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def burstiness_score(sentence_lengths: list[int]) -> float:
    """Coefficient-of-variation of sentence lengths in WORDS.

    Higher = more human-like burstiness. Lower = more uniform / LLM-like.
    We invert and rescale so larger output = MORE risk.
    """
    if len(sentence_lengths) < 2:
        return 50.0  # too short to judge — neutral
    mean = statistics.mean(sentence_lengths)
    if mean == 0:
        return 50.0
    sd = statistics.stdev(sentence_lengths)
    cov = sd / mean
    # Typical human academic prose ~0.55-0.80; LLM ~0.30-0.50
    # Map cov 0.30 -> risk 80, cov 0.80 -> risk 10
    risk = max(0.0, min(100.0, 100 - (cov - 0.20) * 130))
    return risk


def paragraph_risk(para: str) -> dict:
    """Return per-paragraph risk metrics + composite score."""
    sentences = split_sentences(para)
    sentence_lengths = [len(s.split()) for s in sentences]
    words = para.split()
    n_words = max(1, len(words))

    lower = para.lower()
    ai_tell_hits = sum(lower.count(w) for w in AI_TELL_WORDS)
    em_dash_hits = para.count("—")
    triple_lists = len(re.findall(r"\b[\w-]+,\s+[\w-]+,\s+and\s+[\w-]+\b", lower))
    starter_hits = sum(1 for s in sentences if s.startswith(SENTENCE_STARTERS))
    starter_ratio = starter_hits / max(1, len(sentences))

    burst_risk = burstiness_score(sentence_lengths)

    # Normalize to per-100-words for some signals
    ai_per_100 = (ai_tell_hits / n_words) * 100
    em_per_100 = (em_dash_hits / n_words) * 100
    triple_per_100 = (triple_lists / n_words) * 100

    # Composite: weighted sum, clipped to 0-100
    composite = (
        0.30 * min(100, ai_per_100 * 60)
        + 0.20 * min(100, em_per_100 * 25)
        + 0.20 * burst_risk
        + 0.15 * min(100, triple_per_100 * 50)
        + 0.15 * min(100, starter_ratio * 130)
    )
    composite = max(0.0, min(100.0, composite))

    return {
        "n_words": n_words,
        "n_sentences": len(sentences),
        "ai_tell_hits": ai_tell_hits,
        "em_dash_hits": em_dash_hits,
        "triple_lists": triple_lists,
        "starter_ratio": round(starter_ratio, 2),
        "burst_risk": round(burst_risk, 1),
        "composite_risk": round(composite, 1),
    }


def analyze_doc(path: Path, sample_n: int = 8) -> dict:
    paras = read_docx_paragraphs(path)
    # only score "body" paragraphs (>= 35 words) — skip headings, captions
    body = [p for p in paras if len(p.split()) >= 35]
    scored = [(i, p, paragraph_risk(p)) for i, p in enumerate(body)]
    scored.sort(key=lambda x: -x[2]["composite_risk"])

    overall_risk = (
        statistics.mean([s[2]["composite_risk"] for s in scored]) if scored else 0.0
    )

    return {
        "file": path.name,
        "n_body_paragraphs": len(body),
        "n_total_paragraphs": len(paras),
        "overall_risk": round(overall_risk, 1),
        "high_risk_count": sum(1 for s in scored if s[2]["composite_risk"] > 55),
        "very_high_risk_count": sum(1 for s in scored if s[2]["composite_risk"] > 70),
        "top_risky": [
            {
                "rank": i + 1,
                "composite_risk": s[2]["composite_risk"],
                "metrics": s[2],
                "preview": s[1][:180] + ("…" if len(s[1]) > 180 else ""),
            }
            for i, s in enumerate(scored[:sample_n])
        ],
    }


if __name__ == "__main__":
    import json

    out = {}
    for arg in sys.argv[1:]:
        path = Path(arg)
        out[path.name] = analyze_doc(path)
    print(json.dumps(out, indent=2))
