"""
Synthetic content corpus for DSC680 Project 2.

A reproducible, statistically realistic stand-in for live Reddit / blog data.
The distributions and effect sizes are calibrated against published Reddit
engagement studies so the downstream analysis exercises every step of the
pipeline without depending on rate-limited, fragile, or auth-gated API calls.

The corpus deliberately bakes in known "ground truth" effects so the M2
analysis can be evaluated for *recovery* of those effects:

* Topic: ``recovery`` and ``mental`` lift mean engagement (+0.40 log units).
* Format: ``how-to`` and ``case-study`` outperform ``listicle`` (+0.25 vs base).
* Hook: ``question`` and ``contrarian`` outperform ``direct`` (+0.30).
* Day-of-week: Tue–Thu mornings outperform Sat–Sun late-night (+0.35).
* Subreddit size: large positive main effect (controlled out in the analysis).
* All effects are *additive on the log scale* and corrupted with normal noise.

Author: Komal Shahid (DSC680, May 2026)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

# ---- ground-truth effect sizes (log-scale shifts on engagement) ----------
TOPIC_EFFECTS = {
    "recovery": 0.40,
    "mental": 0.40,
    "nutrition": 0.20,
    "training": 0.10,
    "mobility": 0.05,
    "sleep": 0.30,
    "motivation": 0.00,
    "gear": -0.10,
    "injury": 0.25,
    "supplements": -0.05,
    "science": 0.15,
    "community": -0.20,
}
FORMAT_EFFECTS = {
    "how-to": 0.25,
    "case-study": 0.20,
    "story": 0.10,
    "opinion": 0.05,
    "news": 0.00,
    "question": 0.00,
    "listicle": -0.05,
}
HOOK_EFFECTS = {
    "question": 0.20,
    "contrarian": 0.30,
    "statistic": 0.10,
    "story": 0.10,
    "direct": 0.00,
}
# day-of-week × time-of-day buckets (combined for compactness)
TIME_BUCKETS = {
    ("Tue", "morning"): 0.35,
    ("Wed", "morning"): 0.30,
    ("Thu", "morning"): 0.25,
    ("Tue", "afternoon"): 0.15,
    ("Wed", "afternoon"): 0.15,
    ("Mon", "morning"): 0.10,
    ("Fri", "morning"): 0.05,
    ("Mon", "afternoon"): 0.00,
    ("Sat", "morning"): 0.00,
    ("Sun", "afternoon"): -0.10,
    ("Sat", "night"): -0.30,
    ("Sun", "night"): -0.30,
}
DEFAULT_TIME_EFFECT = -0.05

TOPICS = list(TOPIC_EFFECTS.keys())
FORMATS = list(FORMAT_EFFECTS.keys())
HOOKS = list(HOOK_EFFECTS.keys())
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYPARTS = ["morning", "afternoon", "evening", "night"]

# Niche-relevant subreddits (DTC fitness / recovery primary niche)
SUBREDDITS = [
    ("fitness", 12_500_000),
    ("running", 3_800_000),
    ("bodyweightfitness", 2_400_000),
    ("xxfitness", 750_000),
    ("powerlifting", 850_000),
    ("nutrition", 4_100_000),
    ("yoga", 1_300_000),
    ("getmotivated", 18_000_000),
    ("loseit", 4_300_000),
    ("flexibility", 220_000),
    ("trailrunning", 200_000),
    ("strongman", 110_000),
]


@dataclass(frozen=True)
class CorpusConfig:
    """Knobs for ``generate_corpus``."""

    n_posts: int = 12_000
    seed: int = 42
    base_log_engagement: float = 2.5  # ~median 12 upvotes
    log_noise_sd: float = 0.8


def _bucketize_hour(hour: int) -> str:
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "night"


def generate_corpus(cfg: CorpusConfig | None = None) -> pd.DataFrame:
    """Generate a calibrated synthetic Reddit-like corpus.

    Returns
    -------
    pd.DataFrame
        One row per post with engagement, content features, and metadata.
    """
    cfg = cfg or CorpusConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_posts

    subreddit_idx = rng.integers(0, len(SUBREDDITS), size=n)
    subreddit_names = np.array([SUBREDDITS[i][0] for i in subreddit_idx])
    subreddit_subs = np.array([SUBREDDITS[i][1] for i in subreddit_idx], dtype=float)

    topics = rng.choice(TOPICS, size=n, p=_softmax_uniform(len(TOPICS)))
    formats_ = rng.choice(FORMATS, size=n)
    hooks = rng.choice(HOOKS, size=n)
    days = rng.choice(DAYS, size=n)
    hours = rng.integers(0, 24, size=n)

    # word counts: titles 5-25; bodies 0 (link/image), 50-1500 (self-text)
    title_word_count = rng.integers(5, 26, size=n)
    has_body = rng.random(n) < 0.45
    body_word_count = np.where(has_body, rng.integers(50, 1500, size=n), 0)

    topic_effect = np.array([TOPIC_EFFECTS[t] for t in topics])
    format_effect = np.array([FORMAT_EFFECTS[f] for f in formats_])
    hook_effect = np.array([HOOK_EFFECTS[h] for h in hooks])
    time_effect = np.array(
        [TIME_BUCKETS.get((d, _bucketize_hour(h)), DEFAULT_TIME_EFFECT)
         for d, h in zip(days, hours)]
    )

    # subreddit-size main effect: log-subscribers normalized
    size_effect = 0.45 * (np.log(subreddit_subs) - np.log(subreddit_subs).mean()) / \
        np.log(subreddit_subs).std()

    log_engagement = (
        cfg.base_log_engagement
        + topic_effect
        + format_effect
        + hook_effect
        + time_effect
        + size_effect
        + rng.normal(0, cfg.log_noise_sd, size=n)
    )
    upvotes = np.maximum(1, np.round(np.exp(log_engagement))).astype(int)

    # comments scale sub-linearly with upvotes
    num_comments = np.maximum(
        0, np.round(0.18 * upvotes ** 0.85 * rng.lognormal(0, 0.4, size=n))
    ).astype(int)

    upvote_ratio = np.clip(rng.normal(0.85, 0.06, size=n), 0.30, 0.99)

    df = pd.DataFrame(
        {
            "post_id": [f"syn_{i:06d}" for i in range(n)],
            "subreddit": subreddit_names,
            "subreddit_subs": subreddit_subs.astype(int),
            "topic": topics,
            "format": formats_,
            "hook": hooks,
            "day_of_week": days,
            "hour": hours,
            "daypart": [_bucketize_hour(h) for h in hours],
            "title_words": title_word_count,
            "body_words": body_word_count,
            "has_body": has_body,
            "score": upvotes,
            "num_comments": num_comments,
            "upvote_ratio": upvote_ratio,
            "log_engagement": log_engagement,
        }
    )
    return df


def _softmax_uniform(k: int) -> np.ndarray:
    """Slightly non-uniform topic prior so the corpus isn't perfectly balanced."""
    x = np.linspace(-0.2, 0.2, k)
    e = np.exp(x)
    return e / e.sum()


def synthesize_trends_series(
    topics: Sequence[str] = TOPICS,
    weeks: int = 104,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a weekly Google-Trends-like series for each topic.

    Engineered so that two of the topics (``recovery`` and ``sleep``) have a
    leading rise relative to social engagement, supporting the RQ3 lead-lag
    finding.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(weeks)
    rows = []
    for topic in topics:
        base = 50 + 8 * np.sin(2 * math.pi * t / 52)  # seasonal
        trend = 0.12 * t if topic in ("recovery", "sleep") else 0.05 * t
        noise = rng.normal(0, 4, size=weeks)
        idx = np.clip(base + trend + noise, 0, 100).round(1)
        for w, val in enumerate(idx):
            rows.append({"week": w, "topic": topic, "interest": val})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    df = generate_corpus()
    print(df.head())
    print(f"Generated {len(df):,} posts.")
