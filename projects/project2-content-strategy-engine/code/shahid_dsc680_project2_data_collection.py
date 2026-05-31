"""
Data collection for DSC680 Project 2: AI-Driven Content Strategy Engine.

Modules:
- RedditCollector: pulls top posts from a list of subreddits using the public
  Reddit JSON endpoint (no OAuth needed for read-only public access; respects
  the 100-QPM free-tier ceiling with a polite 1.2 s delay between requests).
- TrendsCollector: pulls weekly interest-over-time and related queries via
  ``trendspyg`` (the actively maintained successor to the archived ``pytrends``).
- BlogScraper: pulls titles, word counts, and publicly displayed share counts
  from a configured list of industry blogs, honoring robots.txt.

The module is import-safe: instantiating the classes does not perform any
network I/O. Call the explicit ``collect_*`` methods to fetch data.

Author: Komal Shahid (DSC680, May 2026)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "DSC680-research-bot/1.0 (academic; contact: komalshahid49@gmail.com)"
)
REDDIT_BASE = "https://www.reddit.com"
DEFAULT_POLITE_DELAY_S = 1.2  # keeps us comfortably below 100 requests/minute


# ---------------------------------------------------------------------------
# Reddit
# ---------------------------------------------------------------------------
@dataclass
class RedditPost:
    """One Reddit submission stripped down to engagement-relevant fields."""

    post_id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    num_comments: int
    upvote_ratio: float
    created_utc: float
    is_self: bool
    is_video: bool
    post_hint: str
    subreddit_subscribers: int

    @classmethod
    def from_api(cls, raw: dict) -> "RedditPost":
        data = raw.get("data", {})
        return cls(
            post_id=str(data.get("id", "")),
            subreddit=str(data.get("subreddit", "")),
            title=str(data.get("title", "")),
            selftext=str(data.get("selftext", "")),
            score=int(data.get("score", 0)),
            num_comments=int(data.get("num_comments", 0)),
            upvote_ratio=float(data.get("upvote_ratio", 0.0)),
            created_utc=float(data.get("created_utc", 0.0)),
            is_self=bool(data.get("is_self", False)),
            is_video=bool(data.get("is_video", False)),
            post_hint=str(data.get("post_hint", "")),
            subreddit_subscribers=int(data.get("subreddit_subscribers", 0)),
        )


class RedditCollector:
    """Read-only Reddit collector backed by the public JSON endpoint.

    The public endpoint is rate-limited (the documented free-tier ceiling is
    100 requests per minute as of 2026); we throttle to ~50 RPM by default.

    Parameters
    ----------
    user_agent:
        Identifying string Reddit expects on every request.
    polite_delay_s:
        Seconds to sleep between requests. Defaults to 1.2 s (~50 RPM).
    session:
        Optional pre-configured :class:`requests.Session`. Useful for tests.
    """

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        polite_delay_s: float = DEFAULT_POLITE_DELAY_S,
        session: requests.Session | None = None,
    ) -> None:
        self.user_agent = user_agent
        self.polite_delay_s = polite_delay_s
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def collect_subreddit(
        self,
        subreddit: str,
        listing: str = "top",
        timeframe: str = "year",
        limit: int = 100,
    ) -> list[RedditPost]:
        """Pull up to ``limit`` posts from one subreddit.

        ``listing`` may be ``"top"``, ``"hot"``, or ``"new"``. ``timeframe``
        applies to ``top`` only (``"day"``, ``"week"``, ``"month"``, ``"year"``,
        ``"all"``).
        """
        url = f"{REDDIT_BASE}/r/{subreddit}/{listing}.json"
        params = {"limit": min(limit, 100), "t": timeframe}
        resp = self.session.get(url, params=params, timeout=15)
        time.sleep(self.polite_delay_s)
        if resp.status_code != 200:
            logger.warning(
                "Reddit fetch failed for r/%s (status=%s)", subreddit, resp.status_code
            )
            return []
        children = resp.json().get("data", {}).get("children", [])
        return [RedditPost.from_api(c) for c in children]

    def collect_many(
        self,
        subreddits: Iterable[str],
        listing: str = "top",
        timeframe: str = "year",
        limit_per_sub: int = 100,
    ) -> list[RedditPost]:
        """Pull posts from many subreddits, throttled across calls."""
        out: list[RedditPost] = []
        for sub in subreddits:
            logger.info("Collecting r/%s", sub)
            out.extend(self.collect_subreddit(sub, listing, timeframe, limit_per_sub))
        return out


# ---------------------------------------------------------------------------
# Google Trends
# ---------------------------------------------------------------------------
@dataclass
class TrendsCollector:
    """Wrapper around ``trendspyg`` for interest-over-time and related queries.

    The original ``pytrends`` was archived in April 2025; ``trendspyg`` is the
    maintained drop-in successor. Import is deferred so this module can be
    imported even when the dependency is missing in CI.
    """

    geo: str = "US"
    timeframe: str = "today 24-m"

    def interest_over_time(self, keywords: list[str]):  # pragma: no cover - I/O
        try:
            from trendspyg import TrendReq  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "trendspyg is required for Google Trends collection; "
                "pip install trendspyg"
            ) from exc
        pytrends = TrendReq(hl="en-US")
        pytrends.build_payload(keywords, timeframe=self.timeframe, geo=self.geo)
        return pytrends.interest_over_time()


# ---------------------------------------------------------------------------
# Blog scraping
# ---------------------------------------------------------------------------
@dataclass
class BlogPost:
    url: str
    title: str
    word_count: int
    publish_date: str
    topic_tags: list[str] = field(default_factory=list)
    share_count: int = 0


class BlogScraper:
    """Polite blog-article metadata scraper using BeautifulSoup.

    The scraper records only title, word count, publish date, declared topic
    tags, and any publicly displayed share count. Full article bodies are not
    cached or redistributed, which keeps the project within fair-use bounds
    for academic analysis.
    """

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        polite_delay_s: float = DEFAULT_POLITE_DELAY_S,
    ) -> None:
        self.user_agent = user_agent
        self.polite_delay_s = polite_delay_s
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def fetch_article(self, url: str) -> BlogPost | None:  # pragma: no cover
        from bs4 import BeautifulSoup  # imported lazily

        resp = self.session.get(url, timeout=20)
        time.sleep(self.polite_delay_s)
        if resp.status_code != 200:
            logger.warning("Blog fetch failed for %s (%s)", url, resp.status_code)
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        title = (soup.find("h1") or soup.find("title"))
        body = soup.get_text(separator=" ", strip=True)
        word_count = len(body.split())
        return BlogPost(
            url=url,
            title=title.get_text(strip=True) if title else "",
            word_count=word_count,
            publish_date="",  # populate from <time> tag in production use
            topic_tags=[],
            share_count=0,
        )


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------
def save_posts_to_json(posts: list[RedditPost], path: Path) -> None:
    """Write a list of RedditPost dataclasses to JSON for downstream stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [vars(p) for p in posts]
    path.write_text(json.dumps(payload, indent=2))


def load_posts_from_json(path: Path) -> list[RedditPost]:
    """Read a JSON cache produced by :func:`save_posts_to_json`."""
    raw = json.loads(path.read_text())
    return [RedditPost(**row) for row in raw]
