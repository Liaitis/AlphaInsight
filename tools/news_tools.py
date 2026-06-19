"""
news_tools.py
-------------
News retrieval and sentiment classification utilities for AlphaInsight.

News source
~~~~~~~~~~~
Headlines are pulled from the Yahoo Finance RSS feed
(``feeds.finance.yahoo.com``).  The feed is public and does not require an
API key, but it is rate-limited; avoid calling ``get_news_headlines`` in a
tight loop.

Gemini integration
~~~~~~~~~~~~~~~~~~
When ``GOOGLE_API_KEY`` is set, ``get_sentiment_summary`` passes the
headlines to Gemini for a richer, context-aware sentiment classification
instead of the fallback keyword-count heuristic.  Gemini understands
negation (``"not profitable"``), irony, and multi-topic headlines that
simple keyword matching cannot handle.

Keyword fallback
~~~~~~~~~~~~~~~~
When Gemini is unavailable, sentiment falls back to a ``frozenset``
intersection heuristic.  It is fast and deterministic but loses accuracy
on ambiguous or ironic headlines.

Placeholder behaviour
~~~~~~~~~~~~~~~~~~~~~
When the RSS fetch fails or returns zero items the module returns a small
set of synthetic placeholder headlines clearly marked as such, so the UI
is never left empty.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import TypedDict
from xml.etree import ElementTree as ET

import requests
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini setup (shared with agents.py — key read once at import time)
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if _API_KEY:
    genai.configure(api_key=_API_KEY)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Headline(TypedDict):
    """A single news headline record."""

    title: str
    """Headline text."""

    date: str
    """Publication date as a string (ISO 8601 when available)."""

    source: str
    """Name of the publishing outlet."""

    link: str
    """URL to the full article, or ``"#"`` when unavailable."""

    placeholder: bool
    """``True`` when the headline is synthetic fallback data, not real news."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YAHOO_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s={ticker}&region=US&lang=en-US"
)
_REQUEST_TIMEOUT_SECONDS = 10
_GEMINI_MODEL = "gemini-2.0-flash"

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "upgrade", "beat", "profit", "growth", "positive", "strong",
    "buy", "gain", "record", "rally", "surge", "outperform",
    "bullish", "expansion", "dividend", "innovative",
})
_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "downgrade", "miss", "loss", "decline", "negative", "sell",
    "drop", "lawsuit", "investigation", "fall", "slump", "bearish",
    "recall", "layoff", "fraud", "bankruptcy", "warning",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_news_headlines(ticker: str, limit: int = 5) -> list[Headline]:
    """Fetch recent news headlines for *ticker* from Yahoo Finance RSS.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
        Case-insensitive; converted to upper-case internally.
    limit:
        Maximum number of headlines to return (default ``5``, max ``20``).

    Returns
    -------
    list[Headline]
        A list of headline dicts.  Never empty — falls back to clearly-marked
        synthetic placeholders when the live feed is unavailable.
        Check the ``"placeholder"`` key to distinguish real vs. synthetic
        headlines in the UI.

    Examples
    --------
    >>> headlines = get_news_headlines("AAPL", limit=3)
    >>> len(headlines) <= 3
    True
    >>> "title" in headlines[0]
    True
    """
    limit = min(limit, 20)
    ticker = ticker.upper()
    headlines: list[Headline] = []

    url = _YAHOO_RSS_URL.format(ticker=ticker)
    try:
        response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS,headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        headlines = _parse_rss_xml(response.content, limit=limit)
    except requests.RequestException as exc:
        logger.warning("Yahoo Finance RSS fetch failed for %s: %s", ticker, exc)
    except ET.ParseError as exc:
        logger.warning("RSS XML parse error for %s: %s", ticker, exc)
    print(f"DEBUG: Fetched {len(headlines)} headlines. Error log: check logs above.")
    if not headlines:
        logger.info("Using synthetic fallback headlines for %s", ticker)
        headlines = _synthetic_fallback(ticker, limit=limit)

    return headlines


def get_sentiment_summary(headlines: list[Headline]) -> str:
    """Classify the aggregate sentiment of a list of headlines.

    Uses Gemini when ``GOOGLE_API_KEY`` is set for context-aware
    classification.  Falls back to a keyword-count heuristic otherwise.

    Parameters
    ----------
    headlines:
        A list of headline dicts as returned by :func:`get_news_headlines`.
        An empty list returns ``"Neutral"``.

    Returns
    -------
    str
        One of ``"Positive"``, ``"Negative"``, or ``"Neutral"``.

    Notes
    -----
    Gemini path: headlines are passed as a JSON list; the model is asked
    to respond with exactly one of the three labels and nothing else.
    The response is stripped and title-cased before being returned.

    Keyword fallback: each headline title is tokenised by whitespace and
    matched against positive/negative frozensets via set intersection.
    No negation handling; no weighting by recency or source.

    Examples
    --------
    >>> hl = [{"title": "AAPL surges on record earnings beat", "date": "",
    ...        "source": "", "link": "", "placeholder": False}]
    >>> get_sentiment_summary(hl) in ("Positive", "Negative", "Neutral")
    True
    """
    if not headlines:
        return "Neutral"

    if _API_KEY:
        return _gemini_sentiment(headlines)

    return _keyword_sentiment(headlines)


# ---------------------------------------------------------------------------
# Gemini-powered sentiment
# ---------------------------------------------------------------------------

def _gemini_sentiment(headlines: list[Headline]) -> str:
    """Use Gemini to classify aggregate headline sentiment.

    Parameters
    ----------
    headlines:
        List of headline dicts.

    Returns
    -------
    str
        ``"Positive"``, ``"Negative"``, or ``"Neutral"``.
        Falls back to the keyword heuristic on any API error.
    """
    # Skip placeholders — they carry no real signal
    real_titles = [
        h["title"] for h in headlines
        if not h.get("placeholder", False)
    ]
    if not real_titles:
        return "Neutral"

    prompt = f"""You are a financial sentiment analyst.
Classify the overall market sentiment conveyed by these stock news headlines.
Consider tone, implications, and context — not just keyword matching.

Headlines:
{json.dumps(real_titles, indent=2)}

Respond with EXACTLY one word: Positive, Negative, or Neutral.
No explanation. No punctuation. Just the single word."""

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            generation_config=genai.GenerationConfig(temperature=0.0),
        )
        response = model.generate_content(prompt)
        label = response.text.strip().title()
        if label in ("Positive", "Negative", "Neutral"):
            logger.info("Gemini sentiment: %s", label)
            return label
        logger.warning("Unexpected Gemini sentiment response: %s", response.text)
    except Exception as exc:
        logger.warning("Gemini sentiment call failed: %s", exc)

    return _keyword_sentiment(headlines)


# ---------------------------------------------------------------------------
# Keyword fallback sentiment
# ---------------------------------------------------------------------------

def _keyword_sentiment(headlines: list[Headline]) -> str:
    """Classify sentiment via keyword-count heuristic.

    Parameters
    ----------
    headlines:
        List of headline dicts.

    Returns
    -------
    str
        ``"Positive"``, ``"Negative"``, or ``"Neutral"``.
    """
    positive_count = 0
    negative_count = 0

    for headline in headlines:
        title_tokens = set(headline.get("title", "").lower().split())
        positive_count += len(title_tokens & _POSITIVE_WORDS)
        negative_count += len(title_tokens & _NEGATIVE_WORDS)

    if positive_count > negative_count:
        return "Positive"
    if negative_count > positive_count:
        return "Negative"
    return "Neutral"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_rss_xml(raw_xml: bytes, limit: int) -> list[Headline]:
    """Parse Yahoo Finance RSS XML bytes into a list of :class:`Headline` dicts.

    Parameters
    ----------
    raw_xml:
        Raw bytes from the RSS endpoint.
    limit:
        Maximum items to return.

    Returns
    -------
    list[Headline]
        Parsed headlines.  Empty list when no ``<item>`` elements are found.
    """
    root = ET.fromstring(raw_xml)
    headlines: list[Headline] = []

    for item in root.findall(".//item"):
        if len(headlines) >= limit:
            break

        title_el = item.find("title")
        date_el  = item.find("pubDate")
        link_el  = item.find("link")

        headlines.append(
            Headline(
                title=title_el.text.strip() if title_el is not None and title_el.text else "",
                date=date_el.text.strip()   if date_el  is not None and date_el.text  else "",
                source="Yahoo Finance",
                link=link_el.text.strip()   if link_el  is not None and link_el.text  else "#",
                placeholder=False,
            )
        )

    return headlines


def _synthetic_fallback(ticker: str, limit: int) -> list[Headline]:
    """Return generic placeholder headlines when the live feed is unavailable.

    Parameters
    ----------
    ticker:
        Ticker symbol used to personalise the placeholder text.
    limit:
        Maximum number of items to return.

    Returns
    -------
    list[Headline]
        Placeholder headlines clearly marked with ``placeholder=True``.
        These are never real news and should be displayed with a UI warning.
    """
    today = datetime.now()
    placeholders: list[Headline] = [
        Headline(
            title=f"{ticker} market update: trading active",
            date=today.strftime("%Y-%m-%d"),
            source="[No live data]",
            link="#",
            placeholder=True,
        ),
        Headline(
            title=f"Analysts release updated estimates for {ticker}",
            date=(today - timedelta(days=1)).strftime("%Y-%m-%d"),
            source="[No live data]",
            link="#",
            placeholder=True,
        ),
        Headline(
            title=f"{ticker} among top movers today",
            date=(today - timedelta(days=2)).strftime("%Y-%m-%d"),
            source="[No live data]",
            link="#",
            placeholder=True,
        ),
    ]
    return placeholders[:limit]