"""
research_pipeline.py
--------------------
Orchestrates the end-to-end AlphaInsight multi-agent analysis pipeline.

Pipeline stages
~~~~~~~~~~~~~~~
1. **Data collection** — yfinance fetches numerical market data
   (price, financials, OHLCV history).
2. **Technical indicators** — SMA, volatility, momentum derived locally.
3. **News & sentiment** — Yahoo Finance RSS headlines + keyword classifier.
4. **Agent 1 · DataCollectorAgent** — Gemini structures and validates the
   raw data, flags quality issues, infers missing fields.
5. **Agent 2 · AnalystAgent** — Gemini scores the company holistically
   (0-100) and generates factor-level views (valuation, growth, technicals,
   risk, bull/bear cases).
6. **Agent 3 · ReportAgent** — Gemini writes a professional Markdown memo
   synthesising all prior stages.

Fallback behaviour
~~~~~~~~~~~~~~~~~~
Each stage is fault-tolerant.  When ``GOOGLE_API_KEY`` is absent or a
Gemini call fails, the agents return safe defaults and the pipeline
continues.  The final memo will be less rich but never empty.

When Gemini is unavailable entirely, the pipeline falls back to the
original deterministic scoring path (``calculator_tools.evaluate_investment_score``).

Usage
-----
    from research_pipeline import run_financial_analysis

    memo: str = run_financial_analysis("AAPL")
    print(memo)

Command-line
------------
    python research_pipeline.py AAPL
"""

from __future__ import annotations

import logging
import sys
import time

import pandas as pd

from agents import data_collector_agent, analyst_agent, report_agent
from tools.calculator_tools import evaluate_investment_score
from tools.news_tools import get_news_headlines, get_sentiment_summary
from tools.stock_tools import (
    calculate_technical_indicators,
    get_financials,
    get_historical_data,
    get_stock_info,
)

logger = logging.getLogger(__name__)

# Seconds between sequential yfinance calls (rate-limit buffer)
_INTER_REQUEST_DELAY = 0.3

# Period cascade: try progressively shorter windows if data is missing
_HISTORY_PERIODS = ("6mo", "1y", "3mo")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_format(value: object, fmt: str, default: str = "N/A") -> str:
    """Format *value* with *fmt*, returning *default* on any failure.

    Parameters
    ----------
    value:
        Value to format.  ``None``, ``NaN``, ``0``, empty string, and the
        literals ``"nan"`` / ``"none"`` / ``"null"`` all produce *default*.
    fmt:
        A ``str.format`` template such as ``"${:.2f}"`` or ``"{}%"``.
    default:
        Fallback string (default ``"N/A"``).

    Returns
    -------
    str
        Formatted string or *default*.
    """
    if value is None:
        return default
    if isinstance(value, float) and (pd.isna(value) or value != value):
        return default
    if isinstance(value, (int, float)) and value == 0:
        return default
    if isinstance(value, str) and value.lower() in {"nan", "none", "null", ""}:
        return default
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return default


def _safe_price(stock_data: dict) -> float:
    """Extract the first non-zero price from *stock_data*.

    Tries ``current_price``, ``regularMarketPrice``, then ``price``.
    Returns ``0.0`` when all keys are absent or zero.
    """
    for key in ("current_price", "regularMarketPrice", "price"):
        val = stock_data.get(key, 0)
        if val and not (isinstance(val, float) and pd.isna(val)):
            return float(val)
    return 0.0


def _fetch_history(ticker: str) -> pd.DataFrame:
    """Attempt each period in *_HISTORY_PERIODS*, returning the first non-empty DataFrame.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.

    Returns
    -------
    pd.DataFrame
        OHLCV history, or an empty DataFrame when all periods fail.
    """
    for period in _HISTORY_PERIODS:
        try:
            hist = get_historical_data(ticker, period=period)
            if not hist.empty:
                return hist
            logger.warning("No data for %s (%s), trying next period", ticker, period)
        except Exception as exc:
            logger.warning("History fetch error for %s (%s): %s", ticker, period, exc)
    logger.warning("No historical data available for %s across all periods", ticker)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_financial_analysis(ticker: str) -> dict:
    """Run the full multi-agent AlphaInsight pipeline for *ticker*.

    The pipeline executes three Gemini-powered agents in sequence
    (DataCollector → Analyst → ReportWriter), each building on the prior
    stage's output.  When ``GOOGLE_API_KEY`` is not set or a Gemini call
    fails, each agent falls back gracefully so the function always returns
    a usable result dict.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
        Case-insensitive; normalised to upper-case internally.

    Returns
    -------
    dict
        A result dictionary with keys:

        ``"memo"``           – Markdown investment memo (str).
        ``"score"``          – Gemini analyst score 0-100 (int).
        ``"recommendation"`` – ``"BUY"``, ``"HOLD"``, or ``"SELL"`` (str).
        ``"confidence"``     – ``"High"``, ``"Medium"``, or ``"Low"`` (str).
        ``"quant_score"``    – Deterministic rule-based score 0-100 (int).
        ``"quant_rec"``      – Rule-based recommendation (str).
        ``"quant_conf"``     – Rule-based confidence (str).

    Raises
    ------
    This function is designed **not** to raise.  All exceptions are caught
    internally and produce a warning log entry plus a safe default.  Callers
    should still wrap the call in ``try / except`` as a defensive measure.

    Examples
    --------
    >>> result = run_financial_analysis("MSFT")
    >>> "RECOMMENDATION" in result["memo"]
    True
    >>> result["score"] >= 0
    True
    """
    ticker = ticker.upper()
    logger.info("Pipeline starting for %s", ticker)

    # ── Stage 1: yfinance — stock metadata ───────────────────────────────
    try:
        raw_stock = get_stock_info(ticker)
    except Exception as exc:
        logger.warning("Stock info fetch failed for %s: %s", ticker, exc)
        raw_stock = {}
    time.sleep(_INTER_REQUEST_DELAY)

    # ── Stage 2: yfinance — financial statements ──────────────────────────
    try:
        raw_financials = get_financials(ticker)
    except Exception as exc:
        logger.warning("Financials fetch failed for %s: %s", ticker, exc)
        raw_financials = {}
    time.sleep(_INTER_REQUEST_DELAY)

    # ── Stage 3: yfinance — OHLCV history + technical indicators ─────────
    hist = _fetch_history(ticker)

    _tech_defaults: dict = {
        "current_price":             0.0,
        "sma_20":                    None,
        "sma_50":                    None,
        "above_sma_20":              False,
        "position_in_52_week_range": None,
        "volatility_annual":         0.0,
        "price_change_1m":           None,
        "price_change_6m":           None,
    }
    try:
        raw_tech = calculate_technical_indicators(hist)
    except Exception as exc:
        logger.warning("Technical indicator calculation failed: %s", exc)
        raw_tech = _tech_defaults
    time.sleep(_INTER_REQUEST_DELAY)

    # ── Stage 4: News & sentiment ─────────────────────────────────────────
    try:
        headlines = get_news_headlines(ticker, limit=5)
        sentiment = get_sentiment_summary(headlines)
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", ticker, exc)
        headlines, sentiment = [], "Neutral"

    # ── Stage 5: Agent 1 — DataCollectorAgent (Gemini) ───────────────────
    logger.info("Running DataCollectorAgent for %s", ticker)
    structured_data = data_collector_agent(
        ticker=ticker,
        raw_stock=raw_stock,
        raw_financials=raw_financials,
        raw_tech=raw_tech,
    )

    # ── Stage 6: Agent 2 — AnalystAgent (Gemini) ─────────────────────────
    logger.info("Running AnalystAgent for %s", ticker)
    analysis = analyst_agent(structured_data)

    # ── Stage 7: Agent 3 — ReportAgent (Gemini) ──────────────────────────
    logger.info("Running ReportAgent for %s", ticker)
    memo = report_agent(
        ticker=ticker,
        structured_data=structured_data,
        analysis=analysis,
        headlines=headlines,
        sentiment=sentiment,
    )

    # ── Stage 8: Deterministic quant score (local cross-check) ──────────
    position_52w = raw_tech.get("position_in_52_week_range")
    position_52w_input = (
        float(position_52w)
        if position_52w is not None and not (isinstance(position_52w, float) and pd.isna(position_52w))
        else 50.0
    )
    try:
        quant = evaluate_investment_score(
            pe_ratio=raw_stock.get("pe_ratio") or 20.0,
            revenue_growth=raw_financials.get("revenue_growth_yoy") or 5.0,
            net_margin=raw_financials.get("net_margin") or 10.0,
            debt_to_equity=raw_financials.get("debt_to_equity") or 1.0,
            price_position_52w=position_52w_input,
        )
    except Exception as exc:
        logger.warning("Quant score calculation failed: %s", exc)
        quant = {"score": 50, "recommendation": "HOLD", "confidence": "Low"}

    logger.info(
        "Pipeline complete for %s — Gemini: %s %s | Quant: %s %s",
        ticker,
        analysis.get("score", "N/A"),
        analysis.get("recommendation", "N/A"),
        quant.get("score", "N/A"),
        quant.get("recommendation", "N/A"),
    )

    return {
        "memo":           memo,
        "score":          analysis.get("score", 50),
        "recommendation": analysis.get("recommendation", "HOLD"),
        "confidence":     analysis.get("confidence", "Low"),
        "quant_score":    quant.get("score", 50),
        "quant_rec":      quant.get("recommendation", "HOLD"),
        "quant_conf":     quant.get("confidence", "Low"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = run_financial_analysis(symbol)
    print(result["memo"])
    print(f"\nGemini: {result['recommendation']} ({result['score']}/100, {result['confidence']} confidence)")
    print(f"Quant:  {result['quant_rec']} ({result['quant_score']}/100, {result['quant_conf']} confidence)")