"""
agents.py
---------
Multi-agent analysis layer for AlphaInsight, powered by Google Gemini.

Architecture
~~~~~~~~~~~~
Three specialised agents execute in sequence, each with a distinct
responsibility.  The orchestrator in ``research_pipeline.py`` calls them
in order and passes structured data between stages.

  ┌─────────────────────────────────────────────────────────────────┐
  │  yfinance (numerical market data)                               │
  └───────────────────────┬─────────────────────────────────────────┘
                          │ raw dict
                          ▼
          ┌───────────────────────────────┐
          │  Agent 1 · DataCollectorAgent │  Structures + enriches raw data
          └───────────────┬───────────────┘
                          │ structured JSON
                          ▼
          ┌───────────────────────────────┐
          │  Agent 2 · AnalystAgent       │  Scores, interprets, reasons
          └───────────────┬───────────────┘
                          │ analysis JSON
                          ▼
          ┌───────────────────────────────┐
          │  Agent 3 · ReportAgent        │  Writes the investment memo
          └───────────────┬───────────────┘
                          │ markdown string
                          ▼
                     dashboard.py

Model
~~~~~
All agents use ``gemini-2.5-flash`` with ``temperature=0.2`` for consistent,
near-deterministic output.  Set ``GEMINI_TEMPERATURE`` in the environment to
override (range 0.0–1.0; lower = more deterministic).

Configuration
~~~~~~~~~~~~~
Set the ``GOOGLE_API_KEY`` environment variable before running::

    export GOOGLE_API_KEY=your_key_here

Or add it to a ``.env`` file and load it with ``python-dotenv``::

    pip install python-dotenv
    # then at the top of your entry point:
    from dotenv import load_dotenv; load_dotenv()

Rate limits
~~~~~~~~~~~
Gemini free-tier allows roughly 15 requests per minute.  With three agents
per analysis run plus a 0.5 s inter-call delay the pipeline stays well
within that limit for single-user use.

Error handling
~~~~~~~~~~~~~~
Each agent wraps its Gemini call in a try/except and returns a safe default
dict or string so the pipeline never crashes on a transient API error.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import google.generativeai as genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client — initialised once at import time
# ---------------------------------------------------------------------------

_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not _API_KEY:
    logger.warning(
        "GOOGLE_API_KEY is not set.  Agents will return safe defaults. "
        "Set the variable and restart the application."
    )
else:
    genai.configure(api_key=_API_KEY)

_MODEL_NAME   = "gemini-2.5-flash"
_TEMPERATURE  = float(os.environ.get("GEMINI_TEMPERATURE", "0.2"))
_INTER_AGENT_DELAY = 0.5   # seconds between Gemini calls (rate-limit buffer)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model() -> genai.GenerativeModel:
    """Return a configured GenerativeModel instance.

    Returns
    -------
    genai.GenerativeModel
        Model configured with the project's standard generation parameters.
    """
    return genai.GenerativeModel(
        model_name=_MODEL_NAME,
        generation_config=genai.GenerationConfig(
            temperature=_TEMPERATURE,
            response_mime_type="text/plain",   # agents parse the text themselves
        ),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in *text*.

    Gemini sometimes wraps JSON in markdown fences (```json ... ```).
    This function strips those fences before parsing.

    Parameters
    ----------
    text:
        Raw text returned by the Gemini API.

    Returns
    -------
    dict
        Parsed JSON object, or an empty dict on any parse failure.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Find the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in Gemini response: %.200s", text)
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s — raw: %.200s", exc, text)
        return {}


# ---------------------------------------------------------------------------
# Agent 1 — Data Collector
# ---------------------------------------------------------------------------

def data_collector_agent(
    ticker: str,
    raw_stock: dict[str, Any],
    raw_financials: dict[str, Any],
    raw_tech: dict[str, Any],
) -> dict[str, Any]:
    """Structure and enrich raw financial data using Gemini.

    This agent takes the dictionaries produced by ``stock_tools`` and
    ``calculator_tools``, asks Gemini to validate, clean, and add context
    (e.g. sector interpretation, data-quality flags), then returns a
    normalised dictionary that Agent 2 will score.

    Parameters
    ----------
    ticker:
        Normalised upper-case ticker symbol.
    raw_stock:
        Output of ``get_stock_info(ticker)``.
    raw_financials:
        Output of ``get_financials(ticker)``.
    raw_tech:
        Output of ``calculate_technical_indicators(hist)``.

    Returns
    -------
    dict
        Structured data dictionary with keys:

        ``ticker``           – Ticker symbol.
        ``company_name``     – Full company name inferred by Gemini.
        ``sector``           – GICS sector.
        ``current_price``    – Current price (float).
        ``market_cap_b``     – Market cap in USD billions.
        ``pe_ratio``         – Trailing P/E.
        ``forward_pe``       – Forward P/E.
        ``revenue_growth``   – YoY revenue growth (%).
        ``net_margin``       – Net margin (%).
        ``gross_margin``     – Gross margin (%).
        ``debt_to_equity``   – D/E ratio.
        ``sma_20``           – 20-day SMA.
        ``sma_50``           – 50-day SMA.
        ``volatility``       – Annualised volatility (%).
        ``position_52w``     – 52-week range position (%).
        ``price_change_6m``  – 6-month price change (%).
        ``above_sma_20``     – Boolean: price > 20-day SMA.
        ``data_quality``     – Gemini's assessment: "good" / "partial" / "poor".
        ``data_notes``       – Any caveats Gemini flagged about the data.

        Falls back to a dict built directly from the raw inputs when the
        Gemini call fails.

    Notes
    -----
    The agent prompt explicitly asks for JSON only so that ``_extract_json``
    can parse the response reliably.
    """
    if not _API_KEY:
        return _collector_fallback(ticker, raw_stock, raw_financials, raw_tech)

    prompt = f"""
You are a financial data engineer. Your job is to validate and structure
raw stock data into a clean JSON object for downstream analysis.

Ticker: {ticker}

RAW STOCK INFO:
{json.dumps(raw_stock, default=str, indent=2)}

RAW FINANCIALS:
{json.dumps(raw_financials, default=str, indent=2)}

RAW TECHNICAL INDICATORS:
{json.dumps(raw_tech, default=str, indent=2)}

Instructions:
1. Extract and validate the key fields listed below.
2. Flag any field that is missing, zero, or suspicious as a data quality issue.
3. Infer the full company name from the ticker if not present in the raw data.
4. Return ONLY a valid JSON object with exactly these keys:

{{
  "ticker": string,
  "company_name": string,
  "sector": string,
  "current_price": number,
  "market_cap_b": number,
  "pe_ratio": number or null,
  "forward_pe": number or null,
  "revenue_growth": number or null,
  "net_margin": number or null,
  "gross_margin": number or null,
  "debt_to_equity": number or null,
  "sma_20": number or null,
  "sma_50": number or null,
  "volatility": number or null,
  "position_52w": number or null,
  "price_change_6m": number or null,
  "above_sma_20": boolean,
  "data_quality": "good" | "partial" | "poor",
  "data_notes": string
}}

Return only the JSON — no preamble, no explanation, no markdown fences.
"""

    try:
        model = _get_model()
        response = model.generate_content(prompt)
        structured = _extract_json(response.text)
        if structured:
            logger.info("DataCollectorAgent: structured data for %s (quality: %s)",
                        ticker, structured.get("data_quality", "unknown"))
            return structured
    except Exception as exc:
        logger.warning("DataCollectorAgent Gemini call failed: %s", exc)

    return _collector_fallback(ticker, raw_stock, raw_financials, raw_tech)


def _collector_fallback(
    ticker: str,
    raw_stock: dict,
    raw_financials: dict,
    raw_tech: dict,
) -> dict[str, Any]:
    """Build a structured dict directly from raw inputs when Gemini is unavailable."""
    mc = raw_stock.get("market_cap", 0) or 0
    return {
        "ticker":         ticker,
        "company_name":   raw_stock.get("industry", ticker),
        "sector":         raw_stock.get("sector", "Unknown"),
        "current_price":  raw_stock.get("current_price", 0.0),
        "market_cap_b":   round(mc / 1e9, 2) if mc else None,
        "pe_ratio":       raw_stock.get("pe_ratio"),
        "forward_pe":     raw_stock.get("forward_pe"),
        "revenue_growth": raw_financials.get("revenue_growth_yoy"),
        "net_margin":     raw_financials.get("net_margin"),
        "gross_margin":   raw_financials.get("gross_margin"),
        "debt_to_equity": raw_financials.get("debt_to_equity"),
        "sma_20":         raw_tech.get("sma_20"),
        "sma_50":         raw_tech.get("sma_50"),
        "volatility":     raw_tech.get("volatility_annual"),
        "position_52w":   raw_tech.get("position_in_52_week_range"),
        "price_change_6m": raw_tech.get("price_change_6m"),
        "above_sma_20":   raw_tech.get("above_sma_20", False),
        "data_quality":   "partial",
        "data_notes":     "Gemini unavailable — data passed through directly from yfinance.",
    }


# ---------------------------------------------------------------------------
# Agent 2 — Analyst
# ---------------------------------------------------------------------------

def analyst_agent(structured_data: dict[str, Any]) -> dict[str, Any]:
    """Score and interpret the structured financial data using Gemini.

    This agent reasons about the data holistically — considering fundamentals,
    technicals, valuation, and momentum together — and produces a score,
    recommendation, and natural-language reasoning for each factor.

    Parameters
    ----------
    structured_data:
        Output of :func:`data_collector_agent`.

    Returns
    -------
    dict
        Analysis result with keys:

        ``score``            – Integer 0–100.
        ``recommendation``   – ``"BUY"``, ``"HOLD"``, or ``"SELL"``.
        ``confidence``       – ``"High"``, ``"Medium"``, or ``"Low"``.
        ``valuation_view``   – Gemini's view on valuation (1–2 sentences).
        ``growth_view``      – Gemini's view on growth trajectory.
        ``technical_view``   – Gemini's view on price momentum / technicals.
        ``risk_view``        – Gemini's key risk assessment.
        ``bull_case``        – Brief bull case (1–2 sentences).
        ``bear_case``        – Brief bear case (1–2 sentences).
        ``reasoning``        – Overall reasoning for the recommendation.

        Falls back to a neutral HOLD / 50 result when Gemini is unavailable.
    """
    if not _API_KEY:
        return _analyst_fallback()

    ticker = structured_data.get("ticker", "UNKNOWN")
    prompt = f"""
You are a senior equity research analyst at a top-tier investment bank.
Analyse the following structured financial data for {ticker} and produce
a rigorous investment assessment.

STRUCTURED DATA:
{json.dumps(structured_data, default=str, indent=2)}

Scoring guidelines (start at 50, adjust up/down):
- Valuation (P/E vs sector norms):   +10 cheap / -10 expensive
- Revenue growth (YoY %):            +15 if >15% / +8 if 5-15% / -10 if <0%
- Profitability (net margin):        +10 if >15% / +5 if 5-15% / -15 if <0%
- Balance sheet (D/E ratio):         +8 if <0.5 / -10 if >2.0
- Momentum (52-week position + SMAs): +10 if strong / -10 if weak
- Overall qualitative adjustment:    ±10 based on your holistic view

Return ONLY a valid JSON object with exactly these keys:

{{
  "score": integer 0-100,
  "recommendation": "BUY" | "HOLD" | "SELL",
  "confidence": "High" | "Medium" | "Low",
  "valuation_view": string,
  "growth_view": string,
  "technical_view": string,
  "risk_view": string,
  "bull_case": string,
  "bear_case": string,
  "reasoning": string
}}

Return only the JSON — no preamble, no explanation, no markdown fences.
"""

    try:
        model = _get_model()
        time.sleep(_INTER_AGENT_DELAY)
        response = model.generate_content(prompt)
        analysis = _extract_json(response.text)
        if analysis:
            logger.info("AnalystAgent: %s scored %s (%s)",
                        ticker, analysis.get("score"), analysis.get("recommendation"))
            return analysis
    except Exception as exc:
        logger.warning("AnalystAgent Gemini call failed: %s", exc)

    return _analyst_fallback()


def _analyst_fallback() -> dict[str, Any]:
    """Return a neutral analysis result when Gemini is unavailable."""
    return {
        "score":          50,
        "recommendation": "HOLD",
        "confidence":     "Low",
        "valuation_view": "Unable to assess — Gemini unavailable.",
        "growth_view":    "Unable to assess — Gemini unavailable.",
        "technical_view": "Unable to assess — Gemini unavailable.",
        "risk_view":      "Unable to assess — Gemini unavailable.",
        "bull_case":      "N/A",
        "bear_case":      "N/A",
        "reasoning":      "Analysis generated without Gemini; using neutral defaults.",
    }


# ---------------------------------------------------------------------------
# Agent 3 — Report Writer
# ---------------------------------------------------------------------------

def report_agent(
    ticker: str,
    structured_data: dict[str, Any],
    analysis: dict[str, Any],
    headlines: list[dict],
    sentiment: str,
) -> str:
    """Write a professional Markdown investment memo using Gemini.

    This agent synthesises all prior-stage outputs into a polished,
    human-readable research memo.  It uses Gemini to write natural-language
    prose rather than mechanically filling a template.

    Parameters
    ----------
    ticker:
        Normalised upper-case ticker symbol.
    structured_data:
        Output of :func:`data_collector_agent`.
    analysis:
        Output of :func:`analyst_agent`.
    headlines:
        List of headline dicts from ``news_tools.get_news_headlines``.
    sentiment:
        Aggregate sentiment string from ``news_tools.get_sentiment_summary``.

    Returns
    -------
    str
        Markdown-formatted investment research memo.  Falls back to a
        template-based memo built directly from the inputs when Gemini
        is unavailable.
    """
    if not _API_KEY:
        return _report_fallback(ticker, structured_data, analysis, headlines, sentiment)

    headline_text = "\n".join(
        f"- {h.get('title', '')}" for h in headlines[:5]
    ) or "- No live headlines available."

    prompt = f"""
You are a financial writer at a leading investment research firm.
Write a professional investment research memo for {ticker} using the
data and analysis provided below.

STRUCTURED DATA:
{json.dumps(structured_data, default=str, indent=2)}

ANALYST ASSESSMENT:
{json.dumps(analysis, default=str, indent=2)}

NEWS HEADLINES:
{headline_text}

NEWS SENTIMENT: {sentiment}

Format requirements:
- Use Markdown (headers with ###, bold with **, tables with |)
- Start with: ### INVESTMENT RESEARCH MEMO: {ticker}
- Include sections: EXECUTIVE SUMMARY, KEY METRICS (table), TECHNICAL ANALYSIS
  (table), ANALYST VIEWS, NEWS SENTIMENT, RECOMMENDATION, RISK FACTORS
- EXECUTIVE SUMMARY must be 3–4 sentences of clear prose.
- ANALYST VIEWS should include valuation, growth, technical, and risk views
  as short paragraphs (2–3 sentences each).
- RECOMMENDATION section must clearly state: Score, Recommendation, Confidence,
  Bull Case, Bear Case.
- End with a one-line disclaimer about educational purposes only.
- Do not add any text before the ### INVESTMENT RESEARCH MEMO header.
- Write in a professional, objective tone. No hype.
"""

    try:
        model = _get_model()
        time.sleep(_INTER_AGENT_DELAY)
        response = model.generate_content(prompt)
        memo = response.text.strip()
        if memo:
            logger.info("ReportAgent: memo written for %s (%d chars)", ticker, len(memo))
            return memo
    except Exception as exc:
        logger.warning("ReportAgent Gemini call failed: %s", exc)

    return _report_fallback(ticker, structured_data, analysis, headlines, sentiment)


def _report_fallback(
    ticker: str,
    data: dict,
    analysis: dict,
    headlines: list[dict],
    sentiment: str,
) -> str:
    """Build a template memo when Gemini is unavailable."""
    price    = f"${data.get('current_price', 0):.2f}"
    pe       = f"{data.get('pe_ratio', 'N/A')}"
    mc       = f"${data.get('market_cap_b', 'N/A')}B"
    growth   = f"{data.get('revenue_growth', 'N/A')}%"
    margin   = f"{data.get('net_margin', 'N/A')}%"
    score    = analysis.get("score", 50)
    rec      = analysis.get("recommendation", "HOLD")
    conf     = analysis.get("confidence", "Low")
    hl_lines = "\n".join(f"{i+1}. {h.get('title','')}" for i, h in enumerate(headlines[:3]))

    return f"""
### INVESTMENT RESEARCH MEMO: {ticker}
---

***EXECUTIVE SUMMARY***

{ticker} is currently trading at **{price}** with a trailing P/E of **{pe}**.
Revenue growth is **{growth}** and net margin is **{margin}**.
Market sentiment is **{sentiment}**. Recommendation: **{rec}**.

***KEY METRICS***

| Metric            | Value    |
|-------------------|----------|
| Current Price     | {price}  |
| P/E Ratio         | {pe}     |
| Market Cap        | {mc}     |
| Revenue Growth    | {growth} |
| Net Margin        | {margin} |

***NEWS SENTIMENT***

**Sentiment:** {sentiment}

{hl_lines}

***RECOMMENDATION***

- **Score:** {score} / 100
- **Recommendation:** {rec}
- **Confidence:** {conf}

*Disclaimer: Educational purposes only. Not investment advice.*
""".strip()