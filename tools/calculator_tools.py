"""
calculator_tools.py
-------------------
Fundamental and quantitative financial calculation utilities for AlphaInsight.

Role in the multi-agent pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the Gemini-powered pipeline, ``evaluate_investment_score`` is used as a
**deterministic cross-check** alongside the Gemini AnalystAgent score.
The dashboard surfaces both numbers so users can see where the rule-based
model and the LLM reasoning agree or diverge.

The three ratio helpers (``calculate_pe_ratio``, ``calculate_peg_ratio``,
``calculate_debt_to_equity``) remain useful for pre-processing raw numbers
before passing them to agents.

Scoring methodology (evaluate_investment_score)
------------------------------------------------
Five factors feed a base score of 50 that is clamped to [0, 100]:

  Factor              Max delta  Direction
  ──────────────────  ─────────  ─────────
  P/E ratio             ±15      Lower → better (value signal)
  Revenue growth        ±20      Higher → better (growth signal)
  Net margin            ±15      Higher → better (quality signal)
  Debt-to-equity        ±15      Lower → better (safety signal)
  52-week position      ±15      Higher → better (momentum signal)

Thresholds are calibrated to broad US equity market norms (2020-2024).
They will not translate well to high-growth sectors (e.g. biotech, early SaaS)
where negative earnings or high D/E ratios are structurally expected.

Recommendation bands
--------------------
  Score ≥ 70  →  BUY  (High confidence)
  Score 45-69 →  HOLD (Medium confidence)
  Score < 45  →  SELL (High confidence)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Simple ratio helpers
# ---------------------------------------------------------------------------

def calculate_pe_ratio(price: float, eps: float) -> float:
    """Return the trailing price-to-earnings (P/E) ratio.

    Parameters
    ----------
    price:
        Current market price per share (USD or local currency).
    eps:
        Trailing twelve-month earnings per share in the same currency.

    Returns
    -------
    float
        P/E ratio rounded to two decimal places, or ``0.0`` when *eps*
        is zero (avoids division by zero; callers should treat 0 as N/A).

    Examples
    --------
    >>> calculate_pe_ratio(150.0, 6.0)
    25.0
    >>> calculate_pe_ratio(100.0, 0.0)
    0.0
    """
    if eps == 0:
        return 0.0
    return round(price / eps, 2)


def calculate_peg_ratio(pe_ratio: float, growth_rate: float) -> float:
    """Return the Price/Earnings-to-Growth (PEG) ratio.

    A PEG below 1.0 is conventionally considered undervalued relative to
    growth; above 2.0 is considered expensive.

    Parameters
    ----------
    pe_ratio:
        Trailing or forward P/E ratio (dimensionless).
    growth_rate:
        Expected annual EPS growth rate as a **percentage**
        (e.g. pass ``15`` for 15 %).

    Returns
    -------
    float
        PEG ratio rounded to two decimal places, or ``0.0`` when
        *growth_rate* is zero.

    Examples
    --------
    >>> calculate_peg_ratio(25.0, 20.0)
    1.25
    """
    if growth_rate == 0:
        return 0.0
    return round(pe_ratio / growth_rate, 2)


def calculate_debt_to_equity(total_debt: float, total_equity: float) -> float:
    """Return the debt-to-equity (D/E) ratio.

    Parameters
    ----------
    total_debt:
        Total financial debt (short-term + long-term) from the balance sheet.
        Units must match *total_equity*.
    total_equity:
        Total shareholders' equity from the balance sheet.

    Returns
    -------
    float
        D/E ratio rounded to two decimal places, or ``0.0`` when
        *total_equity* is zero.

    Examples
    --------
    >>> calculate_debt_to_equity(500_000, 1_000_000)
    0.5
    """
    if total_equity == 0:
        return 0.0
    return round(total_debt / total_equity, 2)


# ---------------------------------------------------------------------------
# Composite scoring — used as deterministic cross-check alongside Gemini
# ---------------------------------------------------------------------------

def evaluate_investment_score(
    pe_ratio: float,
    revenue_growth: float,
    net_margin: float,
    debt_to_equity: float,
    price_position_52w: float,
) -> dict[str, Any]:
    """Compute a deterministic 0-100 investment score and recommendation.

    This function implements a transparent, rule-based scoring model that
    runs in parallel with the Gemini AnalystAgent.  The dashboard displays
    both scores so users can identify where the LLM and the heuristic
    model agree or diverge — a useful sanity-check layer.

    Parameters
    ----------
    pe_ratio:
        Trailing P/E ratio.  Negative values (loss-making company) subtract
        5 points.  Pass ``0`` when unavailable to apply the penalty.
    revenue_growth:
        Year-over-year revenue growth as a **percentage**
        (e.g. ``12.5`` for 12.5 %).
    net_margin:
        Net profit margin as a **percentage** (e.g. ``18.0`` for 18 %).
    debt_to_equity:
        Debt-to-equity ratio (dimensionless).
    price_position_52w:
        Current price as a percentage of the 52-week high-low range,
        where 0 % = 52-week low and 100 % = 52-week high.

    Returns
    -------
    dict
        Mapping with three keys:

        ``"score"`` : int
            Integer from 0 to 100 (inclusive).
        ``"recommendation"`` : str
            One of ``"BUY"``, ``"HOLD"``, or ``"SELL"``.
        ``"confidence"`` : str
            ``"High"`` for BUY / SELL signals; ``"Medium"`` for HOLD.

    Examples
    --------
    >>> result = evaluate_investment_score(
    ...     pe_ratio=18,
    ...     revenue_growth=12,
    ...     net_margin=20,
    ...     debt_to_equity=0.3,
    ...     price_position_52w=75,
    ... )
    >>> result["recommendation"]
    'BUY'
    """
    score = 50  # Neutral baseline

    # ── Factor 1: P/E ratio (value signal) ─────────────────────────────
    if pe_ratio > 0:
        if pe_ratio < 15:
            score += 15   # Cheap relative to earnings
        elif pe_ratio < 25:
            score += 5    # Fair value
        elif pe_ratio > 40:
            score -= 10   # Expensive; priced for perfection
    else:
        score -= 5        # Negative / missing earnings

    # ── Factor 2: Revenue growth (growth signal) ─────────────────────────
    if revenue_growth > 15:
        score += 20       # High growth
    elif revenue_growth > 5:
        score += 10       # Moderate growth
    elif revenue_growth < 0:
        score -= 15       # Shrinking top line

    # ── Factor 3: Net margin (profitability / quality signal) ────────────
    if net_margin > 15:
        score += 10       # High-quality earnings
    elif net_margin > 5:
        score += 5        # Acceptable profitability
    elif net_margin < 0:
        score -= 15       # Loss-making

    # ── Factor 4: Debt-to-equity (balance-sheet safety signal) ──────────
    if debt_to_equity < 0.5:
        score += 10       # Conservative balance sheet
    elif debt_to_equity > 2:
        score -= 15       # Highly leveraged
    elif debt_to_equity > 1:
        score -= 5        # Moderately leveraged

    # ── Factor 5: 52-week position (momentum / trend signal) ─────────────
    if price_position_52w > 80:
        score += 10       # Strong uptrend
    elif price_position_52w < 20:
        score -= 15       # Near 52-week lows

    # Clamp to valid range
    score = max(0, min(100, score))

    if score >= 70:
        recommendation, confidence = "BUY", "High"
    elif score >= 45:
        recommendation, confidence = "HOLD", "Medium"
    else:
        recommendation, confidence = "SELL", "High"

    return {
        "score": score,
        "recommendation": recommendation,
        "confidence": confidence,
    }