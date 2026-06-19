"""
stock_tools.py
--------------
Yahoo Finance data-access layer for AlphaInsight.

Role in the multi-agent pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module is the **only** source of raw numerical market data in the
pipeline.  Gemini agents do not fetch prices or financials — they receive
the structured output of these functions and reason about it.

  yfinance (this module)
       │
       ├── get_stock_info()       → price, P/E, market cap, sector …
       ├── get_financials()       → margins, growth, D/E …
       └── get_historical_data()
               │
               └── calculate_technical_indicators()  → SMA, volatility …
                           │
                           ▼
                   DataCollectorAgent (agents.py)

Data quality notes
~~~~~~~~~~~~~~~~~~
* ``yfinance`` uses the unofficial Yahoo Finance API.  Fields may be absent
  (``None``), zero, or ``NaN`` for thinly-traded securities or after market
  hours.  Every function defensively coerces missing values.

* ``get_historical_data`` drops rows where ``Close`` is NaN, which occurs
  when Yahoo returns a trailing empty row for a closed market.

* ``calculate_technical_indicators`` returns ``None`` for each indicator
  that requires more historical bars than are available (e.g. the 50-day
  SMA needs at least 50 trading days of data).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StockInfo        = dict[str, Any]
Financials       = dict[str, Any]
TechnicalIndicators = dict[str, Any]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> StockInfo:
    """Fetch current price and descriptive metadata for *ticker*.

    This is the first call in the pipeline.  The returned dict is passed
    directly to ``DataCollectorAgent`` alongside financials and technicals.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol (e.g. ``"AAPL"``, ``"RELIANCE.NS"``).
        Case-insensitive.

    Returns
    -------
    StockInfo
        Dictionary with the following keys (all values default to ``0`` or
        ``"Unknown"`` when absent from the API response):

        ``ticker``           – Normalised upper-case symbol.
        ``current_price``    – Latest trade price (USD or local currency).
        ``market_cap``       – Total market capitalisation.
        ``pe_ratio``         – Trailing twelve-month P/E ratio.
        ``forward_pe``       – Forward (next-twelve-month) P/E ratio.
        ``eps``              – Trailing twelve-month EPS.
        ``dividend_yield``   – Annual dividend yield expressed as a percent.
        ``target_price``     – Mean analyst 12-month price target.
        ``52_week_high``     – 52-week intraday high.
        ``52_week_low``      – 52-week intraday low.
        ``sector``           – GICS sector string.
        ``industry``         – GICS industry string.

    Examples
    --------
    >>> info = get_stock_info("AAPL")
    >>> info["ticker"]
    'AAPL'
    >>> isinstance(info["current_price"], float)
    True
    """
    ticker = ticker.upper()
    stock  = yf.Ticker(ticker)
    raw: dict[str, Any] = stock.info

    current_price = raw.get("currentPrice") or raw.get("regularMarketPrice") or 0.0
    raw_yield     = raw.get("dividendYield")
    dividend_yield = float(raw_yield) * 100 if raw_yield else 0.0

    return {
        "ticker":         ticker,
        "current_price":  float(current_price),
        "market_cap":     raw.get("marketCap", 0),
        "pe_ratio":       raw.get("trailingPE", 0),
        "forward_pe":     raw.get("forwardPE", 0),
        "eps":            raw.get("trailingEps", 0),
        "dividend_yield": round(dividend_yield, 4),
        "target_price":   raw.get("targetMeanPrice", 0),
        "52_week_high":   raw.get("fiftyTwoWeekHigh", 0),
        "52_week_low":    raw.get("fiftyTwoWeekLow", 0),
        "sector":         raw.get("sector", "Unknown"),
        "industry":       raw.get("industry", "Unknown"),
    }


def get_historical_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Download OHLCV price history for *ticker*.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.
    period:
        Duration string accepted by ``yfinance`` — one of
        ``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``, ``"2y"``, ``"5y"``,
        ``"max"``.  Defaults to ``"6mo"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
        Rows where ``Close`` is ``NaN`` are dropped.
        May be **empty** when the ticker is invalid or the API returns
        no data.

    Examples
    --------
    >>> hist = get_historical_data("MSFT", period="1mo")
    >>> "Close" in hist.columns
    True
    >>> hist["Close"].isna().any()
    False
    """
    stock = yf.Ticker(ticker.upper())
    hist: pd.DataFrame = stock.history(period=period)

    if hist.empty:
        logger.warning("No historical data returned for %s (%s)", ticker, period)
        return hist

    before  = len(hist)
    hist    = hist.dropna(subset=["Close"])
    dropped = before - len(hist)
    if dropped:
        logger.debug("Dropped %d NaN-Close rows for %s", dropped, ticker)

    return hist


def calculate_technical_indicators(hist: pd.DataFrame) -> TechnicalIndicators:
    """Derive price-based technical indicators from OHLCV history.

    Called locally before the data is passed to ``DataCollectorAgent``.
    The agent uses these numbers as inputs for its Gemini-powered
    enrichment and validation step.

    Parameters
    ----------
    hist:
        DataFrame as returned by :func:`get_historical_data`.  Must contain
        at least a ``Close`` column.  An empty DataFrame is accepted and
        returns a dict of all-``None`` / zero values.

    Returns
    -------
    TechnicalIndicators
        Dictionary with the following keys:

        ``current_price``             – Most recent closing price (float).
        ``sma_20``                    – 20-day SMA, or ``None`` if < 20 bars.
        ``sma_50``                    – 50-day SMA, or ``None`` if < 50 bars.
        ``above_sma_20``              – ``True`` when price > 20-day SMA.
        ``position_in_52_week_range`` – Price position in high-low range
                                        as a percentage (0=low, 100=high),
                                        or ``None`` when high == low.
        ``volatility_annual``         – Annualised close-to-close volatility
                                        as a percentage.
        ``price_change_1m``           – 22-bar (≈1 month) return (%),
                                        or ``None`` when < 22 bars.
        ``price_change_6m``           – Full-period return (%),
                                        or ``None`` when < 2 bars.

    Notes
    -----
    Volatility is annualised by multiplying daily standard deviation of
    simple returns by √252 (US trading days per year).

    Examples
    --------
    >>> hist = get_historical_data("AAPL", period="6mo")
    >>> tech = calculate_technical_indicators(hist)
    >>> 0 <= tech["position_in_52_week_range"] <= 100
    True
    """
    _empty: TechnicalIndicators = {
        "current_price":             0.0,
        "sma_20":                    None,
        "sma_50":                    None,
        "above_sma_20":              False,
        "position_in_52_week_range": None,
        "volatility_annual":         0.0,
        "price_change_1m":           None,
        "price_change_6m":           None,
    }

    if hist is None or hist.empty:
        return _empty

    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        return _empty

    n     = len(hist)
    close = hist["Close"]
    current_price = float(close.iloc[-1])

    sma_20_val = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else None
    sma_50_val = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else None
    above_sma_20 = (current_price > sma_20_val) if sma_20_val is not None else False

    high_52w = float(hist["High"].max())
    low_52w  = float(hist["Low"].min())
    position_52w: float | None = (
        (current_price - low_52w) / (high_52w - low_52w) * 100
        if high_52w != low_52w else None
    )

    log_returns    = close.pct_change().dropna()
    volatility_ann = float(log_returns.std() * (252 ** 0.5) * 100) if not log_returns.empty else 0.0

    price_change_1m: float | None = (
        (current_price / float(close.iloc[-22]) - 1) * 100 if n >= 22 else None
    )
    price_change_6m: float | None = (
        (current_price / float(close.iloc[0]) - 1) * 100 if n >= 2 else None
    )

    return {
        "current_price":             round(current_price, 2),
        "sma_20":                    round(sma_20_val, 2) if sma_20_val is not None else None,
        "sma_50":                    round(sma_50_val, 2) if sma_50_val is not None else None,
        "above_sma_20":              above_sma_20,
        "position_in_52_week_range": round(position_52w, 1) if position_52w is not None else None,
        "volatility_annual":         round(volatility_ann, 1),
        "price_change_1m":           round(price_change_1m, 1) if price_change_1m is not None else None,
        "price_change_6m":           round(price_change_6m, 1) if price_change_6m is not None else None,
    }


def get_financials(ticker: str) -> Financials:
    """Fetch annual income statement and balance sheet metrics for *ticker*.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.

    Returns
    -------
    Financials
        Dictionary with the following keys (all numeric, rounded):

        ``revenue``            – Most recent annual revenue (USD billions).
        ``revenue_growth_yoy`` – Year-over-year revenue growth (%).
        ``gross_margin``       – Gross profit margin (%).
        ``operating_margin``   – Operating profit margin (%).
        ``net_margin``         – Net profit margin (%).
        ``debt_to_equity``     – Total liabilities ÷ total equity.
        ``net_income``         – Most recent annual net income (USD billions).

        Returns ``{"error": str}`` when no income statement is available.

    Examples
    --------
    >>> fin = get_financials("AAPL")
    >>> fin["net_margin"] > 0
    True
    """
    stock  = yf.Ticker(ticker.upper())
    income = stock.income_stmt
    balance = stock.balance_sheet

    if income.empty:
        logger.warning("No income statement data for %s", ticker)
        return {"error": "No financial data available"}

    def _row(df: pd.DataFrame, label: str) -> float:
        """Return the most-recent value for *label* or 0.0 if absent/NaN."""
        if label in df.index:
            val = df.loc[label].iloc[0]
            return float(val) if pd.notna(val) else 0.0
        return 0.0

    revenue          = _row(income, "Total Revenue")
    gross_profit     = _row(income, "Gross Profit")
    operating_income = _row(income, "Operating Income")
    net_income       = _row(income, "Net Income")

    # YoY revenue growth
    if len(income.columns) >= 2 and "Total Revenue" in income.index:
        prev_revenue   = float(income.loc["Total Revenue"].iloc[1])
        revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else 0.0
    else:
        revenue_growth = 0.0

    def _margin(num: float, den: float) -> float:
        return (num / den * 100) if den else 0.0

    total_equity      = _row(balance, "Total Equity Gross Minority Interest")
    total_liabilities = _row(balance, "Total Liabilities Net Minority Interest")
    total_debt = _row(balance, "Total Debt")  
    debt_to_equity = (total_debt / total_equity) if total_equity else 0.0

    return {
        "revenue":            round(revenue / 1e9, 2),
        "revenue_growth_yoy": round(revenue_growth, 1),
        "gross_margin":       round(_margin(gross_profit, revenue), 1),
        "operating_margin":   round(_margin(operating_income, revenue), 1),
        "net_margin":         round(_margin(net_income, revenue), 1),
        "debt_to_equity":     round(debt_to_equity, 2),
        "net_income":         round(net_income / 1e9, 2),
    }