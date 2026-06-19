"""
dashboard.py
------------
AlphaInsight Streamlit dashboard — entry point for the web UI.

Run with::

    streamlit run dashboard.py

Environment
~~~~~~~~~~~
Requires ``GOOGLE_API_KEY`` to be set for Gemini-powered analysis.
Without it the pipeline falls back to deterministic scoring.
Load from a ``.env`` file with::

    pip install python-dotenv

Then add at the very top of this file (before any other import)::

    from dotenv import load_dotenv; load_dotenv()

Architecture
~~~~~~~~~~~~
The dashboard is a single-file Streamlit application.  It calls
``run_financial_analysis`` from ``research_pipeline`` which orchestrates
three Gemini agents (DataCollector → Analyst → ReportWriter).
yfinance is called independently for the real-time price cards and chart.

The deterministic rule-based score from ``calculator_tools.evaluate_investment_score``
is computed inside ``research_pipeline.run_financial_analysis`` and returned
alongside the Gemini score, so the dashboard receives both scores in one call
without any duplicate yfinance fetches.

Layout
~~~~~~
┌─────────────────────────────────────────────────────────┐
│  Header  +  Gemini status badge                         │
├──────────┬──────────────────────────────────────────────┤
│ Sidebar  │  Ticker input + Run button                   │
│          │  ─────────────────────────────────────────── │
│          │  Price / P/E / Market-cap / Gemini-score     │
│          │  Candlestick + volume chart                  │
│          │  Tabs: Memo | Metrics | News | Technical     │
└──────────┴──────────────────────────────────────────────┘

Session state
~~~~~~~~~~~~~
``st.session_state.analysis_run``   – bool, controls result visibility.
``st.session_state.current_ticker`` – str, last analysed ticker.
"""

# ── Load .env before anything else ──────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set GOOGLE_API_KEY in shell instead

import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from research_pipeline import run_financial_analysis
from tools.news_tools import get_news_headlines, get_sentiment_summary
from tools.stock_tools import calculate_technical_indicators, get_historical_data

# ============================================================================
# PAGE CONFIGURATION — must be the first Streamlit call
# ============================================================================

st.set_page_config(
    page_title="AlphaInsight | AI Research Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: radial-gradient(circle at 20% 30%, #0f172a, #020617, #000000);
        background-attachment: fixed;
    }
    .main .block-container {
        padding-top: 1rem; padding-bottom: 2rem;
        max-width: 1400px; background: transparent;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
    }

    /* Header */
    .header-container {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(2,6,23,0.95));
        backdrop-filter: blur(10px);
        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid rgba(59,130,246,0.3);
        position: relative; overflow: hidden;
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    }
    .header-container::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #3b82f6, #06b6d4, #3b82f6, transparent);
    }
    .header-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #60a5fa 50%, #a78bfa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; letter-spacing: -0.5px;
    }
    .header-subtitle { font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem; }

    /* Gemini status badges */
    .gemini-active {
        background: rgba(99,102,241,0.15); color: #818cf8;
        border: 1px solid rgba(99,102,241,0.4);
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.7rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 0.3rem;
    }
    .gemini-inactive {
        background: rgba(100,116,139,0.15); color: #94a3b8;
        border: 1px solid rgba(100,116,139,0.3);
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.7rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 0.3rem;
    }

    /* Cards */
    .premium-card {
        background: linear-gradient(135deg, rgba(17,24,39,0.9), rgba(15,23,42,0.9));
        backdrop-filter: blur(5px); border: 1px solid rgba(30,41,59,0.8);
        border-radius: 16px; padding: 1.25rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .premium-card:hover {
        border-color: #3b82f6; transform: translateY(-2px);
        box-shadow: 0 20px 25px -12px rgba(0,0,0,0.5);
    }
    .premium-card::after {
        content: ''; position: absolute; bottom: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6, #3b82f6);
        transform: scaleX(0); transition: transform 0.3s ease;
    }
    .premium-card:hover::after { transform: scaleX(1); }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 0.5rem;
    }
    .metric-value-large { font-size: 1.75rem; font-weight: 700; color: #ffffff; line-height: 1.2; }

    /* Score comparison card */
    .score-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(15,23,42,0.9));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px; padding: 1.25rem;
    }

    /* Ticker display */
    .ticker-display {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(2,6,23,0.95));
        backdrop-filter: blur(10px); border: 1px solid rgba(30,41,59,0.8);
        border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }
    .ticker-display::before {
        content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
        background: radial-gradient(circle at 100% 0%, rgba(59,130,246,0.1), transparent);
        pointer-events: none;
    }
    .ticker-symbol {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #94a3b8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .ticker-name  { font-size: 0.9rem; color: #64748b; margin-top: 0.25rem; }
    .price-large  { font-size: 2.8rem; font-weight: 700; color: #ffffff; }
    .metric-change-positive { color: #10b981; font-size: 0.8rem; font-weight: 500; margin-left: 0.5rem; }
    .metric-change-negative { color: #ef4444; font-size: 0.8rem; font-weight: 500; margin-left: 0.5rem; }

    /* News cards */
    .news-card-premium {
        background: rgba(17,24,39,0.8); backdrop-filter: blur(5px);
        border: 1px solid rgba(30,41,59,0.6); border-radius: 12px;
        padding: 1rem; margin: 0.75rem 0; transition: all 0.2s ease;
    }
    .news-card-premium:hover {
        border-color: #3b82f6; transform: translateX(4px);
        background: rgba(17,24,39,0.95);
    }
    .news-card-placeholder { border-color: rgba(245,158,11,0.3) !important; }
    .news-title-premium { font-weight: 600; font-size: 0.9rem; color: #f1f5f9; margin-bottom: 0.5rem; }
    .news-meta-premium  { font-size: 0.7rem; color: #64748b; }

    /* Sentiment */
    .sentiment-badge {
        padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem;
        font-weight: 600; display: inline-flex; align-items: center; gap: 0.25rem;
    }
    .sentiment-positive { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .sentiment-negative { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
    .sentiment-neutral  { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
    .sentiment-gemini   { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white; border: none; font-weight: 600; font-size: 0.85rem;
        padding: 0.6rem 1.2rem; border-radius: 40px; transition: all 0.2s;
        width: 100%; letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px -10px rgba(59,130,246,0.5);
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem; border-bottom: 1px solid rgba(30,41,59,0.5);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 0.85rem; color: #94a3b8;
        padding: 0.75rem 0; background: transparent;
    }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom-color: #3b82f6; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(2,6,23,0.98));
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(30,41,59,0.5);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0; }

    /* Welcome */
    .welcome-message {
        text-align: center; padding: 3rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(2,6,23,0.8));
        backdrop-filter: blur(10px); border-radius: 20px;
        border: 1px solid rgba(30,41,59,0.5); margin: 2rem 0;
    }

    /* Misc */
    .stTextInput input {
        background: rgba(15,23,42,0.9); border: 1px solid rgba(30,41,59,0.6);
        border-radius: 40px; color: #ffffff; padding: 0.6rem 1rem;
    }
    .stTextInput input:focus {
        border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
    }
    hr { border-color: rgba(30,41,59,0.5); margin: 1.5rem 0; }
    code {
        background: rgba(30,41,59,0.6); padding: 0.2rem 0.4rem;
        border-radius: 6px; color: #e2e8f0;
    }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPERS
# ============================================================================

_GEMINI_ACTIVE = bool(os.environ.get("GOOGLE_API_KEY", ""))


def create_price_chart(ticker: str):
    """Build a Plotly candlestick + volume chart for the past 12 months.

    Parameters
    ----------
    ticker:
        Exchange ticker symbol.

    Returns
    -------
    plotly.graph_objects.Figure | None
        Configured figure, or ``None`` when data fetch fails.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            name="Price",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ))
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"],
            name="Volume", marker_color="#3b82f6",
            opacity=0.3, yaxis="y2",
        ))
        fig.update_layout(
            template="plotly_dark",
            title=f"{ticker} — 1-Year Price Action",
            title_font_color="#f1f5f9",
            paper_bgcolor="rgba(15,23,42,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            xaxis_title="Date", xaxis_title_font_color="#94a3b8",
            yaxis_title="Price (USD)", yaxis_title_font_color="#94a3b8",
            yaxis2=dict(
                title="Volume", overlaying="y", side="right",
                showgrid=False, title_font_color="#94a3b8",
            ),
            height=400, margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
            ),
            xaxis=dict(gridcolor="rgba(30,41,59,0.3)", linecolor="rgba(30,41,59,0.5)"),
            yaxis=dict(gridcolor="rgba(30,41,59,0.3)", linecolor="rgba(30,41,59,0.5)"),
        )
        return fig
    except Exception:
        return None


def add_disclaimer(memo_text: str) -> str:
    """Prepend a styled disclaimer block to the Markdown memo.

    Parameters
    ----------
    memo_text:
        Raw Markdown string from the pipeline.

    Returns
    -------
    str
        HTML disclaimer + original memo text.
    """
    engine = "Gemini 2.5 Flash + Yahoo Finance" if _GEMINI_ACTIVE else "Rule-based scoring + Yahoo Finance"
    disclaimer = f"""
<div style="background:rgba(59,130,246,0.08); border-left:3px solid #3b82f6;
            padding:1rem; border-radius:12px; margin:1rem 0;">
    <strong style="color:#fbbf24;">⚠️ Disclaimer</strong><br>
    <span style="font-size:0.7rem; color:#94a3b8;">
        For <strong>educational purposes only</strong>.
        Not investment advice. Analysis engine: <em>{engine}</em>.
    </span>
    <span style="font-size:0.65rem; color:#64748b; display:block; margin-top:0.5rem;">
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
    </span>
</div>
"""
    return disclaimer + "\n\n" + memo_text



# ============================================================================
# HEADER
# ============================================================================

gemini_badge = (
    '<span class="gemini-active">✦ Gemini 2.5 Flash Active</span>'
    if _GEMINI_ACTIVE else
    '<span class="gemini-inactive">○ Gemini Offline — rule-based fallback</span>'
)

st.markdown(f"""
<div class="header-container">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div class="header-title">AlphaInsight</div>
            <div class="header-subtitle">
                Multi-Agent Research Platform &nbsp;|&nbsp;
                Gemini 2.5 Flash + Yahoo Finance
            </div>
        </div>
        <div style="margin-top:0.5rem;">{gemini_badge}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### Multi-Agent Pipeline")
    st.markdown("---")

    agent_color = "#818cf8" if _GEMINI_ACTIVE else "#64748b"
    st.markdown(f"""
    <div style="margin:0.5rem 0;">
        <span style="color:#10b981;">●</span> <strong>yfinance</strong><br>
        <span style="font-size:0.7rem;color:#64748b;margin-left:1rem;">
            Price, financials, OHLCV history
        </span>
    </div>
    <div style="margin:0.5rem 0;">
        <span style="color:{agent_color};">●</span>
        <strong>Agent 1 · Data Collector</strong><br>
        <span style="font-size:0.7rem;color:#64748b;margin-left:1rem;">
            Gemini validates &amp; structures raw data
        </span>
    </div>
    <div style="margin:0.5rem 0;">
        <span style="color:{agent_color};">●</span>
        <strong>Agent 2 · Analyst</strong><br>
        <span style="font-size:0.7rem;color:#64748b;margin-left:1rem;">
            Gemini scores &amp; interprets holistically
        </span>
    </div>
    <div style="margin:0.5rem 0;">
        <span style="color:{agent_color};">●</span>
        <strong>Agent 3 · Report Writer</strong><br>
        <span style="font-size:0.7rem;color:#64748b;margin-left:1rem;">
            Gemini writes the investment memo
        </span>
    </div>
    <div style="margin:0.5rem 0;">
        <span style="color:#ec489a;">●</span>
        <strong>Quant Cross-Check</strong><br>
        <span style="font-size:0.7rem;color:#64748b;margin-left:1rem;">
            Deterministic 5-factor score (local)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Recommendation Bands")
    st.markdown("""
    <div style="font-size:0.75rem;color:#cbd5e1;line-height:1.8;">
        <span style="color:#10b981;">●</span> <b>BUY</b>  → Score ≥ 70<br>
        <span style="color:#f59e0b;">●</span> <b>HOLD</b> → Score 45–69<br>
        <span style="color:#ef4444;">●</span> <b>SELL</b> → Score &lt; 45
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Market Coverage")
    st.markdown("""
    <div style="font-size:0.75rem;color:#cbd5e1;">
        • US Equities (NYSE / NASDAQ)<br>
        • Indian Equities (NSE / BSE)<br>
        • Global Indices &amp; ETFs<br>
        • Cryptocurrencies
    </div>
    """, unsafe_allow_html=True)

    if not _GEMINI_ACTIVE:
        st.markdown("---")
        st.warning(
            "**Gemini is offline.**\n\n"
            "Set `GOOGLE_API_KEY` in your `.env` file to enable "
            "AI-powered analysis.",
            icon="🔑",
        )

    st.markdown("---")
    if st.button("Clear Results", use_container_width=True):
        st.session_state.analysis_run = False
        st.rerun()
    st.markdown("---")


# ============================================================================
# TICKER INPUT
# ============================================================================

col_ticker, col_button = st.columns([4, 1])
with col_ticker:
    ticker = st.text_input(
        "Ticker",
        value="",
        placeholder="AAPL, MSFT, RELIANCE.NS, BTC-USD",
        label_visibility="collapsed",
    ).upper().strip()

with col_button:
    analyze = st.button("**Run Analysis**", use_container_width=True, type="primary")

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

if analyze and ticker:
    st.session_state.analysis_run = True
    st.session_state.current_ticker = ticker


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.analysis_run:
    st.markdown("""
    <div class="welcome-message">
        <h2 style="color:#ffffff; margin-top:1rem;">Ready for Analysis</h2>
        <p style="color:#94a3b8; max-width:500px; margin:1rem auto;">
            Enter a ticker symbol above and click <strong>Run Analysis</strong>
            to trigger the three-agent Gemini pipeline.
        </p>
        <div style="display:flex; justify-content:center; gap:0.75rem;
                    margin-top:1.5rem; flex-wrap:wrap;">
            <code>AAPL</code><code>MSFT</code><code>NVDA</code>
            <code>TSLA</code><code>GOOGL</code><code>RELIANCE.NS</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.analysis_run and ticker:

    # Progress bar — one step per pipeline stage so users see real progress
    progress = st.progress(0, text="Fetching market data…")

    try:
        progress.progress(10, text="Agent 1 · Data Collector running…")
        # run_financial_analysis returns a dict with memo + all scores
        result   = run_financial_analysis(ticker)
        memo_raw = result["memo"]
        progress.progress(75, text="Agent 2 · Analyst scoring…")

        # Independent yfinance pull for display cards (does not re-run agents)
        info = yf.Ticker(ticker).info
        progress.progress(95, text="Agent 3 · Report Writer finishing…")
        progress.progress(100, text="✅ Done")
        progress.empty()

        st.success(f"✅ Analysis complete — {ticker}")

        # Scores already computed by the pipeline — no extra yfinance calls needed
        g_score = result["score"]
        g_rec   = result["recommendation"]
        g_conf  = result["confidence"]
        q_score = result["quant_score"]
        q_rec   = result["quant_rec"]
        q_conf  = result["quant_conf"]

        rec_colors = {"BUY": "#10b981", "HOLD": "#f59e0b", "SELL": "#ef4444"}

        # ── Header cards row ─────────────────────────────────────────────
        # col1: ticker + price  col2: P/E  col3: market cap  col4: Gemini score
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            price  = info.get("currentPrice", info.get("regularMarketPrice", 0))
            change = info.get("regularMarketChangePercent", 0)
            css    = "metric-change-positive" if change >= 0 else "metric-change-negative"
            arrow  = "▲" if change >= 0 else "▼"
            st.markdown(f"""
            <div class="ticker-display">
                <div class="ticker-symbol">{ticker}</div>
                <div class="ticker-name">{info.get('longName', '')[:50]}</div>
                <div style="margin-top:1rem;">
                    <span class="price-large">${price:.2f}</span>
                    <span class="{css}">{arrow} {abs(change):.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            pe = info.get("trailingPE", "—")
            st.markdown(f"""
            <div class="premium-card">
                <div class="metric-label">P/E Ratio</div>
                <div class="metric-value-large">{pe if pe != '—' else '—'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            mc   = info.get("marketCap", 0)
            mc_b = mc / 1e9 if mc else 0
            st.markdown(f"""
            <div class="premium-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value-large">${mc_b:.1f}B</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            # PRIMARY card — shows Gemini's score (the authoritative AI score)
            g_color = rec_colors.get(g_rec, "#94a3b8")
            st.markdown(f"""
            <div class="score-card">
                <div class="metric-label">✦ Gemini Score</div>
                <div class="metric-value-large" style="color:{g_color};">
                    {g_score}<span style="font-size:1rem;color:#64748b;">/100</span>
                </div>
                <div style="font-size:0.75rem;color:{g_color};font-weight:600;margin-top:0.25rem;">
                    {g_rec}
                    <span style="color:#64748b;font-weight:400;"> · {g_conf} confidence</span>
                </div>
                <div style="font-size:0.65rem;color:#475569;margin-top:0.5rem;">
                    Rule-based: {q_score}/100 {q_rec}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────────────────────────
        fig = create_price_chart(ticker)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_memo, tab_metrics, tab_news, tab_tech = st.tabs([
            "**Investment Memo**",
            "**Financial Analysis**",
            "**News Intelligence**",
            "**Technical View**",
        ])

        # ── Tab 1: Investment Memo (Gemini-written) ───────────────────────
        with tab_memo:
            if _GEMINI_ACTIVE:
                st.markdown(
                    '<span class="gemini-active">✦ Written by Gemini 2.5 Flash</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(add_disclaimer(memo_raw), unsafe_allow_html=True)
            st.download_button(
                "Export Research Report",
                data=memo_raw,
                file_name=f"{ticker}_AlphaInsight_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            )

        # ── Tab 2: Financial Metrics ──────────────────────────────────────
        with tab_metrics:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Valuation")
                val_df = pd.DataFrame({
                    "Metric": ["P/E", "Forward P/E", "PEG", "P/B", "P/S"],
                    "Value": [
                        info.get("trailingPE", "—"),
                        info.get("forwardPE", "—"),
                        info.get("pegRatio", "—"),
                        info.get("priceToBook", "—"),
                        info.get("priceToSalesTrailing12Months", "—"),
                    ],
                })
                st.dataframe(val_df, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("#### Profitability")

                def _pct(key: str) -> str:
                    v = info.get(key)
                    return f"{v * 100:.1f}%" if v else "—"

                prof_df = pd.DataFrame({
                    "Metric": ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"],
                    "Value": [
                        _pct("grossMargins"), _pct("operatingMargins"),
                        _pct("profitMargins"), _pct("returnOnEquity"), _pct("returnOnAssets"),
                    ],
                })
                st.dataframe(prof_df, use_container_width=True, hide_index=True)

            # Score comparison — now uses real values from the pipeline dict
            st.markdown("#### Score Comparison")
            st.markdown("""
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:0.5rem;">
                Gemini reasons holistically; the rule-based model uses fixed thresholds.
                Large divergences (e.g. BUY vs SELL) are worth reading the memo carefully.
            </div>
            """, unsafe_allow_html=True)

            score_df = pd.DataFrame({
                "Model":          ["✦ Gemini 2.5 Flash", "Rule-Based (quant)"],
                "Score":          [f"{g_score} / 100",   f"{q_score} / 100"],
                "Recommendation": [g_rec,                 q_rec],
                "Confidence":     [g_conf,                q_conf],
            })
            st.dataframe(score_df, use_container_width=True, hide_index=True)

        # ── Tab 3: News Intelligence ──────────────────────────────────────
        with tab_news:
            headlines = get_news_headlines(ticker, limit=10)
            sentiment = get_sentiment_summary(headlines)

            has_placeholders = any(h.get("placeholder") for h in headlines)
            if has_placeholders:
                st.warning(
                    "⚠️ Live news feed unavailable — showing placeholder headlines. "
                    "Sentiment analysis has been skipped for placeholder items.",
                    icon="📡",
                )

            sent_map = {
                "Positive": ("sentiment-positive", "📈"),
                "Negative": ("sentiment-negative", "📉"),
                "Neutral":  ("sentiment-neutral",  "📊"),
            }
            sent_class, sent_icon = sent_map.get(sentiment, ("sentiment-neutral", "📊"))
            engine_label = "Gemini" if _GEMINI_ACTIVE else "Keyword heuristic"

            st.markdown(f"""
            <div style="text-align:center; margin:1rem 0;">
                <span class="sentiment-badge {sent_class}">
                    {sent_icon} Aggregate Sentiment: {sentiment.upper()}
                </span>
                &nbsp;
                <span class="sentiment-badge sentiment-gemini">
                    ✦ {engine_label}
                </span>
            </div>
            """, unsafe_allow_html=True)

            for h in headlines:
                placeholder_cls = "news-card-placeholder" if h.get("placeholder") else ""
                placeholder_label = " <em style='color:#f59e0b;font-size:0.65rem;'>[placeholder]</em>" if h.get("placeholder") else ""
                st.markdown(f"""
                <div class="news-card-premium {placeholder_cls}">
                    <div class="news-title-premium">{h['title']}{placeholder_label}</div>
                    <div class="news-meta-premium">📅 {h['date']} &nbsp;|&nbsp; 📰 {h['source']}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Tab 4: Technical View ─────────────────────────────────────────
        with tab_tech:
            try:
                hist = get_historical_data(ticker, period="6mo")
                if not hist.empty:
                    tech = calculate_technical_indicators(hist)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Price Levels")
                        price_df = pd.DataFrame({
                            "Indicator": ["Current", "20-Day SMA", "50-Day SMA", "52W Position"],
                            "Value": [
                                f"${tech.get('current_price', '—')}",
                                f"${tech.get('sma_20', '—')}",
                                f"${tech.get('sma_50', '—')}" if tech.get("sma_50") else "—",
                                f"{tech.get('position_in_52_week_range', '—')}%",
                            ],
                        })
                        st.dataframe(price_df, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### Risk Metrics")
                        risk_df = pd.DataFrame({
                            "Indicator": ["Volatility (Ann.)", "1M Change", "6M Change", "Above 20-SMA"],
                            "Value": [
                                f"{tech.get('volatility_annual', '—')}%",
                                f"{tech.get('price_change_1m', '—')}%" if tech.get("price_change_1m") else "—",
                                f"{tech.get('price_change_6m', '—')}%",
                                "✅" if tech.get("above_sma_20") else "❌",
                            ],
                        })
                        st.dataframe(risk_df, use_container_width=True, hide_index=True)

                    pos = tech.get("position_in_52_week_range") or 50
                    if pos > 70:
                        st.success(f"**Technical Outlook:** Strong momentum — {pos:.0f}% of 52-week range")
                    elif pos < 30:
                        st.warning(f"**Technical Outlook:** Weak momentum — {pos:.0f}% of 52-week range")
                    else:
                        st.info(f"**Technical Outlook:** Neutral — {pos:.0f}% of 52-week range")

            except Exception:
                st.info("Technical indicators are included in the Investment Memo tab.")

    except Exception as exc:
        progress.empty()
        st.error(f"Analysis failed: {exc}")
        st.markdown("""
        **Troubleshooting:**
        1. Check your internet connection (Yahoo Finance API required).
        2. Verify the ticker symbol is valid (e.g. `AAPL`, `RELIANCE.NS`).
        3. If Gemini is active, check your `GOOGLE_API_KEY` is valid.
        4. Try again — Yahoo Finance occasionally rate-limits requests.
        """)
        if st.button("Try Again"):
            st.session_state.analysis_run = False
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
engine = "Gemini 2.5 Flash + Yahoo Finance" if _GEMINI_ACTIVE else "Rule-based + Yahoo Finance"
st.markdown(f"""
<div style="text-align:center; color:#475569; font-size:0.65rem; padding:1rem;">
    <span style="color:#64748b;">AlphaInsight</span> &nbsp;·&nbsp; Engine: {engine}<br>
    <span style="color:#334155;">For educational purposes only — not investment advice</span>
</div>
""", unsafe_allow_html=True)