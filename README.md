# AlphaInsight

### AI-Powered Multi-Agent Financial Research Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Overview

AlphaInsight is a **production-grade financial analysis platform** that combines:

- **Real-time market data** from Yahoo Finance
- **Multi-agent AI pipeline** using Google's Gemini 2.5 Flash
- **Deterministic cross-validation** with rule-based scoring
- **Beautiful interactive dashboard** built with Streamlit

The platform orchestrates three specialized AI agents:

1. **Data Collector** → Validates and structures raw financial data
2. **Analyst** → Performs holistic investment scoring (0-100)
3. **Report Writer** → Generates comprehensive investment memos

---

## ✨ Features

### Core Capabilities
- 🔍 **Universal Asset Coverage**: US Equities, Indian Equities (NSE/BSE), ETFs, Cryptocurrencies, Global Indices
- 🤖 **AI-Powered Analysis**: Gemini 2.5 Flash with financial reasoning
- 📊 **Dual Scoring System**: AI Score + Rule-Based Cross-Validation
- 📈 **Interactive Visualizations**: Candlestick charts with volume overlay
- 📰 **News Intelligence**: Real-time headlines with sentiment analysis
- 📄 **Professional Reports**: Markdown-formatted investment memos with export

### Dashboard Modules
| Tab | Functionality |
|-----|---------------|
| 📄 **Investment Memo** | Full narrative report written by Gemini |
| 📊 **Financial Analysis** | Valuation, profitability metrics, score comparison |
| 📰 **News Intelligence** | Live headlines with sentiment aggregation |
| 📈 **Technical View** | Price levels, risk metrics, momentum indicators |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit, Plotly, CSS3 |
| **AI Engine** | Google Gemini 2.5 Flash API |
| **Data Provider** | Yahoo Finance (yfinance) |
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Environment** | python-dotenv |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/AlphaInsight.git
cd AlphaInsight
