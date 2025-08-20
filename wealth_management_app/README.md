# Wealth Management Portfolio Optimizer 🧠📊

This is an AI-powered portfolio management application built with Python and Streamlit. It allows users to:

- Predict future stock returns using ML models (LSTM, Transformer)
- Incorporate macroeconomic & technical indicators
- Determine risk profile via LLM (Cohere via Hugging Face)
- Optimize portfolios using Modern Portfolio Theory
- Execute trades via Alpaca API
- Rebalance based on portfolio drift

---

## 🗂 Project Structure

wealth_management_app/
│
├── README.md
├── requirements.txt
│
└── src/
├── models/
│ ├── lstm.py # LSTM-based return predictor
│ ├── transformer.py # Transformer-based return predictor
│ └── portfolio_optimization.py # Portfolio optimizer with Sharpe/Sortino ratios
│
├── services/
│ ├── alpaca_trading.py # Alpaca trading and data interface
│ ├── data_fetcher.py # Stock data via yfinance
│ ├── macro_data_fetcher.py # Macroeconomic indicators from FRED
│ ├── huggingface_service.py # Risk profiling and portfolio explanations via LLM
│ └── execute_trades.py # Trade execution and rebalance logic
│
├── main.py # Core logic: prediction + optimization
└── app_streamlit.py # Streamlit UI app
