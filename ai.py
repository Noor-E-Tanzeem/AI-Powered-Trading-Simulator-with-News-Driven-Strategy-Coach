import streamlit as st
import yfinance as yf
import pandas as pd
import time

# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty:
                return None
            return data
        except:
            time.sleep(5)
    return None


def simple_sentiment(text):
    positive_words = ["gain", "growth", "profit", "high", "up", "strong", "optimistic"]
    negative_words = ["loss", "drop", "decline", "risk", "down", "weak", "regulatory"]

    text = text.lower()

    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("AI-Powered Trading Simulator")

symbol = st.text_input("Enter Stock Symbol", "AAPL").upper()
investment = st.number_input("Investment Amount ($)", min_value=100, value=1000)

if st.button("Analyze Stock"):
    st.info("Fetching stock data...")

    data = fetch_stock_data(symbol)

    if data is None:
        st.error("Failed to fetch stock data. Try again later.")
    else:
        st.success("Stock data loaded")
        st.line_chart(data["Close"])

        news = [
            f"{symbol} stock hits new quarterly high",
            f"Regulatory concerns affect {symbol}",
            f"Analysts optimistic about {symbol} growth"
        ]

        st.subheader("News Sentiment Analysis")

        sentiments = []
        for n in news:
            sentiment = simple_sentiment(n)
            sentiments.append(sentiment)
            st.write(f"📰 {n} → **{sentiment}**")

        st.subheader("Strategy Recommendation")

        if sentiments.count("Positive") > sentiments.count("Negative"):
            st.success(f"Recommendation: BUY {symbol}")
        elif sentiments.count("Negative") > sentiments.count("Positive"):
            st.error(f"Recommendation: SELL {symbol}")
        else:
            st.info("Recommendation: HOLD")
