import streamlit as st
import yfinance as yf
import pandas as pd
import time

# -----------------------------
# Helper functions
# -----------------------------

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    max_attempts = 3
    for _ in range(max_attempts):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty:
                return None
            return data
        except Exception:
            time.sleep(5)
    return None


def simple_sentiment(news):
    positive_words = ["growth", "profit", "gain", "high", "strong", "optimistic"]
    negative_words = ["loss", "decline", "drop", "risk", "regulatory", "weak"]

    score = 0
    text = news.lower()

    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("AI-Powered Trading Simulator")

symbol = st.text_input("Stock Symbol", "AAPL").upper()
investment = st.number_input("Investment Amount ($)", 100, 10000, 1000)

if st.button("Analyze Stock"):
    st.info("Fetching stock data...")

    data = fetch_stock_data(symbol)

    if data is None:
        st.error("Could not fetch stock data.")
    else:
        st.success("Stock data loaded")
        st.line_chart(data["Close"])

        news_samples = [
            f"{symbol} reports strong quarterly growth",
            f"{symbol} faces regulatory risk",
            f"Analysts remain optimistic about {symbol}"
        ]

        sentiments = []

        st.subheader("News Analysis")
        for news in news_samples:
            sentiment = simple_sentiment(news)
            sentiments.append(sentiment)
            st.write(f"📰 {news} → **{sentiment}**")

        st.subheader("Strategy Recommendation")

        if sentiments.count("Positive") > sentiments.count("Negative"):
            st.success(f"BUY signal for {symbol}")
        elif sentiments.count("Negative") > sentiments.count("Positive"):
            st.error(f"SELL signal for {symbol}")
        else:
            st.info("HOLD")
