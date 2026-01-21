import streamlit as st
import yfinance as yf
import pandas as pd
import time
from textblob import TextBlob

# -----------------------------
# Helper functions
# -----------------------------

def fetch_stock_data(symbol):
    """
    Fetch stock data from yfinance safely with rate-limit handling.
    Returns a DataFrame.
    """
    max_attempts = 5
    wait_seconds = 5

    for attempt in range(max_attempts):
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="3mo")
            if data.empty:
                raise ValueError("No data found for this symbol.")
            return data
        except yf.shared._exceptions.YFRateLimitError:
            st.warning(f"Rate limited by Yahoo Finance. Waiting {wait_seconds} seconds...")
            time.sleep(wait_seconds)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    st.error("Failed to fetch stock data after multiple attempts.")
    return None

def analyze_sentiment(text):
    """
    Simple sentiment analysis using TextBlob
    Returns 'Positive', 'Negative', or 'Neutral'
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("News-Driven Strategy Coach")

st.sidebar.header("Trading Controls")
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()
investment = st.sidebar.number_input("Investment Amount ($)", min_value=100, value=1000, step=100)

if st.sidebar.button("Analyze Stock"):

    # Fetch stock data
    st.info("Fetching stock data...")
    data = fetch_stock_data(symbol)
    if data is not None:
        st.success("Stock data fetched successfully!")
        st.line_chart(data['Close'])

        # Fake news headlines for demo
        news_headlines = [
            f"{symbol} stock hits new 3-month high!",
            f"{symbol} faces regulatory challenges.",
            f"Analysts are optimistic about {symbol}'s future growth."
        ]

        st.subheader("News Headlines & Sentiment")
        sentiments = []
        for news in news_headlines:
            sentiment = analyze_sentiment(news)
            sentiments.append(sentiment)
            st.write(f"**News:** {news} | **Sentiment:** {sentiment}")

        # Simple strategy based on sentiment
        positive_count = sentiments.count("Positive")
        negative_count = sentiments.count("Negative")

        st.subheader("Strategy Recommendation")
        if positive_count > negative_count:
            st.success(f"Recommendation: Consider BUYING {symbol} for ${investment}")
        elif negative_count > positive_count:
            st.error(f"Recommendation: Consider SELLING {symbol}")
        else:
            st.info("Recommendation: HOLD / No strong signal")
