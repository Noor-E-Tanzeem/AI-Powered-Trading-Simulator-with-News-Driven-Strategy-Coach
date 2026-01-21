import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline
import requests

st.set_page_config(page_title="AI Trading Simulator", layout="wide")

st.title("📈 AI-Powered Trading Simulator with News-Driven Strategy Coach")

# Load sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_model()

# Sidebar
st.sidebar.header("Trading Controls")

stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
investment_amount = st.sidebar.number_input("Investment Amount ($)", 1000)

# Fetch stock data
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo")
    return hist

# Fetch news
def get_stock_news(symbol):
    API_KEY = "YOUR_NEWS_API_KEY"
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data.get("articles", [])[:5]

# Analyze sentiment
def analyze_sentiment(news_list):
    sentiments = []
    for article in news_list:
        result = sentiment_model(article["title"])[0]
        sentiments.append(result)
    return sentiments

# Main logic
if st.button("Analyze Stock"):
    try:
        data = get_stock_data(stock_symbol)
        st.subheader("📊 Recent Stock Prices")
        st.line_chart(data["Close"])

        news = get_stock_news(stock_symbol)
        st.subheader("📰 Latest News")
        for article in news:
            st.write("-", article["title"])

        sentiments = analyze_sentiment(news)
        st.subheader("🧠 News Sentiment Analysis")
        for s in sentiments:
            st.write(s)

        positive = sum(1 for s in sentiments if s["label"] == "POSITIVE")
        negative = sum(1 for s in sentiments if s["label"] == "NEGATIVE")

        if positive > negative:
            insight = "📈 Positive sentiment detected. Market mood looks bullish."
        elif negative > positive:
            insight = "📉 Negative sentiment detected. Market mood looks bearish."
        else:
            insight = "⚖️ Mixed sentiment. Market direction unclear."

        st.subheader("📌 AI Market Insight")
        st.success(insight)

    except Exception as e:
        st.error("Error fetching data or analyzing sentiment.")
        st.write(e)
