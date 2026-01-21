import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from groq import Groq

st.set_page_config(page_title="AI Trading Simulator", layout="wide")

st.title("📈 AI-Powered Trading Simulator with News-Driven Strategy Coach")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Sidebar controls
st.sidebar.header("Trading Controls")
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
investment = st.sidebar.number_input("Investment Amount ($)", 1000)

# Fetch stock data
def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="3mo")
    return data

# Fetch news (Google RSS)
def get_stock_news(symbol):
    url = f"https://news.google.com/rss/search?q={symbol}+stock"
    response = requests.get(url)
    return response.text[:2000]

# Analyze sentiment using Groq
def analyze_sentiment_with_groq(text):
    prompt = f"""
Analyze the sentiment of the following stock news.
Classify overall sentiment as Positive, Negative, or Neutral.
Also mention if any risky events (lawsuit, fraud, earnings, downgrade, investigation) appear.

News:
{text}
"""

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content

# Main logic
if st.button("Analyze Stock"):
    try:
        data = get_stock_data(symbol)

        st.subheader("📊 Stock Price Trend (3 Months)")
        st.line_chart(data["Close"])

        news_text = get_stock_news(symbol)

        st.subheader("📰 Raw News Feed")
        st.write(news_text[:1000])

        sentiment_result = analyze_sentiment_with_groq(news_text)

        st.subheader("🧠 AI News Sentiment & Risk Analysis")
        st.success(sentiment_result)

        # Simple AI Insight
        last_close = data["Close"].iloc[-1]
        first_close = data["Close"].iloc[0]

        if last_close > first_close:
            trend = "Uptrend 📈"
        else:
            trend = "Downtrend 📉"

        st.subheader("📌 AI Market Insight")
        st.info(f"Trend: {trend}")

        if "Negative" in sentiment_result:
            st.error("⚠️ Risk Alert: Negative news sentiment detected. Avoid aggressive buying.")
        elif "Positive" in sentiment_result:
            st.success("✅ Bullish Signal: Positive news sentiment detected. Consider buying opportunities.")
        else:
            st.warning("⚖️ Neutral Signal: Market direction unclear. Trade cautiously.")

    except Exception as e:
        st.error("Something went wrong.")
        st.write(e)
