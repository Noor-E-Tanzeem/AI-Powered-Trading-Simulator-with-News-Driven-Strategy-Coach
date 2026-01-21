import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from groq import Groq
import finnhub
import numpy as np
import requests

# --- 1. CONFIG & SYSTEM CHECK ---
st.set_page_config(page_title="Tanzeem's AlphaNexus Pro", layout="wide")

# Safe Secret Loading
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        st.error(f"Missing {key} in Streamlit Secrets!")
        st.stop()

GROQ_API_KEY = get_secret("GROQ_API_KEY")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
AV_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# --- 2. THE STABLE DATA ENGINE ---
@st.cache_data(ttl=600)
def fetch_stock_data(ticker):
    """Uses a direct URL request to avoid library-specific ValueErrors"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={AV_API_KEY}'
    r = requests.get(url)
    data = r.json()
    
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.astype(float).sort_index()
        return df
    elif "Note" in data or "Information" in data:
        return "LIMIT"
    else:
        return "ERROR"

# --- 3. UI LAYOUT ---
st.title("🏛️Tanzeem's ALPHANEXUS: INSTITUTIONAL TERMINAL")
ticker = st.sidebar.text_input("SYMBOL", value="AAPL").upper()

tab1, tab2 = st.tabs(["📊 COMMAND CENTER", "🧠 AI COUNCIL"])

with tab1:
    # Get Data
    df = fetch_stock_data(ticker)
    
    if isinstance(df, pd.DataFrame):
        # Professional Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", height=500, title=f"{ticker} Historical Feed")
        st.plotly_chart(fig, use_container_width=True)
    elif df == "LIMIT":
        st.error("🚨 API LIMIT REACHED. Alpha Vantage is blocking this key for today.")
    else:
        st.warning("Could not find ticker data. Please verify the symbol.")

with tab2:
    st.header("The Council of Agents")
    if st.button("SUMMON COUNCIL"):
        client = Groq(api_key=GROQ_API_KEY)
        with st.spinner("AI is deliberating..."):
            # Using the new llama-3.3-70b-versatile model
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Analyze {ticker} market sentiment."}]
            )
            st.markdown(chat.choices[0].message.content)

st.sidebar.caption("Stable Build 2.5")
