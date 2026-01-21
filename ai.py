import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from groq import Groq
import finnhub
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="AlphaNexus Pro", layout="wide")

# Robust Secret Loading
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        st.error(f"Missing {key} in Streamlit Secrets! Use KEY = 'VALUE' format.")
        st.stop()

GROQ_API_KEY = get_secret("GROQ_API_KEY")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
AV_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# --- 2. STABLE DATA ENGINE (Direct Request Method) ---
@st.cache_data(ttl=600)
def fetch_stock_data(ticker):
    """Fetches historical daily data for charting and backtesting"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={AV_API_KEY}'
    try:
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
        return None
    except:
        return None

# --- 3. UI LAYOUT ---
st.title("🏛️ ALPHANEXUS: INSTITUTIONAL TERMINAL")
ticker = st.sidebar.text_input("SYMBOL", value="AAPL").upper()
show_whales = st.sidebar.toggle("Whale Activity Overlay", value=True)

# THE THREE TABS
tab1, tab2, tab3 = st.tabs(["📊 COMMAND CENTER", "🧠 AI COUNCIL", "🧪 BACKTEST LAB"])

# TAB 1: COMMAND CENTER
with tab1:
    df = fetch_stock_data(ticker)
    if isinstance(df, pd.DataFrame):
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market")])
        
        if show_whales:
            df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
            df['Whale'] = np.where(df['Volume'] > (df['Vol_Avg'] * 3), df['Close'], np.nan)
            fig.add_trace(go.Scatter(x=df.index, y=df['Whale'], mode='markers', name="Whale Spike", marker=dict(size=12, color="orange", symbol="diamond")))
            
        fig.update_layout(template="plotly_dark", height=500, title=f"{ticker} Market Feed")
        st.plotly_chart(fig, use_container_width=True)
    elif df == "LIMIT":
        st.error("🚨 ALPHA VANTAGE LIMIT REACHED (25 Requests/Day).")

# TAB 2: AI COUNCIL
with tab2:
    st.header("The Strategy Council")
    if st.button("SUMMON COUNCIL"):
        client = Groq(api_key=GROQ_API_KEY)
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        news = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))[:5]
        headlines = " | ".join([n['headline'] for n in news])
        
        with st.spinner("Council is deliberating..."):
            chat = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Analyze {ticker} based on news: {headlines}. Debate as a Bull and a Bear."}]
            )
            st.markdown(chat.choices[0].message.content)

# TAB 3: BACKTEST LAB (The Missing Tab)
with tab3:
    st.header("5-Year Historical Stress Test")
    if st.button("RUN SIMULATION"):
        df_bt = fetch_stock_data(ticker)
        if isinstance(df_bt, pd.DataFrame):
            # Strategy: EMA 20/50 Cross
            df_bt['EMA20'] = ta.ema(df_bt['Close'], length=20)
            df_bt['EMA50'] = ta.ema(df_bt['Close'], length=50)
            df_bt['Signal'] = np.where(df_bt['EMA20'] > df_bt['EMA50'], 1, 0)
            df_bt['Returns'] = (df_bt['Close'].pct_change() * df_bt['Signal'].shift(1) + 1).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Returns'], name="AI Strategy Growth", line=dict(color="#00ffcc")))
            fig_bt.update_layout(template="plotly_dark", title="Wealth Cumulative Growth")
            st.plotly_chart(fig_bt, use_container_width=True)
            
            total_growth = (df_bt['Returns'].iloc[-1] - 1) * 100
            st.success(f"Strategy Simulation Complete: {total_growth:.2f}% Total Growth")
        else:
            st.error("Could not run backtest. Check API limits.")

st.sidebar.caption("v2.6 Stable Build | © 2026")
