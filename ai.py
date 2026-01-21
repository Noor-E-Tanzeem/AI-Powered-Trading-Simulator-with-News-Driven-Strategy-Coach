import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from groq import Groq
import finnhub
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# --- 1. SET PAGE CONFIG ---
st.set_page_config(page_title="Tanzeem's AlphaNexus: Global Intelligence", layout="wide")

# --- 2. API INITIALIZATION & CACHING ---
def init_clients():
    try:
        g_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        f_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
        av_key = st.secrets["ALPHAVANTAGE_API_KEY"]
        return g_client, f_client, av_key
    except Exception as e:
        st.error(f"⚠️ Secrets Error: {e}. Check your TOML formatting!")
        st.stop()

groq_client, finnhub_client, AV_KEY = init_clients()

# CACHED DATA FETCHING (Prevents 25-request limit burnout)
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        df = data.iloc[:1260].rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'}).sort_index()
        return df
    except ValueError:
        return "LIMIT"
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_intraday_data(ticker):
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        df, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
        return df.rename(columns={'4. close': 'Close', '5. volume': 'Volume'})
    except Exception:
        return None

# --- 3. THE "RARE" ANALYTICS ENGINES ---

def detect_whales(df):
    """Institutional Volume Tracker"""
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['Whale_Signal'] = np.where(df['Volume'] > (df['Vol_Avg'] * 3), df['Close'], np.nan)
    return df

# --- 4. THE UI LAYOUT ---
st.title("🏛️ ALPHANEXUS: INSTITUTIONAL TERMINAL")
ticker = st.sidebar.text_input("TICKER", value="AAPL").upper()
show_whales = st.sidebar.toggle("Whale Activity Overlay", value=True)

tab1, tab2, tab3 = st.tabs(["📊 COMMAND CENTER", "🧠 AI COUNCIL", "🧪 BACKTEST LAB"])

# TAB 1: LIVE COMMAND
with tab1:
    quote = finnhub_client.quote(ticker)
    if quote['c'] != 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("PRICE", f"${quote['c']}")
        c2.metric("CHANGE", f"{quote['dp']}%")
        c3.metric("OPEN", f"${quote['o']}")

        df_live = get_intraday_data(ticker)
        if df_live is not None:
            df_live = detect_whales(df_live)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Close'], name="Price", line=dict(color="#00ffcc")))
            if show_whales:
                fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Whale_Signal'], mode='markers', name="Whale Spike", marker=dict(size=10, color="orange")))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Intraday data limit reached. Upgrade to Premium for real-time streams.")

# TAB 3: BACKTEST LAB (Handling the ValueError)
with tab3:
    st.header("5-Year Alpha Simulation")
    if st.button("RUN SIMULATION"):
        results = get_historical_data(ticker)
        
        if isinstance(results, str) and results == "LIMIT":
            st.error("🚨 ALPHA VANTAGE LIMIT REACHED (25 Requests/Day). The simulator is locked to prevent API bans. Try again in 24 hours.")
        elif results is not None:
            # Simple Backtest Logic
            results['EMA20'] = ta.ema(results['Close'], length=20)
            results['EMA50'] = ta.ema(results['Close'], length=50)
            results['Signal'] = np.where(results['EMA20'] > results['EMA50'], 1, 0)
            results['Strat_Returns'] = (results['Close'].pct_change() * results['Signal'].shift(1)).add(1).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=results.index, y=results['Strat_Returns'], name="AI Strategy", line=dict(color="#00ffcc")))
            st.plotly_chart(fig_bt, use_container_width=True)
            st.success(f"Final Strategy Growth: {((results['Strat_Returns'].iloc[-1]-1)*100):.2f}%")
        else:
            st.error("Could not retrieve data for this ticker.")

st.sidebar.caption("v2.0 Stable Build | © 2026")
