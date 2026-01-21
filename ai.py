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
st.set_page_config(page_title="Tanzeem's AlphaNexus: Institutional Terminal", layout="wide")

# --- 2. API INITIALIZATION ---
def init_clients():
    try:
        # Check for keys in Streamlit Secrets
        g_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        f_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
        av_key = st.secrets["ALPHAVANTAGE_API_KEY"]
        return g_client, f_client, av_key
    except Exception as e:
        st.error(f"⚠️ Secrets Error: {e}. Check your TOML formatting in Streamlit dashboard!")
        st.stop()

groq_client, finnhub_client, AV_KEY = init_clients()

# --- 3. CACHED ENGINES (Protects your 25 daily request limit) ---

@st.cache_data(ttl=3600) # Remembers data for 1 hour
def get_historical_data(ticker):
    """Fetches 5 years of daily data for backtesting"""
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        df = data.iloc[:1260].rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'}).sort_index()
        return df
    except ValueError:
        return "LIMIT"
    except Exception:
        return None

@st.cache_data(ttl=600) # Remembers live chart for 10 minutes
def get_intraday_data(ticker):
    """Fetches hourly data for the live command center"""
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        df, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
        return df.rename(columns={'4. close': 'Close', '5. volume': 'Volume'})
    except Exception:
        return None

# --- 4. ANALYTICS ENGINE ---
def detect_whales(df):
    """Detects institutional 'Whale' volume spikes (3x average)"""
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['Whale_Signal'] = np.where(df['Volume'] > (df['Vol_Avg'] * 3), df['Close'], np.nan)
    return df

# --- 5. UI LAYOUT ---
st.title("🏛️Tanzeem's ALPHANEXUS: ELITE STRATEGY TERMINAL")
ticker = st.sidebar.text_input("ENTER TICKER SYMBOL", value="AAPL").upper()
show_whales = st.sidebar.toggle("Whale Activity Overlay", value=True)

tab1, tab2, tab3 = st.tabs(["📊 COMMAND CENTER", "🧠 AI COUNCIL", "🧪 BACKTEST LAB"])

# TAB 1: COMMAND CENTER
with tab1:
    quote = finnhub_client.quote(ticker)
    if quote and quote['c'] != 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("PRICE", f"${quote['c']}")
        c2.metric("24H CHANGE", f"{quote['dp']}%")
        c3.metric("OPEN", f"${quote['o']}")

        df_live = get_intraday_data(ticker)
        if df_live is not None:
            df_live = detect_whales(df_live)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Close'], name="Price", line=dict(color="#00ffcc")))
            if show_whales:
                fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Whale_Signal'], mode='markers', name="Whale Spike", marker=dict(size=12, color="orange", symbol="diamond")))
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Intraday data limit reached. Upgrade for real-time streams.")

# TAB 2: AI COUNCIL
with tab2:
    st.header("The Strategy Council (Multi-Agent Debate)")
    if st.button("RUN COUNCIL ANALYSIS"):
        news = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))[:5]
        headlines = " | ".join([n['headline'] for n in news])
        with st.spinner("Council is deliberating..."):
            # UPDATED TO SUPPORTED MODEL llama-3.3-70b-versatile
            debate = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Analyze {ticker} based on news: {headlines}. Debate the Bull and Bear case."}]
            )
            st.markdown(debate.choices[0].message.content)

# TAB 3: BACKTEST LAB
with tab3:
    st.header("5-Year Historical Stress Test")
    if st.button("RUN SIMULATION"):
        results = get_historical_data(ticker)
        if isinstance(results, str) and results == "LIMIT":
            st.error("🚨 ALPHA VANTAGE LIMIT REACHED (25 Requests/Day). The system is locked to protect your API status.")
        elif results is not None:
            results['EMA20'] = ta.ema(results['Close'], length=20)
            results['EMA50'] = ta.ema(results['Close'], length=50)
            results['Sig'] = np.where(results['EMA20'] > results['EMA50'], 1, 0)
            results['Strat_Growth'] = (results['Close'].pct_change() * results['Sig'].shift(1) + 1).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=results.index, y=results['Strat_Growth'], name="AI Strategy", line=dict(color="#00ffcc")))
            fig_bt.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bt, use_container_width=True)
            st.success(f"Final Strategy Growth: {((results['Strat_Growth'].iloc[-1]-1)*100):.2f}%")
        else:
            st.error("Data retrieval failed.")

st.sidebar.divider()
st.sidebar.caption("v2.3 Stable Build | © 2026")
