import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
import finnhub
import numpy as np
import requests
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIG & ELITE STYLING ---
st.set_page_config(page_title="Tanzeem's AlphaNexus: Institutional Quant", layout="wide")

# Custom CSS for Institutional "Glow" and Dark Theme Branding
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeading h1 { 
        color: #00ffcc; 
        text-shadow: 0 0 10px #00ffcc, 0 0 20px #00ffcc;
        font-family: 'Courier New', Courier, monospace;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1c24;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #00ffcc !important; color: black !important; }
    </style>
    """, unsafe_allow_html=True)

# Safe Secret Loading
def get_secret(key):
    try: return st.secrets[key]
    except: st.error(f"Missing {key}"); st.stop()

GROQ_API_KEY = get_secret("GROQ_API_KEY")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
AV_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

# --- 2. MULTI-ASSET DATA ENGINE ---
@st.cache_data(ttl=600)
def fetch_data(ticker, function='TIME_SERIES_DAILY'):
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={AV_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        key = "Time Series (Daily)" if "DAILY" in function else "Time Series (Digital Currency Daily)"
        if key in data:
            df = pd.DataFrame.from_dict(data[key], orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] if "DAILY" in function else ['Open', 'High', 'Low', 'Close', 'Vol', 'MarketCap']
            df.index = pd.to_datetime(df.index)
            return df.astype(float).sort_index()
        return "LIMIT"
    except: return None

# --- 3. SIDEBAR COMMANDER ---
with st.sidebar:
    st.markdown("# 🏛️ ALPHANEXUS")
    st.markdown("### QUANT TERMINAL v3.1")
    st.divider()
    ticker = st.text_input("PRIMARY TICKER", value="TSLA").upper()
    st.divider()
    st.subheader("🎲 Stress Test Params")
    sim_days = st.slider("Horizon", 30, 250, 90)
    num_paths = st.slider("Paths", 10, 100, 50)
    st.divider()
    st.caption("Developed by Tanzeem | © 2026")

# --- 4. MAIN TERMINAL UI ---
st.title("Tanzeem's AlphaNexus: Institutional Quant")

tabs = st.tabs(["🌐 GLOBAL MACRO", "📊 ASSET ANALYSIS", "🧠 AI COUNCIL", "🧪 STRATEGY LAB"])

# --- TAB 1: GLOBAL MACRO (The 'Rare' Feature) ---
with tabs[0]:
    st.header("Global Market Health Dashboard")
    col1, col2, col3 = st.columns(3)
    
    # We use common indices to show market health
    with st.spinner("Syncing Global Markets..."):
        btc = fetch_data("BTC", "DIGITAL_CURRENCY_DAILY")
        gold = fetch_data("GLD") # Gold ETF
        spy = fetch_data("SPY")  # S&P 500
        
        if isinstance(btc, pd.DataFrame):
            col1.metric("BITCOIN (24H)", f"${btc['Close'].iloc[-1]:,.2f}")
        if isinstance(gold, pd.DataFrame):
            col2.metric("GOLD (GLD)", f"${gold['Close'].iloc[-1]:,.2f}")
        if isinstance(spy, pd.DataFrame):
            col3.metric("S&P 500 (SPY)", f"${spy['Close'].iloc[-1]:,.2f}")
            
    st.info("Market Sentiment: Institutional flows are currently monitoring high-volatility zones in the assets above.")

# --- TAB 2: ASSET ANALYSIS ---
with tabs[1]:
    df = fetch_data(ticker)
    if isinstance(df, pd.DataFrame):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # Whale Detection logic
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()
        df['Whale'] = np.where(df['Volume'] > (df['Vol_Avg'] * 2.5), 'orange', '#00ffcc')
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=df['Whale'], name="Volume"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data Limit Reached.")

# --- TAB 3: AI COUNCIL ---
with tabs[2]:
    if st.button("RUN COUNCIL DEBATE"):
        client = Groq(api_key=GROQ_API_KEY)
        with st.spinner("AI Agents Deliberating..."):
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Analyze {ticker} from a Macro and Micro perspective."}]
            )
            st.markdown(res.choices[0].message.content)

# --- TAB 4: STRATEGY LAB (Monte Carlo) ---
with tabs[3]:
    st.header("Monte Carlo Probability Projection")
    if isinstance(df, pd.DataFrame):
        returns = df['Close'].pct_change().dropna()
        last_price = df['Close'].iloc[-1]
        
        sim_results = []
        for _ in range(num_paths):
            prices = [last_price]
            for _ in range(sim_days):
                prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
            sim_results.append(prices)
            
        fig_sim = go.Figure()
        for path in sim_results:
            fig_sim.add_trace(go.Scatter(y=path, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        fig_sim.update_layout(template="plotly_dark", title=f"Future Projection ({num_paths} Paths)")
        st.plotly_chart(fig_sim, use_container_width=True)
