import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from groq import Groq
import numpy as np
import requests
from datetime import datetime

# --- 1. ELITE UI CONFIG ---
st.set_page_config(page_title="AlphaNexus Elite", layout="wide")

# Custom CSS for the "Institutional Glow"
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    .stHeading h1 { color: #00ffcc; text-shadow: 0 0 10px #00ffcc; text-align: center; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricValue"] { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE SMART FETCH ENGINE (Saves your 25 requests) ---
# This keeps data in memory for 24 hours so you don't burn your limit!
@st.cache_data(ttl=86400) 
def fetch_alpha_smart(ticker, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
    try:
        r = requests.get(url)
        data = r.json()
        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df.index = pd.to_datetime(df.index)
            return df.astype(float).sort_index()
        return data # Returns the error message (like "API limit reached")
    except:
        return None

# --- 3. MAIN INTERFACE ---
st.title("Tanzeem's AlphaNexus: Institutional Quant")

with st.sidebar:
    st.header("🏛️ COMMANDER")
    # Use session state so the ticker doesn't reset and trigger a new API call
    ticker = st.text_input("SYMBOL", value="TSLA").upper()
    av_key = st.secrets["ALPHAVANTAGE_API_KEY"]
    groq_key = st.secrets["GROQ_API_KEY"]
    st.divider()
    st.caption("AlphaNexus v3.9 | Anti-Throttle Mode")

tab1, tab2, tab3 = st.tabs(["📊 ANALYSIS", "🧠 AI COUNCIL", "🎲 PROJECTION"])

# --- TAB 1: SMART ANALYSIS ---
with tab1:
    data = fetch_alpha_smart(ticker, av_key)
    
    if isinstance(data, pd.DataFrame):
        # Professional Candlestick + Whale Tracker
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price")])
        
        # Rare Feature: The "Institutional Entry" markers
        data['MA50'] = data['Close'].rolling(50).mean()
        whales = data[data['Volume'] > (data['Volume'].rolling(20).mean() * 2.5)]
        fig.add_trace(go.Scatter(x=whales.index, y=whales['Close'], mode='markers', name="Whale Entry", marker=dict(color='orange', size=10, symbol='diamond')))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"⚠️ API Status: {data.get('Note', 'Connection Error') if isinstance(data, dict) else 'Check API Key'}")

# --- TAB 2: AI COUNCIL ---
with tab2:
    if st.button("RUN COUNCIL DEBATE"):
        client = Groq(api_key=groq_key)
        # We pass only the last 5 days of data to save tokens
        context = data.tail(5).to_string() if isinstance(data, pd.DataFrame) else "No data"
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Analyze these stock stats for {ticker}: {context}. Give a Bull/Bear verdict."}]
        )
        st.markdown(res.choices[0].message.content)

# --- TAB 3: QUANT PROJECTION ---
with tab3:
    if isinstance(data, pd.DataFrame):
        st.header("Monte Carlo Probability paths")
        returns = data['Close'].pct_change().dropna()
        last_p = data['Close'].iloc[-1]
        
        # Generate 30 paths for 60 days
        all_paths = []
        for _ in range(30):
            p = [last_p]
            for _ in range(60):
                p.append(p[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
            all_paths.append(p)
            
        fig_sim = go.Figure()
        for path in all_paths:
            fig_sim.add_trace(go.Scatter(y=path, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
        fig_sim.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sim, use_container_width=True)
