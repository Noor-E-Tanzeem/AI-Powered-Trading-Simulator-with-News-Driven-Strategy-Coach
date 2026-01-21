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

# --- 1. ELITE UI CONFIG & NEON BRANDING ---
st.set_page_config(page_title="Tanzeem's AlphaNexus", layout="wide", page_icon="🏛️")

# Custom CSS for the "Institutional Glow"
st.markdown("""
    <style>
    .main { background-color: #05070a; }
    .stHeading h1 { 
        color: #00ffcc; 
        text-shadow: 0 0 15px #00ffcc;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 2px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] { color: #00ffcc; font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0e1117; border-radius: 10px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #808080; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom: 2px solid #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE MULTI-THREAD DATA ENGINE ---
def get_secret(key):
    try: return st.secrets[key]
    except: st.error(f"Missing {key}"); st.stop()

GROQ_API_KEY = get_secret("GROQ_API_KEY")
FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY")
AV_API_KEY = get_secret("ALPHAVANTAGE_API_KEY")

@st.cache_data(ttl=600)
def fetch_alpha_data(ticker, function='TIME_SERIES_DAILY'):
    """High-reliability fetcher for Stocks and Crypto"""
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={AV_API_KEY}'
    try:
        r = requests.get(url)
        data = r.json()
        # Handle both Stock and Crypto keys
        ts_key = next((k for k in data.keys() if "Time Series" in k), None)
        if ts_key:
            df = pd.DataFrame.from_dict(data[ts_key], orient='index')
            # Standardize columns
            df.columns = [c.split('. ')[-1].title() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            return df.astype(float).sort_index()
        return "LIMIT" if "Information" in str(data) else None
    except: return None

# --- 3. SIDEBAR: THE QUANT COMMANDER ---
with st.sidebar:
    st.markdown("<h2 style='color:#00ffcc;'>🏛️ COMMANDER</h2>", unsafe_allow_html=True)
    ticker = st.text_input("PRIMARY ASSET", value="TSLA").upper()
    st.divider()
    st.subheader("🎲 Simulation Matrix")
    sim_days = st.slider("Forecast Horizon", 30, 250, 90)
    num_paths = st.slider("Quantum Paths", 10, 100, 50)
    st.divider()
    st.caption("AlphaNexus Terminal v3.2 | Built by Tanzeem")

# --- 4. MAIN TERMINAL INTERFACE ---
st.title("Tanzeem's AlphaNexus: Institutional Quant")

tabs = st.tabs(["🌐 GLOBAL MACRO", "📊 PRICE ACTION", "🧠 AI RISK COUNCIL", "🎲 PREDICTIVE LAB"])

# --- TAB 1: GLOBAL MACRO DASHBOARD ---
with tabs[0]:
    st.subheader("Global Health & Correlation")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with st.spinner("Accessing Global Feeds..."):
        # Fetching 'The Big Three'
        spy_data = fetch_alpha_data("SPY")
        btc_data = fetch_alpha_data("BTC", "DIGITAL_CURRENCY_DAILY")
        gold_data = fetch_alpha_data("GLD")
        
        if isinstance(spy_data, pd.DataFrame):
            m_col1.metric("S&P 500 (SPY)", f"${spy_data['Close'].iloc[-1]:,.2f}")
        if isinstance(btc_data, pd.DataFrame):
            m_col2.metric("BITCOIN (BTC)", f"${btc_data['Close'].iloc[-1]:,.2f}")
        if isinstance(gold_data, pd.DataFrame):
            m_col3.metric("GOLD (GLD)", f"${gold_data['Close'].iloc[-1]:,.2f}")
    
    st.markdown("---")
    st.info("💡 **Institutional Note:** Monitor BTC/SPY correlations. Divergence often signals impending volatility.")

# --- TAB 2: PRICE ACTION & WHALE TRACKER ---
with tabs[1]:
    df = fetch_alpha_data(ticker)
    if isinstance(df, pd.DataFrame):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Pro Candlesticks
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        
        # Whale Detection (Volume Heatmap)
        df['Vol_Avg'] = df['Volume'].rolling(20).mean()
        df['Whale'] = np.where(df['Volume'] > (df['Vol_Avg'] * 2.5), 'orange', '#00ffcc')
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=df['Whale'], name="Vol/Whale Spikes"), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("🚨 API LIMIT REACHED. Please wait for reset.")

# --- TAB 3: AI RISK COUNCIL (Multi-Agent Debate) ---
with tabs[2]:
    st.header("Strategic Deliberation")
    if st.button("INITIATE COUNCIL DEBATE"):
        client = Groq(api_key=GROQ_API_KEY)
        f_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        
        # Fetch news context
        news = f_client.company_news(ticker, _from=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))[:5]
        headlines = " | ".join([n['headline'] for n in news])
        
        with st.spinner("AI Agents Deliberating..."):
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"You are a hedge fund risk committee. Analyze {ticker} news: {headlines}. Present a 'Bull Case', a 'Bear Case', and a 'Final Quant Verdict'."}]
            )
            st.markdown(res.choices[0].message.content)

# --- TAB 4: PREDICTIVE LAB (Monte Carlo Simulation) ---
with tabs[3]:
    st.header("Monte Carlo Probability Stress Test")
    if isinstance(df, pd.DataFrame):
        returns = df['Close'].pct_change().dropna()
        last_price = df['Close'].iloc[-1]
        
        # Simulation Matrix
        sim_data = np.zeros((sim_days, num_paths))
        for i in range(num_paths):
            sim_path = [last_price]
            for _ in range(sim_days - 1):
                sim_path.append(sim_path[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
            sim_data[:, i] = sim_path
            
        fig_sim = go.Figure()
        for i in range(num_paths):
            fig_sim.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
            
        fig_sim.update_layout(template="plotly_dark", title=f"Future Price Projection: {num_paths} Scenarios", xaxis_title="Days Forward", yaxis_title="Projected Price")
        st.plotly_chart(fig_sim, use_container_width=True)
        
        avg_end_price = sim_data[-1, :].mean()
        st.metric("Mean Target Price", f"${avg_end_price:.2f}", delta=f"{((avg_end_price/last_price)-1)*100:.2f}%")
