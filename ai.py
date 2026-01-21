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
st.set_page_config(page_title="AlphaNexus: Institutional Intelligence", layout="wide")

# --- 2. ROBUST API INITIALIZATION ---
# This prevents the 'NameError' by checking for keys first
def init_clients():
    try:
        g_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        f_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
        av_key = st.secrets["ALPHAVANTAGE_API_KEY"]
        return g_client, f_client, av_key
    except Exception as e:
        st.error(f"⚠️ API Key Error: {e}. Check your Streamlit Secrets!")
        st.stop()

groq_client, finnhub_client, ALPHAVANTAGE_KEY = init_clients()

# --- 3. THE "RARE" FEATURE ENGINES ---

def get_council_debate(ticker, news):
    """Multi-Agent Strategy Debate"""
    prompt = f"Act as a Council of 3 Traders. Analyze {ticker} news: {news}. Debate the Bull Case, Bear Case, and Institutional 'Whale' perspective."
    chat = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return chat.choices[0].message.content

def detect_whales(df):
    """Institutional Volume Spike Detector"""
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    # If volume is 3x higher than average, it's a 'Whale' movement
    df['Whale_Signal'] = np.where(df['Volume'] > (df['Vol_Avg'] * 3), df['Close'], np.nan)
    return df

def backtest_engine(ticker):
    """5-Year Historical Performance Engine"""
    ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format='pandas')
    # Fetch 5 years (approx 1260 trading days)
    data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
    df = data.iloc[:1260].rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'}).sort_index()
    
    # Strategy: EMA 20/50 Cross + RSI 14
    df['EMA_F'] = ta.ema(df['Close'], length=20)
    df['EMA_S'] = ta.ema(df['Close'], length=50)
    df['Signal'] = np.where(df['EMA_F'] > df['EMA_S'], 1, 0)
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Returns'] * df['Signal'].shift(1)
    df['Market_Cum'] = (1 + df['Returns']).cumprod()
    df['Strategy_Cum'] = (1 + df['Strategy']).cumprod()
    return df

# --- 4. THE UI LAYOUT ---
st.title("🏛️ ALPHANEXUS: ELITE STRATEGY TERMINAL")
st.caption(f"System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data Stream: STABLE")

ticker = st.sidebar.text_input("ENTER TICKER SYMBOL", value="NVDA").upper()
st.sidebar.divider()
st.sidebar.markdown("### 🛠️ Terminal Tools")
show_whales = st.sidebar.checkbox("Show Whale Activity", value=True)

tab1, tab2, tab3 = st.tabs(["📊 LIVE COMMAND", "🧠 AI COUNCIL", "🧪 BACKTEST LAB"])

# --- TAB 1: LIVE COMMAND ---
with tab1:
    quote = finnhub_client.quote(ticker)
    if quote:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CURRENT PRICE", f"${quote['c']}")
        c2.metric("24H CHANGE", f"{quote['dp']}%", f"{quote['d']}")
        c3.metric("OPEN", f"${quote['o']}")
        c4.metric("VOLATILITY", "HIGH" if abs(quote['dp']) > 2 else "LOW")

        # Live-ish Chart (Last 100 periods)
        ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format='pandas')
        df_live, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
        df_live = df_live.rename(columns={'4. close': 'Close', '5. volume': 'Volume'})
        df_live = detect_whales(df_live)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Close'], name="Price", line=dict(color="#00ffcc", width=2)))
        
        if show_whales:
            fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Whale_Signal'], mode='markers', 
                                     name="Institutional Whale", marker=dict(size=12, color="orange", symbol="diamond")))

        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: AI COUNCIL ---
with tab2:
    st.header("The Strategy Council (Multi-Agent Debate)")
    news = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))[:5]
    
    if st.button("RUN COUNCIL ANALYSIS"):
        headlines = " | ".join([n['headline'] for n in news])
        with st.spinner("Council is deliberating..."):
            debate_result = get_council_debate(ticker, headlines)
            st.markdown(f"### Live Debate Result\n{debate_result}")
    
    with st.expander("Latest Intelligence Feed"):
        for n in news:
            st.write(f"📅 {datetime.fromtimestamp(n['datetime']).strftime('%Y-%m-%d')} - **{n['headline']}**")

# --- TAB 3: BACKTEST LAB ---
with tab3:
    st.header("5-Year Historical Stress Test")
    if st.button("RUN 5-YEAR BACKTEST"):
        with st.spinner("Processing 1,260 days of market data..."):
            bt_results = backtest_engine(ticker)
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Strategy_Cum'], name="AI Strategy", line=dict(color="#00ffcc")))
            fig_bt.add_trace(go.Scatter(x=bt_results.index, y=bt_results['Market_Cum'], name="Market Bench", line=dict(color="gray", dash='dash')))
            fig_bt.update_layout(template="plotly_dark", title="Wealth Growth: Strategy vs Market")
            st.plotly_chart(fig_bt, use_container_width=True)
            
            # Key Stats
            total_return = (bt_results['Strategy_Cum'].iloc[-1] - 1) * 100
            st.success(f"Strategy Total Return: {total_return:.2f}%")

st.divider()
st.caption("AlphaNexus Terminal | Institutional Grade AI Simulator | © 2026")
