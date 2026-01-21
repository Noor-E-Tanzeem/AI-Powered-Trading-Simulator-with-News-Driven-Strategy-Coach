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
st.set_page_config(page_title="Tanzeem's AlphaNexus: Institutional Intelligence", layout="wide")

# --- 2. ROBUST API INITIALIZATION ---
def init_clients():
    try:
        g_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        f_client = finnhub.Client(api_key=st.secrets["FINNHUB_API_KEY"])
        av_key = st.secrets["ALPHAVANTAGE_API_KEY"]
        return g_client, f_client, av_key
    except Exception as e:
        st.error(f"⚠️ Secrets Error: {e}. Ensure you used the TOML format exactly!")
        st.stop()

groq_client, finnhub_client, AV_KEY = init_clients()

# --- 3. CACHED DATA FETCHING (Protects your 25 daily requests) ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        return data.iloc[:1260].rename(columns={'5. adjusted close': 'Close', '6. volume': 'Volume'}).sort_index()
    except Exception as e:
        return f"LIMIT: {str(e)}"

@st.cache_data(ttl=600)
def get_intraday_data(ticker):
    try:
        ts = TimeSeries(key=AV_KEY, output_format='pandas')
        df, _ = ts.get_intraday(symbol=ticker, interval='60min', outputsize='compact')
        return df.rename(columns={'4. close': 'Close', '5. volume': 'Volume'})
    except Exception:
        return None

# --- 4. THE UI LAYOUT ---
st.title("🏛️ Tanzeem's ALPHANEXUS: INSTITUTIONAL TERMINAL")

with st.sidebar:
    ticker = st.text_input("ENTER TICKER", value="AAPL").upper()
    show_whales = st.toggle("Whale Activity Overlay", value=True)
    st.caption("v2.1 Stable Build | © 2026")

tab1, tab2, tab3 = st.tabs(["📊 COMMAND CENTER", "🧠 AI COUNCIL", "🧪 BACKTEST LAB"])

# TAB 1: COMMAND CENTER
with tab1:
    quote = finnhub_client.quote(ticker)
    if quote:
        c1, c2, c3 = st.columns(3)
        c1.metric("PRICE", f"${quote['c']}")
        c2.metric("CHANGE", f"{quote['dp']}%")
        c3.metric("STATUS", "STABLE" if abs(quote['dp']) < 3 else "VOLATILE")

        df_live = get_intraday_data(ticker)
        if isinstance(df_live, pd.DataFrame):
            fig = go.Figure(data=[go.Scatter(x=df_live.index, y=df_live['Close'], name="Price", line=dict(color="#00ffcc"))])
            if show_whales:
                df_live['Vol_Avg'] = df_live['Volume'].rolling(window=20).mean()
                df_live['Whale'] = np.where(df_live['Volume'] > (df_live['Vol_Avg'] * 3), df_live['Close'], np.nan)
                fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Whale'], mode='markers', name="Whale", marker=dict(size=12, color="orange", symbol="diamond")))
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Intraday data limit reached or unavailable.")

# TAB 2: AI COUNCIL
with tab2:
    st.header("The Strategy Council")
    if st.button("SUMMON COUNCIL"):
        news = finnhub_client.company_news(ticker, _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), to=datetime.now().strftime('%Y-%m-%d'))[:5]
        headlines = " | ".join([n['headline'] for n in news])
        with st.spinner("Council is deliberating..."):
            prompt = f"Analyze {ticker} based on: {headlines}. Debate as a Bull, a Bear, and a Whale."
            res = groq_client.chat.completions.create(model="llama3-70b-8192", messages=[{"role": "user", "content": prompt}])
            st.markdown(res.choices[0].message.content)

# TAB 3: BACKTEST LAB
with tab3:
    st.header("5-Year Historical Performance")
    if st.button("RUN SIMULATION"):
        results = get_historical_data(ticker)
        if isinstance(results, pd.DataFrame):
            results['EMA_F'] = ta.ema(results['Close'], length=20)
            results['EMA_S'] = ta.ema(results['Close'], length=50)
            results['Sig'] = np.where(results['EMA_F'] > results['EMA_S'], 1, 0)
            results['Bench'] = (results['Close'].pct_change() + 1).cumprod()
            results['Strat'] = (results['Close'].pct_change() * results['Sig'].shift(1) + 1).cumprod()
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=results.index, y=results['Strat'], name="AI Strategy", line=dict(color="#00ffcc")))
            fig_bt.add_trace(go.Scatter(x=results.index, y=results['Bench'], name="Market Bench", line=dict(color="gray", dash='dash')))
            fig_bt.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.error("🚨 Alpha Vantage Limit Reached (25 requests/day). Caching is active; try again in 1 hour.")
