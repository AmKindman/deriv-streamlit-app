import streamlit as st
import pandas as pd
import numpy as np
import json
import asyncio
import websockets
from collections import deque

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deriv Market Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Deriv Market Predictor")
st.caption("Real-Time Quantitative Momentum Analysis using Deriv WebSocket API")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Settings")
symbol = st.sidebar.selectbox(
    "Select Synthetic Index",
    ["1HZ100V", "1HZ75V", "1HZ50V", "1HZ25V", "1HZ10V"],
    format_func=lambda x: {
        "1HZ100V": "Volatility 100 (1s) Index",
        "1HZ75V": "Volatility 75 (1s) Index",
        "1HZ50V": "Volatility 50 (1s) Index",
        "1HZ25V": "Volatility 25 (1s) Index",
        "1HZ10V": "Volatility 10 (1s) Index"
    }.get(x, x)
)

fast_span = st.sidebar.slider("Fast EMA Span", min_value=3, max_value=15, value=5)
slow_span = st.sidebar.slider("Slow EMA Span", min_value=10, max_value=50, value=20)
confidence_threshold = st.sidebar.slider("Signal Threshold (%)", min_value=10, max_value=80, value=30)

run_stream = st.sidebar.button("▶ Start Live Stream", type="primary")

# --- SESSION STATE INITIALISATION ---
if "ticks" not in st.session_state:
    st.session_state.ticks = deque(maxlen=100)

# --- PREDICTIVE ENGINE ---
def analyze_ticks(ticks_list):
    df = pd.DataFrame(ticks_list)
    if len(df) < slow_span:
        return {"prediction": "WARMING UP", "confidence": 0.0, "fast_ema": 0, "slow_ema": 0, "latest": df['price'].iloc[-1]}
    
    # Calculate Exponential Moving Averages
    df['fast_ema'] = df['price'].ewm(span=fast_span, adjust=False).mean()
    df['slow_ema'] = df['price'].ewm(span=slow_span, adjust=False).mean()
    df['volatility'] = df['price'].rolling(window=10).std().fillna(0.001)
    
    latest_price = df['price'].iloc[-1]
    fast_val = df['fast_ema'].iloc[-1]
    slow_val = df['slow_ema'].iloc[-1]
    vol = df['volatility'].iloc[-1] if df['volatility'].iloc[-1] > 0 else 0.001
    
    # Normalized momentum score via hyperbolic tangent
    delta = fast_val - slow_val
    score = np.tanh(delta / vol)
    confidence = abs(float(score)) * 100
    
    if score > (confidence_threshold / 100):
        prediction = "RISE (CALL) 🟢"
    elif score < -(confidence_threshold / 100):
        prediction = "FALL (PUT) 🔴"
    else:
        prediction = "NEUTRAL ⚪"
        
    return {
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "fast_ema": round(fast_val, 3),
        "slow_ema": round(slow_val, 3),
        "latest": round(latest_price, 3)
    }

# --- DASHBOARD LAYOUT ---
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    price_placeholder = st.empty()
with metric_col2:
    signal_placeholder = st.empty()
with metric_col3:
    confidence_placeholder = st.empty()

chart_placeholder = st.empty()

# --- WEBSOCKET STREAMING ENGINE ---
async def fetch_deriv_data():
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
        
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if "tick" in data:
                price = float(data["tick"]["quote"])
                epoch = data["tick"]["epoch"]
                
                st.session_state.ticks.append({"time": pd.to_datetime(epoch, unit='s'), "price": price})
                
                # Analyze market state
                res = analyze_ticks(st.session_state.ticks)
                
                # Update Dashboard UI
                price_placeholder.metric("Latest Price", f"{res['latest']}")
                signal_placeholder.metric("Market Prediction", res['prediction'])
                confidence_placeholder.metric("Signal Strength", f"{res['confidence']}%")
                
                # Render Live Line Chart
                df_chart = pd.DataFrame(st.session_state.ticks).set_index("time")
                chart_placeholder.line_chart(df_chart['price'])
                
            await asyncio.sleep(0.1)

# Execution trigger
if run_stream:
    st.info("Connected to Deriv API. Streaming real-time tick analysis...")
    asyncio.run(fetch_deriv_data())
else:
    st.warning("Click 'Start Live Stream' in the sidebar to initiate real-time market analysis.")
