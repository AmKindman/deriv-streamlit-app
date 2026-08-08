import streamlit as st
import pandas as pd
import numpy as np
import json
from websocket import create_connection
from collections import deque

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Live Deriv Market Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Deriv Market Predictor")
st.caption("Real-Time Quantitative Momentum Analysis using Deriv WebSocket API")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Settings")
symbol_map = {
    "Volatility 100 (1s) Index": "1HZ100V",
    "Volatility 75 (1s) Index": "1HZ75V",
    "Volatility 50 (1s) Index": "1HZ50V",
    "Volatility 25 (1s) Index": "1HZ25V",
    "Volatility 10 (1s) Index": "1HZ10V"
}

selected_label = st.sidebar.selectbox("Select Synthetic Index", list(symbol_map.keys()))
symbol = symbol_map[selected_label]

fast_span = st.sidebar.slider("Fast EMA Span", min_value=3, max_value=15, value=5)
slow_span = st.sidebar.slider("Slow EMA Span", min_value=10, max_value=50, value=20)
confidence_threshold = st.sidebar.slider("Signal Threshold (%)", min_value=10, max_value=80, value=30)

# Toggle streaming state
if "streaming" not in st.session_state:
    st.session_state.streaming = False

col_start, col_stop = st.sidebar.columns(2)
if col_start.button("▶ Start", type="primary"):
    st.session_state.streaming = True
if col_stop.button("⏹ Stop"):
    st.session_state.streaming = False

# --- SESSION STATE BUFFERS ---
if "ticks" not in st.session_state:
    st.session_state.ticks = deque(maxlen=60)

if "ws" not in st.session_state:
    st.session_state.ws = None

# --- PREDICTIVE ENGINE ---
def analyze_ticks(ticks_list):
    df = pd.DataFrame(ticks_list)
    if len(df) < slow_span:
        return {
            "prediction": f"BUILDING BUFFER ({len(df)}/{slow_span})",
            "confidence": 0.0,
            "fast_ema": 0,
            "slow_ema": 0,
            "latest": df['price'].iloc[-1] if not df.empty else 0
        }
    
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

# --- DASHBOARD METRICS DISPLAY ---
res = analyze_ticks(st.session_state.ticks)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Latest Price", f"{res['latest']}")
metric_col2.metric("Market Prediction", res['prediction'])
metric_col3.metric("Signal Strength", f"{res['confidence']}%")

# Render Chart if data exists
if len(st.session_state.ticks) > 0:
    df_chart = pd.DataFrame(st.session_state.ticks).set_index("time")
    st.line_chart(df_chart['price'])

# --- REAL-TIME TICK FETCH ENGINE ---
if st.session_state.streaming:
    try:
        # Establish WebSocket connection if not already connected
        if st.session_state.ws is None:
            ws = create_connection("wss://ws.derivws.com/websockets/v3?app_id=1089", timeout=3)
            ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))
            st.session_state.ws = ws

        # Receive single tick
        result = st.session_state.ws.recv()
        data = json.loads(result)
        
        if "tick" in data:
            price = float(data["tick"]["quote"])
            epoch = data["tick"]["epoch"]
            st.session_state.ticks.append({
                "time": pd.to_datetime(epoch, unit='s'),
                "price": price
            })
            
        # Trigger immediate Streamlit UI refresh for next tick
        st.rerun()

    except Exception as e:
        # Reset connection on failure/timeout
        if st.session_state.ws:
            try:
                st.session_state.ws.close()
            except:
                pass
        st.session_state.ws = None
        st.warning("Reconnecting to live data stream...")
        st.rerun()
else:
    # Close socket when stopped
    if st.session_state.ws is not None:
        try:
            st.session_state.ws.close()
        except:
            pass
        st.session_state.ws = None
    st.info("Click **▶ Start** in the sidebar to stream live ticks.")
