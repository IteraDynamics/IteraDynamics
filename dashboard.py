# dashboard.py
# 🦅 ARGUS MISSION CONTROL - V2.3 (CLEAN LOGS FIX)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
import sys
import os

# --- PATH SETUP ---
current_file = Path(__file__).resolve()
project_root = current_file.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

# --- IMPORT LIVE BROKER ---
try:
    from src.real_broker import RealBroker
except ImportError:
    st.error("❌ Could not import RealBroker. Check path.")
    st.stop()

st.set_page_config(
    page_title="Argus Commander",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    
    /* HEADERS: Force H1, H2, H3 to be Pure White */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Monospace';
    }

    /* METRIC VALUES (The Numbers): Neon Green */
    [data-testid="stMetricValue"] {
        font-size: 26px;
        font-family: 'Monospace';
        color: #00ff00;
    }

    /* METRIC LABELS (The Text "Avg Entry"): Pure White */
    div[data-testid="stMetricLabel"] > label {
        color: #ffffff !important; /* Fixed: Was gray, now bright white */
        font-size: 14px;
        font-weight: bold;
    }

    /* LOG BOX: Neon Green & 100% Visible */
    .log-box {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
        color: #00ff00 !important;
        background-color: #000000;
        padding: 15px;
        border: 1px solid #333;
        border-radius: 5px;
        height: 350px;
        overflow-y: scroll;
        white-space: pre-wrap;
    }

    /* UNIVERSAL LOG FIX: Force everything inside log-box to be bright */
    .log-box * {
        color: #00ff00 !important;
        opacity: 1 !important;
        background-color: transparent !important;
        text-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- LIVE DATA FUNCTIONS ---

@st.cache_resource
def get_broker():
    """ Connect to Broker once """
    return RealBroker()

def get_live_price():
    """ 
    🎯 DIRECT HIT to Coinbase Public API for instant price. 
    Bypasses CSV and Broker logic to ensure accuracy.
    """
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        resp = requests.get(url, timeout=3)
        data = resp.json()
        return float(data['data']['amount'])
    except Exception:
        return 0.0

def load_market_data():
    """ Reads CSV for the chart history only """
    csv_path = project_root / "flight_recorder.csv"
    if csv_path.exists():
        try:
            # Handle variable columns if header exists or not
            df = pd.read_csv(csv_path)
            # If columns don't match expected, force rename (safety)
            if 'Close' not in df.columns:
                 df = pd.read_csv(csv_path, names=["Timestamp","Open","High","Low","Close","Volume"], header=0)
            
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.sort_values('Timestamp', inplace=True)
            return df
        except:
            pass
    return pd.DataFrame()

def read_logs():
    """ Robust Log Reader """
    log_path = project_root / "argus.log"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                # Read last 50 lines to keep it snappy
                lines = f.readlines()
                return "".join(lines[-300:])
        except:
            return "Error reading log file."
    return "Waiting for Scheduler to start..."

def get_cortex_state():
    """ Reads the latest brain state from JSON """
    json_path = project_root / "cortex.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except:
            pass
    # Default fallback if no file yet
    return {"conviction_score": 50, "regime": "Waiting...", "risk_mult": 0.0}

# --- MAIN DASHBOARD ---
def main():
    st.title("🦅 ARGUS // LIVE COMMANDER")
    
    # 1. FETCH DATA
    try:
        broker = get_broker()
        cash = broker.cash
        btc_bal = broker.positions
        
        # Live Price from API
        current_price = get_live_price()
        
        # Chart History
        df = load_market_data()

        # Equity Math
        equity = cash + (btc_bal * current_price)
        
        # Sidebar Entry
        avg_entry = st.sidebar.number_input("Avg Entry Price (Cost Basis)", value=90163.00) 
        
        # PnL Math
        if btc_bal > 0:
            unrealized_pnl_usd = (current_price - avg_entry) * btc_bal
            pnl_pct = (unrealized_pnl_usd / (avg_entry * btc_bal)) * 100
            breakeven = avg_entry * 1.006 # 0.6% fees
        else:
            unrealized_pnl_usd = 0.0
            pnl_pct = 0.0
            breakeven = 0.0

    except Exception as e:
        st.error(f"System Error: {e}")
        return

    # --- ROW 1: LIQUID STATUS ---
    st.markdown("### 🏦 Liquid Status")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Liquid Equity", f"${equity:,.2f}", delta=f"{unrealized_pnl_usd:+.2f}")
    k2.metric("Dry Powder (USD)", f"${cash:,.2f}")
    k3.metric("BTC Exposure", f"${(btc_bal * current_price):,.2f}", f"{btc_bal:.6f} BTC")
    k4.metric("Market Price", f"${current_price:,.2f}")

    # --- ROW 2: ACTIVE POSITION ANALYSIS ---
    st.markdown("### 📊 Active Position Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Entry Price", f"${avg_entry:,.2f}")
    m2.metric("Unrealized P&L ($)", f"${unrealized_pnl_usd:+.2f}")
    
    color_mode = "normal" if pnl_pct >= 0 else "inverse"
    m3.metric("Unrealized P&L (%)", f"{pnl_pct:+.2f}%", delta=pnl_pct, delta_color=color_mode)
    
    m4.metric("Breakeven Price", f"${breakeven:,.2f}")

    st.markdown("---")

    # --- ROW 3: CHARTS & GAUGE ---
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.subheader("Performance Curve")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Timestamp'], y=df['Close'], 
                mode='lines', name='BTC', 
                line=dict(color='#00ff00', width=2)
            ))
            fig.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for Flight Recorder Data...")

    with c2:
        st.subheader("Cortex State")
        
        # READ LIVE STATE FROM JSON
        cortex = get_cortex_state()
        score = cortex.get("conviction_score", 50)
        regime_label = cortex.get("regime", "Unknown")

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score, 
            title = {'text': "AI Conviction"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 60], 'color': "gray"},
                    {'range': [60, 100], 'color': "green"}],
            }
        ))
        
        # Add the text label below the gauge
        fig_gauge.add_annotation(
            x=0.5, y=-0.2,
            text=regime_label,
            showarrow=False,
            font=dict(size=12, color="gray")
        )
        
        fig_gauge.update_layout(
            height=300, 
            margin=dict(l=20, r=20, t=50, b=40), 
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- ROW 4: LOGS (FIXED CSS) ---
    st.subheader("📜 System Logs (Live Stream)")
    logs_content = read_logs()
    st.markdown(f"<div class='log-box'>{logs_content}</div>", unsafe_allow_html=True)

    # Auto-Refresh
    time.sleep(10)
    st.rerun()

if __name__ == "__main__":
    main()