import streamlit as st
import pandas as pd
import json
import re
import time
import requests
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
st.set_page_config(
    page_title="Argus Command Center",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if st.button('🔄 Force Refresh'):
    st.rerun()

# --- PATHS ---
PORTFOLIO_FILE = Path("paper_state.json") 
LOG_FILE = Path("overnight_session.log")
CSV_FILE = Path("flight_recorder.csv")

# --- HELPER: GET REAL-TIME PRICE (API -> LOGS -> CSV) ---
def get_live_price():
    # 1. Try Coinbase API (Real-Time)
    try:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            price = float(data['price'])
            return price, "Coinbase API 🟢"
    except Exception:
        pass # Fallback if API fails

    # 2. Try Logs (Recent Ticks)
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            for line in reversed(lines):
                match = re.search(r"Tick:\s*\$([\d,]+\.?\d*)", line)
                if match:
                    price_str = match.group(1).replace(",", "")
                    return float(price_str), "Log File 🟡"
        except Exception:
            pass

    # 3. Try CSV (Hourly Close)
    if CSV_FILE.exists():
        try:
            df = pd.read_csv(CSV_FILE)
            if not df.empty:
                return df.iloc[-1]['Price'], "Hourly Close 🔴"
        except Exception:
            pass
            
    return 0.0, "Unknown ⚫"

# --- HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🦅 Itera Dynamics | Argus V1.0")

# Status Check
bot_status = "🔴 OFFLINE"
status_color = "red"
last_heartbeat = "Never"

if LOG_FILE.exists():
    mtime = datetime.fromtimestamp(LOG_FILE.stat().st_mtime)
    minutes_ago = (datetime.now() - mtime).total_seconds() / 60
    last_heartbeat = mtime.strftime('%H:%M:%S')
    if minutes_ago < 65: 
        bot_status = "🟢 ONLINE"
        status_color = "green"
    else:
        bot_status = "🟠 STALE"
        status_color = "orange"

with col_head2:
    st.markdown(f"**Status:** :{status_color}[{bot_status}]")
    st.caption(f"Last Heartbeat: {last_heartbeat}")

st.markdown("---")

# --- DATA ENGINE ---
try:
    # 1. GET PRICE
    live_price, source = get_live_price()

    # 2. LOAD PORTFOLIO
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            port = json.load(f)
            
        cash = port.get("cash", 0.0)
        btc_bal = port.get("positions", 0.0)
        equity = cash + (btc_bal * live_price)
        
        # --- CALCULATE P&L ---
        entry_price = 0.0
        pnl_dollar = 0.0
        pnl_pct = 0.0
        
        trade_log = port.get("trade_log", [])
        
        # LOGIC: Find last trade
        if btc_bal > 0 and trade_log:
            last_trade = trade_log[-1]
            raw_action = last_trade.get('action') or last_trade.get('type')
            t_action = str(raw_action).upper()
            
            if t_action == 'BUY':
                entry_price = float(last_trade.get('price', 0.0))
        
        if btc_bal > 0 and entry_price > 0:
            current_val = btc_bal * live_price
            cost_basis = btc_bal * entry_price
            pnl_dollar = current_val - cost_basis
            pnl_pct = (pnl_dollar / cost_basis) * 100

        # --- DISPLAY METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Equity", f"${equity:,.2f}")
        m2.metric("Liquid Cash", f"${cash:,.2f}")
        m3.metric("BTC Position", f"{btc_bal:.6f} BTC", f"${(btc_bal * live_price):,.2f}")
        m4.metric("Bitcoin Price", f"${live_price:,.0f}", delta=None)
        m4.caption(f"Source: {source}")
        
        if btc_bal > 0:
            st.subheader("Active Trade Analysis")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Avg Entry", f"${entry_price:,.2f}")
            
            # Color code the P&L
            pnl_color = "normal" # Streamlit handles green/red automatically for deltas usually, but we force logic here
            t2.metric("Unrealized P&L ($)", f"${pnl_dollar:,.2f}")
            t3.metric("Unrealized P&L (%)", f"{pnl_pct:.2f}%")
            
            if entry_price > 0:
                breakeven = entry_price * 1.002
                dist_to_be = live_price - breakeven
                t4.metric("Break-Even Price", f"${breakeven:,.0f}", f"{dist_to_be:,.0f} to go")

        with st.expander("🔍 Debug Raw Portfolio Data"):
            st.json(port)

    else:
        st.info(f"📍 Waiting for Portfolio Data... (File not found at {PORTFOLIO_FILE})")

    # 3. LIVE LOGS
    st.markdown("---")
    st.subheader("📋 Execution Logs")
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            last_lines = lines[-30:]  
        log_text = "".join(last_lines)
        st.text_area("Terminal Output", log_text, height=300)

except Exception as e:
    st.error(f"Dashboard Crash: {e}")