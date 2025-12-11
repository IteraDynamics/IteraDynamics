import time
import subprocess
import requests
from datetime import datetime
import sys
if sys.platform == "win32":
    import os
    # Set console encoding to UTF-8 for clean printing of special characters
    os.system('chcp 65001')

# --- CONFIGURATION ---
INTERVAL = 60 
SYMBOL = "bitcoin"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open("overnight_session.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_live_price_display():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": SYMBOL, "vs_currencies": "usd"}
        resp = requests.get(url, params=params, timeout=5)
        data = resp.json()
        return float(data[SYMBOL]["usd"])
    except:
        return 0.0

def run_argus():
    try:
        env = os.environ.copy()
        env["MW_INFER_LIVE"] = "1"
        
        result = subprocess.run(
            [sys.executable, "apex_core/signal_generator.py"],
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8"
        )
        
        # --- FILTER NOISE ---
        output_lines = result.stdout.split("\n")
        relevant_lines = [
            line for line in output_lines 
            if ">>" in line 
            or "Regime:" in line 
            or "PORTFOLIO" in line 
            or "EXECUTING" in line
        ]
        
        if relevant_lines:
            for line in relevant_lines:
                log(f"   {line.strip()}")
        
        if result.stderr:
            clean_err = result.stderr.strip()
            # Filter out the harmless INFO logs
            if clean_err and "INFO" not in clean_err:
                log(f"   [STDERR] ⚠️ {clean_err}")
                
    except Exception as e:
        log(f"❌ Execution Error: {e}")

if __name__ == "__main__":
    log("=== 🚀 Itera Dynamics: LIVE ALPHA (CLEAN DASHBOARD) ===")
    
    try:
        while True:
            price = get_live_price_display()
            if price > 0:
                log(f"Tick: ${price:,.2f}")
            run_argus()
            time.sleep(INTERVAL)
            
    except KeyboardInterrupt:
        log("=== Session Ended by User ===")