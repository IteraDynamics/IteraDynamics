import time
import subprocess
import requests
from datetime import datetime, timedelta
import sys
import os

if sys.platform == "win32":
    import os
    os.system('chcp 65001')

# --- CONFIGURATION ---
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
    log("🚀 EXECUTION WINDOW OPEN: Waking Argus...")
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
        else:
            # If no relevant output, show raw to debug
            log(f"   [RAW OUTPUT] {result.stdout[:200]}...")
        
        if result.stderr:
            clean_err = result.stderr.strip()
            if clean_err and "INFO" not in clean_err:
                log(f"   [STDERR] ⚠️ {clean_err}")
                
    except Exception as e:
        log(f"❌ Execution Error: {e}")
    
    log("💤 Argus cycle complete. Sleeping...")

if __name__ == "__main__":
    log("=== 🚀 Itera Dynamics: LIVE HOURLY SCHEDULER ===")
    log("    Mode: Hourly Swing (Execution at XX:00)")
    
    last_run_hour = -1

    try:
        while True:
            now = datetime.now()
            price = get_live_price_display()
            
            # 1. TRIGGER LOGIC (Top of the Hour)
            # We check if minute is 0 AND we haven't run this hour yet
            if now.minute == 0 and now.hour != last_run_hour:
                # Wait 10 seconds to ensure candle closes on the exchange
                time.sleep(10) 
                run_argus()
                last_run_hour = now.hour
                
            # 2. HEARTBEAT (Every ~60 seconds)
            else:
                # Calculate time to next signal
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                minutes_left = (next_hour - now).total_seconds() / 60
                
                if price > 0:
                    print(f"[{now.strftime('%H:%M')}] Tick: ${price:,.2f} | ⏳ Signal in {int(minutes_left)}m")
            
            # Sleep 30s to stay responsive but not burn CPU
            time.sleep(30)
            
    except KeyboardInterrupt:
        log("=== Session Ended by User ===")