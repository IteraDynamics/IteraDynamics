# run_live.py
# 🦅 ARGUS LIVE SCHEDULER - V2.0 (LOGGING ENABLED)

import time
import schedule
import sys
import os
from datetime import datetime
from apex_core.signal_generator import generate_signals

# --- 🔧 LOGGER CLASS ---
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("argus.log", "a", encoding="utf-8") # Append mode

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure dashboard sees it immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout and stderr to the logger
sys.stdout = Logger()
sys.stderr = sys.stdout

def job():
    """ The Hourly Mission """
    # 1. VISUAL SPACER
    print(f"\n[2025-12-17 {datetime.now().strftime('%H:%M:%S')}] 🚀 EXECUTION WINDOW OPEN: Waking Argus...")
    
    # 2. RUN BRAIN
    try:
        generate_signals()
    except Exception as e:
        print(f"❌ CRITICAL SCHEDULER ERROR: {e}")
    
    # 3. SLEEP MESSAGE
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💤 Argus cycle complete. Sleeping...")
    
    # 4. NEXT EXECUTION TIME
    next_run = schedule.next_run()
    if next_run:
        wait_mins = (next_run - datetime.now()).total_seconds() / 60
        print(f"[{datetime.now().strftime('%H:%M')}] ⏳ Next Signal in {int(wait_mins)} min...")

# --- SCHEDULE ---
# Run every hour at :00 minutes
schedule.every().hour.at(":00").do(job)

if __name__ == "__main__":
    print("🦅 ARGUS LIVE SCHEDULER ONLINE")
    print("--------------------------------")
    print(f"⏰ System Time: {datetime.now().strftime('%H:%M:%S')}")
    print("Waiting for next hourly slot...")
    
    # Run once immediately on launch for verification?
    # Uncomment next line if you want an instant check
    # job() 
    
    while True:
        schedule.run_pending()
        time.sleep(1)