import json
import re
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
LOG_FILE = Path("overnight_session.log")
DB_FILE = Path("paper_state.json")

def parse_log_line(line):
    """
    Extracts trade data from log lines like:
    [2025-12-12 13:00:42]    >> [EXECUTING] BUY 0.055532 BTC ($5000.00)
    """
    # Regex to capture: Timestamp, Action, Amount, Cost
    pattern = r"\[(.*?)\]\s+>> \[EXECUTING\] (BUY|SELL) ([\d\.]+) BTC \(\$([\d\.]+)\)"
    match = re.search(pattern, line)
    
    if match:
        ts_str, action, amount_str, cost_str = match.groups()
        amount = float(amount_str)
        cost = float(cost_str)
        
        # Calculate execution price (Cost / Amount)
        price = cost / amount if amount > 0 else 0.0
        
        return {
            "timestamp": ts_str,
            "action": action,
            "price": price,
            "amount": amount,
            "cost": cost,
            "reason": "Reconstructed from Logs"
        }
    return None

def reconstruct():
    if not LOG_FILE.exists():
        print(f"❌ Error: {LOG_FILE} not found.")
        return

    print(f"🔍 Scanning {LOG_FILE} for lost trades...")
    
    restored_trades = []
    
    with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            trade = parse_log_line(line)
            if trade:
                restored_trades.append(trade)
                print(f"   found: {trade['timestamp']} | {trade['action']} {trade['amount']} BTC")

    if not restored_trades:
        print("⚠️ No execution lines found in log.")
        return

    print(f"✅ Extracted {len(restored_trades)} trades.")

    # --- UPDATE THE JSON DATABASE ---
    if DB_FILE.exists():
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"cash": 10000, "positions": 0.0, "trade_log": []}

    # Overwrite the empty log with our restored history
    data["trade_log"] = restored_trades
    
    # Verify totals match your current state (Sanity Check)
    total_btc = sum(t["amount"] for t in restored_trades if t["action"] == "BUY")
    print(f"📊 Reconstructed Total BTC: {total_btc:.6f}")
    
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"💾 Database updated: {DB_FILE}")
    print("🚀 Refresh your Dashboard. PnL should be perfect now.")

if __name__ == "__main__":
    reconstruct()