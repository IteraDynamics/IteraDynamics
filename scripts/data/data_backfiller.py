import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

# --- Configuration ---
# Target exchange (Coinbase Pro is reliable for historical data)
EXCHANGE_ID = 'coinbase' 
SYMBOL = 'BTC/USD'
TIMEFRAME = '1h'  # <-- CHANGED: Zooming out to Hourly candles
START_DATE = datetime(2020, 1, 1) # <-- CHANGED: Grabbing 4+ years of history (The full Cycle)
OUTPUT_FILE = Path("flight_recorder.csv")
LIMIT = 300 # Max candles per API call (Coinbase limit is usually 300 for 1h)

def fetch_data(exchange, symbol, timeframe, start_timestamp, limit):
    """Fetches a chunk of OHLCV data."""
    try:
        # ccxt expects milliseconds timestamp
        since = int(start_timestamp.timestamp() * 1000)
        
        # ccxt.fetch_ohlcv returns a list of lists: [[timestamp, open, high, low, close, volume], ...]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as e:
        print(f"    [ERROR] Failed to fetch data: {e}")
        return []

def run_backfill():
    print(f"🚀 Itera Dynamics Data Backfiller v1.0")
    print(f"   Target: {SYMBOL} on {EXCHANGE_ID}")
    
    # 1. Initialize Exchange
    try:
        exchange = getattr(ccxt, EXCHANGE_ID)()
    except AttributeError:
        print(f"❌ Error: Exchange ID '{EXCHANGE_ID}' not found.")
        return

    # 2. Get the Start Timestamp
    start_time = START_DATE
    end_time = datetime.now()
    
    print(f"   Fetching history from {start_time.strftime('%Y-%m-%d')} to today...")
    
    all_data = []
    current_time = start_time
    
    # Calculate the timedelta in milliseconds for one minute
    # 1 minute = 60 * 1000 ms
    one_minute_ms = 60 * 1000 
    
    # Calculate how far forward one fetch call advances the cursor (1000 * 1 min)
    fetch_duration_ms = LIMIT * one_minute_ms

    # 3. Main Fetch Loop (Paginating Forward)
    while current_time < end_time:
        print(f"   -> Fetching data starting at {current_time.strftime('%Y-%m-%d %H:%M:%S')}...", end='\r')
        
        # Fetch data chunk
        ohlcv_chunk = fetch_data(exchange, SYMBOL, TIMEFRAME, current_time, LIMIT)
        
        if not ohlcv_chunk:
            # Check if we've hit the end or an error occurred
            if current_time < (end_time - timedelta(hours=1)): 
                print("\n   [WARNING] API returned no data. Possible rate limit or end of available history.")
            break
        
        all_data.extend(ohlcv_chunk)
        
        # Advance the cursor to the time of the *last* candle fetched + 1 minute
        # The timestamp is the first element (index 0) of the last fetched candle
        last_timestamp_ms = ohlcv_chunk[-1][0]
        current_time = datetime.fromtimestamp((last_timestamp_ms + one_minute_ms) / 1000)
        
        # API rate limit check: pause for a moment to prevent rate limiting
        time.sleep(exchange.rateLimit / 1000)


    # 4. Process and Save Data
    if not all_data:
        print("\n❌ Failed to retrieve any historical data.")
        return

    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convert ccxt's millisecond timestamp to seconds and then to datetime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    
    # Rename Close to Price for compatibility with train_real_models.py
    df.rename(columns={'Close': 'Price'}, inplace=True)
    
    print(f"\n   ✅ Total {len(df)} candles downloaded.")
    
    # Save the file (Overwrite or create a new file)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   🎯 Mission Complete. Data saved to {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    run_backfill()