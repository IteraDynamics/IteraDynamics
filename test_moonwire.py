import sys
sys.path.insert(0, r'./runtime/argus')
import os
import pandas as pd
from runtime.argus.research.strategies.sg_moonwire_intent_v1 import generate_intent

# Set env vars
os.environ['MOONWIRE_SIGNAL_FILE'] = r'C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend\feeds\btc_overlap_test.jsonl'
os.environ['MOONWIRE_LONG_THRESH'] = '0.55'
os.environ['MOONWIRE_REQUIRE_EXACT_TS'] = '1'

# Load a sample of your BTC data from Dec 4
df = pd.read_csv('data/btcusd_3600s_2019-01-01_to_2025-12-30.csv', parse_dates=['Timestamp'], index_col='Timestamp')
df = df.loc['2025-12-04 04:00':'2025-12-04 04:00']  # Test the 0.5746 signal


print('Test DataFrame:')
print(df.head())
print('\nTrying to generate intent...')

try:
    intent = generate_intent(df, ctx=None)
    print(f'Success! Action: {intent.action}, Confidence: {intent.confidence}, Reason: {intent.reason}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()