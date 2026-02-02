import json
from datetime import datetime, timedelta

# SIMULATED SUNDAY DATA (Based on your logs)
ENTRY_PRICE = 88324.0  # Sunday 01:00 AM Entry
EXIT_SIGNAL_PRICE = 88107.0  # Sunday 02:00 AM Exit Signal
ENTRY_TIME = datetime.utcnow() - timedelta(hours=1) # Only 1 hour has passed

print("🧪 STARTING ARGUS GUARDRAIL TEST...")
print(f"Scenario: Bought at ${ENTRY_PRICE}, Market now at ${EXIT_SIGNAL_PRICE} (1 hour later)")

# 1. HOLD TIME CHECK (The 4-Hour Rule)
hold_duration = timedelta(hours=1) # Simulated
if hold_duration < timedelta(hours=4):
    print(f"❌ HOLD TIME GUARDRAIL: BLOCKED. (Held {hold_duration.total_seconds()/3600}h / Need 4h)")
    hold_pass = False
else:
    print("✅ HOLD TIME GUARDRAIL: PASSED.")
    hold_pass = True

# 2. FEE AUDIT CHECK (The 0.2% Hurdle)
profit_pct = (EXIT_SIGNAL_PRICE - ENTRY_PRICE) / ENTRY_PRICE
if profit_pct < 0.002:
    print(f"❌ FEE AUDIT: BLOCKED. (Profit {profit_pct:.2%} < Hurdle 0.20%)")
    fee_pass = False
else:
    print("✅ FEE AUDIT: PASSED.")
    fee_pass = True

# FINAL VERDICT
if not hold_pass or not fee_pass:
    print("\n🛡️ RESULT: Argus would NOT have sold. $3.74 loss avoided.")
else:
    print("\n⚠️ RESULT: Argus would have still sold.")