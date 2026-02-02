import json
from pathlib import Path
from datetime import datetime

# Set the path relative to this script
project_root = Path(__file__).resolve().parent

# The state we observed in your 11:00 AM logs
state = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "regime": "🐯 RECOVERY",
    "risk_mult": 0.25,
    "raw_signal": "BUY",
    "conviction_score": 25  # 0.25 * 100
}

# Write the memory file
try:
    with open(project_root / "cortex.json", "w") as f:
        json.dump(state, f)
    print(f"✅ Cortex memory seeded at {project_root / 'cortex.json'}")
except Exception as e:
    print(f"❌ Error: {e}")