# Itera ↔ MoonWire feed (consumer only)

**Product boundary:** MoonWire owns all intelligence (features, model, inference, export). Itera only consumes the signal feed and may orchestrate ensuring a feed exists by calling moonwire-backend as an external process.

## How Itera ensures a MoonWire feed exists

1. **Check** — Does `MOONWIRE_SIGNAL_FILE` exist?
2. **Validate** — Schema (timestamp, probability per line) and optional freshness (max age).
3. **If missing or stale** — Call `moonwire-backend/scripts/export_signal_feed.py` via **subprocess** (no MoonWire code imported into Itera).
4. **If exists and valid/fresh** — Do nothing; return path.

No inference or signal generation runs in Itera.

## How to point Itera to moonwire-backend

Set the path to the MoonWire repo so the ensure script can invoke its export:

- **Env:** `MOONWIRE_BACKEND_ROOT=C:\path\to\moonwire-backend`
- **CLI:** `--backend-root C:\path\to\moonwire-backend`

Also set the target feed path (where Itera expects the JSONL):

- **Env:** `MOONWIRE_SIGNAL_FILE=C:\path\to\feeds\btc_signals.jsonl`

## Example PowerShell (Windows)

```powershell
# From Itera repo root

# 1) Where the feed file should live (Itera will read this)
$env:MOONWIRE_SIGNAL_FILE = "C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend\feeds\btc_signals.jsonl"

# 2) Where moonwire-backend repo is (so ensure script can call export_signal_feed.py)
$env:MOONWIRE_BACKEND_ROOT = "C:\Users\admin\OneDrive\Desktop\Desktop\moonwire-backend"

# 3) Ensure feed exists (check → validate → call backend if needed)
python scripts/ensure_moonwire_signal_feed.py

# 4) Optional: derive date range from your backtest CSV
python scripts/ensure_moonwire_signal_feed.py --csv data/btcusd_3600s_2019-01-01_to_2025-12-30.csv
```

If the feed is missing or stale and `MOONWIRE_BACKEND_ROOT` is set, the script runs `python scripts/export_signal_feed.py ...` inside the moonwire-backend directory and copies the result to `MOONWIRE_SIGNAL_FILE`.

## Optional freshness

To treat the feed as stale after a certain age (e.g. re-export daily):

```powershell
$env:MOONWIRE_FEED_MAX_AGE_SECONDS = "86400"   # 24 hours
python scripts/ensure_moonwire_signal_feed.py
```

## Programmatic use

From Python (with `runtime/argus` on path):

```python
from runtime.argus.integrations.moonwire_feed import ensure_feed, validate_schema, is_fresh

ok, path = ensure_feed(
    signal_file=os.environ.get("MOONWIRE_SIGNAL_FILE"),
    backend_root=os.environ.get("MOONWIRE_BACKEND_ROOT"),
    max_age_seconds=86400,
)
if ok:
    print("Feed ready:", path)
```

## Related

- **MoonWire producer:** `moonwire-backend/scripts/export_signal_feed.py`
- **Integration guide:** `docs/MOONWIRE_INTEGRATION.md`
- **Orchestration module:** `runtime/argus/integrations/moonwire_feed.py`
