# VB Live Dry-Run (sg_volatility_breakout_v1)

No real trades. Uses Coinbase public API for BTC-USD hourly candles only.

## Single cycle (live data)

```powershell
python runtime/argus/run_live_once_volatility_breakout_v1.py --data-store runtime/argus/data/btc_live_dry_run.csv --state vb_state.json --log vb_live_log.jsonl --lookback 200 --cap 1.0
```

## Single cycle (static CSV)

```powershell
python runtime/argus/run_live_once_volatility_breakout_v1.py --csv path/to/btc.csv --state vb_state.json --log vb_live_log.jsonl --lookback 200 --cap 1.0
```

## Repeated loop (live data)

```powershell
python runtime/argus/run_live_loop_volatility_breakout_v1.py --data-store runtime/argus/data/btc_live_dry_run.csv --state vb_state.json --log vb_live_log.jsonl --lookback 200 --cap 1.0 --interval 300
```

Stop with Ctrl+C. `--interval` is seconds between cycles (default 300).

## Duplicate-bar protection

- State file stores `_meta.last_processed_bar_ts` (the UTC timestamp of the last bar we acted on).
- Each cycle: refresh data (if using `--data-store`), then take the **latest closed bar** as the decision bar.
- Before running the strategy, we compare that bar’s timestamp to `last_processed_bar_ts`.
- If they are equal, we **skip**: log one line with `"skipped": true, "reason": "already_processed_same_bar"` and return without changing state or placing any logical “order.”
- So running twice on the same latest candle does not reprocess the bar and does not duplicate entries.
