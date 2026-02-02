# sniper/sniper_core.py
# 🔫 BTC SNIPER CORE – REV 2
#
# - Volatility breakout on returns (z-score)
# - Trend filter via fast/slow SMAs
# - ATR-based stop + target
# - Max holding period in hours
# - Proper fee modeling per side
# - Output: equity curve + trade list

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SniperConfig:
    fast_sma: int = 20          # fast trend filter
    slow_sma: int = 100         # slow trend filter
    breakout_zscore: float = 2.0
    lookback_vol: int = 48      # bars for volatility calc
    atr_length: int = 14        # bars for ATR
    stop_atr_mult: float = 1.5  # stop distance in ATR
    target_atr_mult: float = 3.0  # target distance in ATR
    max_hold_hours: int = 72    # max bars to hold (hourly data → hours)
    risk_fraction: float = 0.5  # fraction of equity risked per trade
    fee_bps: float = 10.0       # per side fee in basis points


@dataclass
class SniperTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    qty: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str


def _compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Classic ATR using rolling mean of True Range."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length, min_periods=length).mean()
    return atr


def add_features(df: pd.DataFrame, cfg: SniperConfig) -> pd.DataFrame:
    """
    Add:
      - ret, ret_mean, ret_std, ret_z
      - sma_fast, sma_slow
      - atr
    """
    out = df.copy()

    out["Close"] = out["Close"].astype(float)

    # Volatility breakout on *returns*
    out["ret"] = out["Close"].pct_change()
    out["ret_mean"] = out["ret"].rolling(cfg.lookback_vol, min_periods=cfg.lookback_vol).mean()
    out["ret_std"] = out["ret"].rolling(cfg.lookback_vol, min_periods=cfg.lookback_vol).std()
    out["ret_z"] = (out["ret"] - out["ret_mean"]) / out["ret_std"]

    # Trend filter
    out["sma_fast"] = out["Close"].rolling(cfg.fast_sma, min_periods=cfg.fast_sma).mean()
    out["sma_slow"] = out["Close"].rolling(cfg.slow_sma, min_periods=cfg.slow_sma).mean()

    # ATR for stops/targets
    out["atr"] = _compute_atr(out, cfg.atr_length)

    return out


def run_sniper_backtest(
    df: pd.DataFrame,
    cfg: SniperConfig,
    initial_equity: float = 100.0,
) -> Tuple[pd.DataFrame, List[SniperTrade]]:
    """
    Core engine.

    Assumes:
      - df index is a DatetimeIndex (hourly bars for BTC)
      - columns: Open, High, Low, Close (floats)
      - features from add_features(...) are already present

    Returns:
      - equity_df: index = Timestamp, column = 'equity'
      - trades: list[SniperTrade]
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex (UTC recommended).")

    df = df.sort_index()

    fee_rate = cfg.fee_bps / 10_000.0

    cash = float(initial_equity)
    position_btc = 0.0

    equity_curve = []
    trades: List[SniperTrade] = []

    entry_idx = None
    entry_price = 0.0
    entry_equity = 0.0
    entry_cash_before = 0.0
    qty = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["Close"])
        atr = float(row["atr"]) if not np.isnan(row.get("atr", np.nan)) else np.nan
        sma_fast = float(row["sma_fast"]) if not np.isnan(row.get("sma_fast", np.nan)) else np.nan
        sma_slow = float(row["sma_slow"]) if not np.isnan(row.get("sma_slow", np.nan)) else np.nan
        ret_z = float(row["ret_z"]) if not np.isnan(row.get("ret_z", np.nan)) else np.nan

        # Mark-to-market
        position_value = position_btc * price
        equity = cash + position_value
        equity_curve.append((ts, equity))

        in_position = position_btc > 0.0

        # =====================
        # EXIT LOGIC
        # =====================
        if in_position:
            bars_held = i - entry_idx

            # Define stop & target levels
            if np.isnan(atr) or atr <= 0:
                # Fallback if ATR not ready: simple 1% stop / 2% target
                stop_price = entry_price * (1.0 - 0.01)
                target_price = entry_price * (1.0 + 0.02)
            else:
                stop_price = entry_price - cfg.stop_atr_mult * atr
                target_price = entry_price + cfg.target_atr_mult * atr

            exit_reason = None
            if price <= stop_price:
                exit_reason = "stop"
            elif price >= target_price:
                exit_reason = "target"
            elif bars_held >= cfg.max_hold_hours:
                exit_reason = "time"

            if exit_reason is not None:
                # Close position at market
                gross_proceeds = position_btc * price
                fee_exit = gross_proceeds * fee_rate
                net_proceeds = gross_proceeds - fee_exit

                cash += net_proceeds
                position_btc = 0.0

                # P&L accounting
                net_pnl = cash - entry_cash_before
                gross_pnl = (price - entry_price) * qty
                equity_after = cash  # flat now
                return_pct = (equity_after - entry_equity) / entry_equity if entry_equity > 0 else 0.0

                trades.append(
                    SniperTrade(
                        entry_time=df.index[entry_idx],
                        exit_time=ts,
                        side="LONG",
                        qty=float(qty),
                        entry_price=float(entry_price),
                        exit_price=float(price),
                        gross_pnl=float(gross_pnl),
                        net_pnl=float(net_pnl),
                        return_pct=float(return_pct),
                        bars_held=int(bars_held),
                        exit_reason=exit_reason,
                    )
                )

                # Reset position state
                entry_idx = None
                qty = 0.0
                entry_price = 0.0
                entry_equity = 0.0
                entry_cash_before = 0.0

                # After exit this bar, don't re-enter on same bar
                continue

        # =====================
        # ENTRY LOGIC (flat only)
        # =====================
        if not in_position:
            if (
                not np.isnan(sma_fast)
                and not np.isnan(sma_slow)
                and not np.isnan(ret_z)
                and sma_fast > sma_slow         # only long in uptrend
                and ret_z >= cfg.breakout_zscore  # volatility spike
            ):
                capital_to_use = equity * cfg.risk_fraction
                if capital_to_use > 0:
                    cost = capital_to_use
                    fee_entry = cost * fee_rate
                    total_cash_out = cost + fee_entry
                    qty_new = cost / price

                    # Risk guard: don't enter if we don't have that cash
                    if qty_new > 0 and total_cash_out <= cash:
                        entry_idx = i
                        entry_price = price
                        entry_equity = equity
                        entry_cash_before = cash
                        qty = qty_new

                        cash -= total_cash_out
                        position_btc += qty_new

    # =====================
    # Force-close at end if still in position
    # =====================
    if position_btc > 0.0 and entry_idx is not None:
        last_ts = df.index[-1]
        last_price = float(df["Close"].iloc[-1])
        gross_proceeds = position_btc * last_price
        fee_exit = gross_proceeds * fee_rate
        net_proceeds = gross_proceeds - fee_exit
        cash += net_proceeds

        equity_after = cash
        net_pnl = cash - entry_cash_before
        gross_pnl = (last_price - entry_price) * qty
        return_pct = (equity_after - entry_equity) / entry_equity if entry_equity > 0 else 0.0

        trades.append(
            SniperTrade(
                entry_time=df.index[entry_idx],
                exit_time=last_ts,
                side="LONG",
                qty=float(qty),
                entry_price=float(entry_price),
                exit_price=float(last_price),
                gross_pnl=float(gross_pnl),
                net_pnl=float(net_pnl),
                return_pct=float(return_pct),
                bars_held=int(len(df) - entry_idx - 1),
                exit_reason="eod",
            )
        )

    equity_df = pd.DataFrame(equity_curve, columns=["Timestamp", "equity"]).set_index("Timestamp")
    return equity_df, trades
