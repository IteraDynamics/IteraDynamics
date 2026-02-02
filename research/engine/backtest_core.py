from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    fee_bps: float = 6.0          # 6 bps = 0.06% (tune to your venue)
    slippage_bps: float = 10.0    # 10 bps = 0.10% (conservative baseline)
    initial_cash: float = 100.0
    trade_on_close: bool = True   # enter/exit at close of signal bar (simple, no lookahead)
    min_notional_usd: float = 5.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: Dict[str, Any]


def load_flight_recorder(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect: Timestamp, Open, High, Low, Close, Volume
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Coerce numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)
    return df


def _apply_cost(price: float, side: str, fee_bps: float, slippage_bps: float) -> float:
    """
    Returns effective fill price including slippage.
    Fees are applied separately on notional.
    """
    slip = slippage_bps / 10_000.0
    if side.upper() == "BUY":
        return price * (1.0 + slip)
    return price * (1.0 - slip)


def _fee_amount(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10_000.0)


def compute_summary(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict[str, Any]:
    eq = equity_curve["equity"].values
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    total_return = (eq[-1] / eq[0]) - 1.0 if len(eq) > 1 else 0.0

    # Max drawdown on equity
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12)) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    # “Sharpe-ish” on per-bar returns (not annualized)
    sharpeish = float(rets.mean() / (rets.std() + 1e-12)) if len(rets) else 0.0

    # Trade stats
    if trades.empty:
        win_rate = 0.0
        avg_trade = 0.0
        med_trade = 0.0
        n_trades = 0
        best = 0.0
        worst = 0.0
    else:
        pnl_pct = trades["pnl_pct"].values
        n_trades = int(len(trades))
        win_rate = float((pnl_pct > 0).mean() * 100.0)
        avg_trade = float(pnl_pct.mean() * 100.0)
        med_trade = float(np.median(pnl_pct) * 100.0)
        best = float(pnl_pct.max() * 100.0)
        worst = float(pnl_pct.min() * 100.0)

    return {
        "bars": int(len(equity_curve)),
        "trades": n_trades,
        "win_rate_pct": win_rate,
        "total_return_pct": float(total_return * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "avg_trade_pct": avg_trade,
        "median_trade_pct": med_trade,
        "best_trade_pct": best,
        "worst_trade_pct": worst,
        "sharpeish": sharpeish,
        "final_equity": float(eq[-1]) if len(eq) else 0.0,
    }


def run_backtest_long_only(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    cfg: BacktestConfig,
) -> BacktestResult:
    """
    Minimal long-only, single-position backtester.

    signals columns required:
      - enter_long: bool
      - exit_long: bool

    Assumptions:
      - Trades execute on bar close (trade_on_close=True).
      - Single position (all-in sizing via a sizing signal is handled in strategy if desired).
      - Here we use "fixed fraction of equity" sizing by passing target_notional_usd in signals, optional.
        If not present, we default to 100% of available cash on entry (capped by min_notional).
    """
    df = df.copy().reset_index(drop=True)
    sig = signals.copy().reset_index(drop=True)

    if len(df) != len(sig):
        raise ValueError("signals must align 1:1 with df rows")

    cash = float(cfg.initial_cash)
    btc = 0.0

    rows_trades: List[Dict[str, Any]] = []
    equity_rows: List[Dict[str, Any]] = []

    in_pos = False
    entry_px = None
    entry_ts = None
    entry_notional = None

    for i in range(len(df)):
        ts = df.loc[i, "Timestamp"]
        price = float(df.loc[i, "Close"]) if cfg.trade_on_close else float(df.loc[i, "Open"])

        # Mark equity BEFORE any trade on this bar (standard convention can vary; just be consistent)
        equity = cash + btc * price
        equity_rows.append({"Timestamp": ts, "equity": equity, "cash": cash, "btc": btc, "price": price})

        enter = bool(sig.loc[i, "enter_long"]) if "enter_long" in sig.columns else False
        exit_ = bool(sig.loc[i, "exit_long"]) if "exit_long" in sig.columns else False

        # Exit first (safety): if both true, we prefer exit
        if in_pos and exit_:
            # Sell all
            fill = _apply_cost(price, "SELL", cfg.fee_bps, cfg.slippage_bps)
            notional = btc * fill
            fee = _fee_amount(notional, cfg.fee_bps)

            cash = cash + notional - fee
            btc_sold = btc
            btc = 0.0
            in_pos = False

            # Clean PnL from entry
            exit_notional_net = notional - fee
            entry_notional_net = float(entry_notional) if entry_notional is not None else 0.0
            pnl_usd = exit_notional_net - entry_notional_net
            pnl_pct = (pnl_usd / max(entry_notional_net, 1e-12)) if entry_notional_net > 0 else 0.0

            rows_trades.append(
                {
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "entry_px": entry_px,
                    "exit_px": fill,
                    "btc_qty": btc_sold,
                    "entry_notional_net": entry_notional_net,
                    "exit_notional_net": exit_notional_net,
                    "pnl_usd": pnl_usd,
                    "pnl_pct": pnl_pct,
                }
            )

            entry_px = None
            entry_ts = None
            entry_notional = None

        # Entry
        if (not in_pos) and enter:
            # Determine how much to buy
            target_usd = None
            if "target_usd" in sig.columns:
                v = sig.loc[i, "target_usd"]
                try:
                    target_usd = float(v) if v is not None and not pd.isna(v) else None
                except Exception:
                    target_usd = None

            if target_usd is None:
                target_usd = cash  # default: deploy all cash

            target_usd = min(target_usd, cash)

            if target_usd >= cfg.min_notional_usd:
                fill = _apply_cost(price, "BUY", cfg.fee_bps, cfg.slippage_bps)
                qty = target_usd / fill
                notional = qty * fill
                fee = _fee_amount(notional, cfg.fee_bps)

                # Need cash for notional + fee
                total_cost = notional + fee
                if total_cost > cash:
                    # Scale down qty to fit
                    qty = cash / (fill * (1.0 + cfg.fee_bps / 10_000.0))
                    notional = qty * fill
                    fee = _fee_amount(notional, cfg.fee_bps)
                    total_cost = notional + fee

                if notional >= cfg.min_notional_usd and total_cost <= cash and qty > 0:
                    cash -= total_cost
                    btc += qty
                    in_pos = True

                    entry_px = fill
                    entry_ts = ts
                    entry_notional = notional + fee  # net cost including fee

    trades_df = pd.DataFrame(rows_trades)
    equity_df = pd.DataFrame(equity_rows)

    summary = compute_summary(trades_df, equity_df)
    return BacktestResult(trades=trades_df, equity_curve=equity_df, summary=summary)


def buy_and_hold_baseline(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, Any]:
    """
    Buy at first bar close, hold to last bar close, include fee+slippage on entry and exit.
    """
    if df.empty:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "final_equity": cfg.initial_cash}

    cash = float(cfg.initial_cash)
    p0 = float(df["Close"].iloc[0])
    p1 = float(df["Close"].iloc[-1])

    buy_fill = _apply_cost(p0, "BUY", cfg.fee_bps, cfg.slippage_bps)
    qty = cash / (buy_fill * (1.0 + cfg.fee_bps / 10_000.0))
    buy_notional = qty * buy_fill
    buy_fee = _fee_amount(buy_notional, cfg.fee_bps)
    cash -= (buy_notional + buy_fee)

    # equity curve for DD
    eq = []
    for p in df["Close"].values:
        eq.append(cash + qty * float(p))
    eq = np.array(eq, dtype=float)

    # exit
    sell_fill = _apply_cost(p1, "SELL", cfg.fee_bps, cfg.slippage_bps)
    sell_notional = qty * sell_fill
    sell_fee = _fee_amount(sell_notional, cfg.fee_bps)
    final_equity = cash + sell_notional - sell_fee

    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12)) - 1.0
    max_dd = float(dd.min())

    total_return = (final_equity / cfg.initial_cash) - 1.0

    return {
        "total_return_pct": float(total_return * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "final_equity": float(final_equity),
    }
