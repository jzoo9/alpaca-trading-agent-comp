"""Rolling ATM-IV history, persisted in SQLite (alpaca_quant_agent/ledger.py),
used to compute IV Rank (strategy/signals.py::iv_rank). Alpaca's option
snapshot gives current IV per contract but no history, so we accumulate our
own sample day by day as the daemon runs.

Bootstrap problem: on day 1 there is no IV history yet, so iv_rank() would
fall back to a neutral 50th percentile for every symbol -- which would let
Sleeve A's IV-rank entry filter (>=40th pct) pass by default rather than by
genuine timing evidence. `bootstrap_from_realized_vol` addresses this by
seeding a synthetic IV-history proxy from trailing realized volatility
(available immediately from historical price bars), which is a standard
practitioner substitute for a true IV history when one hasn't been
accumulated yet. As real observed IV accumulates, `record_observation`
below organically replaces reliance on this proxy (the realized-vol-derived
points age out of the `lookback_days` window over time).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from alpaca_quant_agent import ledger


def realized_vol_series(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing annualized realized volatility (close-to-close), as a
    same-length series (leading `window` entries are NaN).
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window).std() * np.sqrt(252)


def bootstrap_from_realized_vol(db_path: str, symbol: str, close: pd.Series, today: date, window: int = 20) -> None:
    """Seeds iv_history with a realized-vol proxy for every day we have
    price history for, but only if no real IV observations exist yet for
    this symbol (never overwrites genuine observed IV).
    """
    existing = ledger.iv_history_for(db_path, symbol, lookback_days=1, before=today)
    if existing:
        return  # already have real observations; don't pollute with a stale proxy

    rv = realized_vol_series(close, window=window).dropna()
    if rv.empty:
        return

    dates = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=len(rv))
    for d, v in zip(dates, rv):
        ledger.record_iv_observation(db_path, symbol=symbol, observed_at=d.date(), atm_iv=float(v))


def record_observation(db_path: str, symbol: str, today: date, atm_iv: float) -> None:
    ledger.record_iv_observation(db_path, symbol=symbol, observed_at=today, atm_iv=atm_iv)


def history_for_rank(db_path: str, symbol: str, today: date, lookback_days: int) -> list[float]:
    return ledger.iv_history_for(db_path, symbol, lookback_days=lookback_days, before=today)
