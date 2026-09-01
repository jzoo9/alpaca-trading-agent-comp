"""Builds a full per-symbol market snapshot (regime signal, option chain,
IV rank, days-to-earnings) that strategy/screener.py consumes. This is the
one place that talks to AlpacaMcpClient for market data, so a response-shape
change in the MCP server only needs fixing here (see also
data/option_quotes.py, which merges contract metadata + market-data quotes
into priced OptionQuote objects).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from alpaca_quant_agent.data import earnings_calendar, iv_history
from alpaca_quant_agent.data.option_quotes import build_option_quotes
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient
from alpaca_quant_agent.strategy.black_scholes import DEFAULT_RISK_FREE_RATE
from alpaca_quant_agent.strategy.screener import OptionQuote, select_expiration
from alpaca_quant_agent.strategy.signals import RegimeSignal, classify_regime, iv_rank

# get_option_contracts (metadata: strike/expiration/OI) supports up to
# 10,000 results per its own schema; get_option_chain (market-data quotes)
# caps at 1,000. On a daily-expiring underlying like SPY, a full 45-day
# window can hold thousands of contracts -- more than the chain endpoint
# allows -- so contracts are fetched broadly first (to discover all
# available expirations reliably) and the chain/quotes call is then scoped
# down to just the specific target expiration(s), which comfortably fits
# under 1,000 even at a generous strike band.
_CONTRACTS_LIMIT = 10000
_CHAIN_LIMIT = 1000


@dataclass(frozen=True)
class SymbolSnapshot:
    symbol: str
    regime: RegimeSignal
    chain: list[OptionQuote]
    iv_rank_value: float
    days_to_earnings: int | None
    underlying_price: float


def _bars_to_series(bars: list[dict]) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = pd.Series([b["h"] for b in bars])
    low = pd.Series([b["l"] for b in bars])
    close = pd.Series([b["c"] for b in bars])
    return high, low, close


def _atm_iv(chain: list[OptionQuote], underlying_price: float) -> float:
    calls = [q for q in chain if q.option_type == "call" and q.iv > 0]
    if not calls:
        return 0.0
    nearest = min(calls, key=lambda q: abs(q.strike - underlying_price))
    return nearest.iv


async def build_symbol_snapshot(
    client: AlpacaMcpClient,
    db_path: str,
    symbol: str,
    today: date,
    config: dict,
) -> SymbolSnapshot | None:
    signals_cfg = config["signals"]
    calendar_days_needed = signals_cfg["momentum_lookback_days"] + signals_cfg["momentum_skip_days"] + 10
    # ~1.6x buffer converts trading days needed to calendar days (weekends/holidays).
    start = (today - timedelta(days=int(calendar_days_needed * 1.6))).isoformat()
    end = today.isoformat()

    bars = await client.get_stock_bars(symbol, "1Day", start, end)
    if len(bars) < signals_cfg["ema_slow"]:
        return None  # not enough history yet to compute regime signals safely

    high, low, close = _bars_to_series(bars)
    underlying_price = float(close.iloc[-1])

    regime = classify_regime(
        high, low, close,
        ema_fast=signals_cfg["ema_fast"],
        ema_slow=signals_cfg["ema_slow"],
        adx_period=signals_cfg["adx_period"],
        adx_trend_threshold=signals_cfg["adx_trend_threshold"],
        momentum_lookback_days=signals_cfg["momentum_lookback_days"],
        momentum_skip_days=signals_cfg["momentum_skip_days"],
    )

    # Widened to cover both sleeves in one discovery pull: Sleeve B can want
    # expirations just 1-2 days out (earnings-adjacent), Sleeve A wants
    # target_dte_max. +5 buffer keeps select_expiration() able to pick the
    # closest-to-target date even if it lands slightly past the configured max.
    sleeve_a_cfg = config["strategy"]["sleeve_a"]
    expiration_gte = (today + timedelta(days=1)).isoformat()
    expiration_lte = (today + timedelta(days=sleeve_a_cfg["target_dte_max"] + 5)).isoformat()

    # Restrict to strikes near the money -- comfortably covers a
    # ~0.20-0.25 delta short strike while keeping request sizes bounded.
    strike_band_pct = signals_cfg.get("option_strike_band_pct", 0.25)
    strike_gte = underlying_price * (1 - strike_band_pct)
    strike_lte = underlying_price * (1 + strike_band_pct)

    contracts = await client.get_option_contracts(
        symbol, expiration_gte, expiration_lte, strike_gte, strike_lte, limit=_CONTRACTS_LIMIT
    )

    available_expirations = sorted(
        {datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() for c in contracts.values()}
    )
    target_expirations: set[date] = set()

    sleeve_a_expiration = select_expiration(
        available_expirations, today, sleeve_a_cfg["target_dte_min"], sleeve_a_cfg["target_dte_max"]
    )
    if sleeve_a_expiration is not None:
        target_expirations.add(sleeve_a_expiration)

    dte_earnings = earnings_calendar.days_to_earnings(symbol, today)
    if dte_earnings is not None:
        earnings_date = today + timedelta(days=dte_earnings)
        post_earnings = [e for e in available_expirations if e >= earnings_date]
        if post_earnings:
            target_expirations.add(min(post_earnings))

    if target_expirations:
        chain_gte = min(target_expirations).isoformat()
        chain_lte = max(target_expirations).isoformat()
        raw_chain = await client.get_option_chain(
            symbol, chain_gte, chain_lte, strike_gte, strike_lte, limit=_CHAIN_LIMIT
        )
        relevant_contracts = {
            sym: c for sym, c in contracts.items()
            if datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() in target_expirations
        }
        risk_free_rate = config.get("black_scholes", {}).get("risk_free_rate", DEFAULT_RISK_FREE_RATE)
        chain = build_option_quotes(relevant_contracts, raw_chain, underlying_price, today, risk_free_rate)
    else:
        chain = []

    iv_history.bootstrap_from_realized_vol(db_path, symbol, close, today)
    atm_iv = _atm_iv(chain, underlying_price)
    history = iv_history.history_for_rank(db_path, symbol, today, signals_cfg["iv_rank_lookback_days"])
    rank = iv_rank(history, atm_iv)
    if atm_iv > 0:
        iv_history.record_observation(db_path, symbol, today, atm_iv)

    dte = earnings_calendar.days_to_earnings(symbol, today)

    return SymbolSnapshot(
        symbol=symbol,
        regime=regime,
        chain=chain,
        iv_rank_value=rank,
        days_to_earnings=dte,
        underlying_price=underlying_price,
    )
