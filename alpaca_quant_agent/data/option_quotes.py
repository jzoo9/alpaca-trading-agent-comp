"""Merges Alpaca's contract metadata (strike/expiration/type/open_interest,
from get_option_contracts) with market-data snapshots (bid/ask, from
get_option_chain), and computes implied volatility + delta + vega ourselves
via Black-Scholes -- since neither Alpaca endpoint returns greeks or IV on
the free indicative feed this account has (verified live). Pure function,
no network -- takes plain dicts (as returned by execution/alpaca_mcp.py's
typed wrappers) and a date/price context, so it's unit-testable with
fixtures independent of any live connection.
"""
from __future__ import annotations

from datetime import date, datetime

from alpaca_quant_agent.strategy.black_scholes import DEFAULT_RISK_FREE_RATE, bs_delta, bs_vega, implied_volatility
from alpaca_quant_agent.strategy.screener import OptionQuote


def _parse_date(value: str) -> date:
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def build_option_quotes(
    contracts: dict[str, dict],
    chain: dict[str, dict],
    underlying_price: float,
    today: date,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> list[OptionQuote]:
    """`contracts` and `chain` are both keyed by OCC symbol (as returned by
    AlpacaMcpClient.get_option_contracts / get_option_chain). A contract is
    skipped (not an error) if: it has no matching chain entry, no usable
    bid/ask quote, has already expired, or its IV can't be inverted (e.g. a
    stale/crossed quote) -- these are exactly the contracts the strategy
    shouldn't trade anyway.
    """
    quotes: list[OptionQuote] = []

    for symbol, contract in contracts.items():
        snapshot = chain.get(symbol)
        if snapshot is None:
            continue

        quote = snapshot.get("latestQuote") or {}
        bid = float(quote.get("bp") or 0.0)
        ask = float(quote.get("ap") or 0.0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue

        try:
            expiration = _parse_date(contract["expiration_date"])
            strike = float(contract["strike_price"])
            option_type = contract["type"]
            open_interest = int(contract.get("open_interest") or 0)
        except (KeyError, ValueError, TypeError):
            continue

        days_to_expiry = (expiration - today).days
        if days_to_expiry <= 0:
            continue
        T = days_to_expiry / 365.0

        mid = (bid + ask) / 2.0
        iv = implied_volatility(option_type, mid, underlying_price, strike, T, risk_free_rate)
        if iv is None:
            continue

        delta = bs_delta(option_type, underlying_price, strike, T, risk_free_rate, iv)
        # bs_vega() returns $ sensitivity per 1.00 (100%) absolute change in
        # vol; the conventional "vega" quants report and cap against is per
        # 1 vol *point* (1%) -- divide by 100 to match that convention (this
        # is what risk_gates.portfolio_vega_cap_pct is calibrated against).
        vega = bs_vega(underlying_price, strike, T, risk_free_rate, iv) / 100.0

        quotes.append(
            OptionQuote(
                occ_symbol=symbol,
                underlying=contract.get("underlying_symbol", ""),
                strike=strike,
                expiration=expiration,
                option_type=option_type,
                bid=bid,
                ask=ask,
                delta=delta,
                vega=vega,
                open_interest=open_interest,
                iv=iv,
            )
        )

    return quotes
