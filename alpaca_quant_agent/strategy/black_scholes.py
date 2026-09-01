"""Black-Scholes pricing, Greeks, and implied-volatility inversion.

Alpaca's free/indicative options data feed (no OPRA subscription -- the
default for a paper account with no market-data add-on) returns bid/ask
quotes and trade prints, but no greeks or implied volatility fields at all
(confirmed empirically against a live paper account: `get_option_chain` and
`get_option_snapshot` responses carry only dailyBar/latestQuote/latestTrade/
minuteBar/prevDailyBar). Since the entire strategy is delta-targeted and
IV-rank-gated, this module computes both ourselves from quote mid-price,
underlying price, strike, and time-to-expiry -- standard practice when a
data vendor doesn't provide analytics directly.

No external pricing library dependency; this is ~40 lines of textbook
Black-Scholes (Hull), pure functions, unit tested against known closed-form
values.
"""
from __future__ import annotations

import math

# Assumed constant risk-free rate for BS pricing (annualized, e.g. short-term
# T-bill yield proxy). A live risk-free curve is out of scope; this is a
# documented approximation -- see config.yaml::black_scholes.risk_free_rate.
DEFAULT_RISK_FREE_RATE = 0.045


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, sigma must all be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European option price (American-vs-European difference is ignored --
    a standard simplification for equity/ETF options priced for entry
    screening rather than exercise decisions)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1.0


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1.00 (100%) change in vol; same formula for calls and puts.
    Divide by 100 if you want vega per 1 vol *point* (1%)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * math.sqrt(T)


def implied_volatility(
    option_type: str,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = DEFAULT_RISK_FREE_RATE,
    *,
    low: float = 0.005,
    high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Inverts Black-Scholes for implied vol via bisection (robust near
    expiry / deep ITM-OTM where Newton's method can misbehave). Returns
    None if `price` isn't achievable within [low, high] vol -- e.g. a
    stale/crossed quote, or a price outside no-arbitrage bounds -- so
    callers can skip the contract rather than trust a garbage value.
    """
    if T <= 0 or S <= 0 or K <= 0 or price <= 0:
        return None

    def f(sigma: float) -> float:
        return bs_price(option_type, S, K, T, r, sigma) - price

    f_low, f_high = f(low), f(high)
    if f_low > 0 or f_high < 0:
        return None  # price outside the achievable range for any vol in [low, high]

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_mid > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0
