"""Protective delta hedge overlay.

`risk/circuit_breaker.py` only *flags* `hedge_needed` when the book's net
delta or net vega drifts outside its band; this module turns that flag into a
concrete, deterministic hedge order and (in cycle.py) executes it. Keeping the
sizing math here as a pure function mirrors risk/gates.py -- no network, no
LLM, fully unit-testable.

Design choice: the delta breach is neutralized with **SPY shares** (each share
has a delta of 1 share-equivalent, precise and always tradeable), sized to pull
net delta from the current level back to zero. A vega breach can't be offset
with shares (shares have no vega), so a vega-only breach is surfaced
(`vega_breach`) for logging / the "buy SPY puts" path noted in WRITEUP.md, but
the automatic overlay here is delta-first: bringing net delta to neutral is
what removes the unwanted directional bet a short-premium book is never
supposed to be making.

Units: `portfolio.net_delta` is a sum of per-position `option_delta * 100`
(screener `_spread_economics`), i.e. *share-equivalents* of the underlying, not
dollars. One share offsets exactly one share-equivalent, so the hedge size is
`round(|net_delta|)` shares directly -- no share price is involved. (This
treats the aggregate book delta as SPY-equivalent, an intentional
approximation for a coarse index hedge.)
"""
from __future__ import annotations

from dataclasses import dataclass

from alpaca_quant_agent.risk.gates import PortfolioState

HEDGE_SYMBOL = "SPY"


@dataclass(frozen=True)
class HedgeOrder:
    needed: bool
    symbol: str
    side: str  # "buy" | "sell"
    shares: int
    reason: str
    delta_breach: bool
    vega_breach: bool


def compute_delta_hedge(portfolio: PortfolioState, config: dict, *, hedge_symbol: str = HEDGE_SYMBOL) -> HedgeOrder:
    """Compute the SPY share order that brings the portfolio's net $-delta back
    inside the configured band.

    `portfolio.net_delta` is a sum of per-position share-equivalents
    (`option_delta * 100`). We size the hedge to offset it back to ~zero:
        shares = round(|net_delta|)
    trading the opposite side of the book's delta sign (positive book delta
    => sell SPY, negative => buy SPY). No share price is needed -- one share
    offsets one share-equivalent.

    Returns a HedgeOrder with needed=False (shares=0) when net delta is already
    inside the band, or when equity is non-positive.
    """
    limits = config["risk_gates"]
    equity = portfolio.equity
    if equity <= 0:
        return HedgeOrder(False, hedge_symbol, "buy", 0, "equity non-positive", False, False)

    delta_band_pct = limits["portfolio_delta_band_pct"]
    vega_cap_pct = limits["portfolio_vega_cap_pct"]

    net_delta = portfolio.net_delta
    delta_pct = abs(net_delta) / equity
    vega_pct = abs(portfolio.net_vega) / equity

    delta_breach = delta_pct > delta_band_pct
    vega_breach = vega_pct > vega_cap_pct

    if not delta_breach:
        reason = (
            f"delta within band ({delta_pct:.4f} <= {delta_band_pct:.4f})"
            + ("; vega breach flagged for put overlay" if vega_breach else "")
        )
        return HedgeOrder(False, hedge_symbol, "buy", 0, reason, delta_breach, vega_breach)

    return HedgeOrder(
        needed=True,
        symbol=hedge_symbol,
        # Book is net long delta => sell SPY to offset; net short => buy SPY.
        side="sell" if net_delta > 0 else "buy",
        shares=hedge_shares(net_delta),
        reason=f"delta breach {delta_pct:.4f} > band {delta_band_pct:.4f}; net_delta={net_delta:.0f}",
        delta_breach=delta_breach,
        vega_breach=vega_breach,
    )


def hedge_shares(net_delta: float) -> int:
    """Number of SPY shares needed to offset `net_delta` share-equivalents,
    rounded to the nearest whole share. One share offsets one share-equivalent,
    so this is just round(|net_delta|) -- no price term."""
    return int(round(abs(net_delta)))
