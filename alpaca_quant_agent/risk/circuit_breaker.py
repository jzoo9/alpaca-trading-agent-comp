"""Cycle-level circuit breaker: decides, before any candidates are even
generated, whether this cycle should (a) halt new entries, (b) drop to
manage-existing-positions-only, and/or (c) trigger the protective delta/vega
hedge overlay. Complements the per-trade checks in risk/gates.py, which
re-validate at the individual-order level.
"""
from __future__ import annotations

from dataclasses import dataclass

from alpaca_quant_agent.risk.gates import PortfolioState


@dataclass(frozen=True)
class CircuitBreakerState:
    halt_new_entries: bool
    manage_only: bool
    hedge_needed: bool
    reasons: tuple[str, ...]


def evaluate(portfolio: PortfolioState, config: dict) -> CircuitBreakerState:
    limits = config["risk_gates"]
    reasons: list[str] = []

    daily_halt = portfolio.daily_pnl_pct <= limits["daily_loss_halt_pct"]
    if daily_halt:
        reasons.append(
            f"daily P&L {portfolio.daily_pnl_pct:.2%} breached halt threshold "
            f"{limits['daily_loss_halt_pct']:.2%}"
        )

    kill_switch = portfolio.drawdown_from_peak_pct <= limits["total_drawdown_kill_switch_pct"]
    if kill_switch:
        reasons.append(
            f"drawdown from peak {portfolio.drawdown_from_peak_pct:.2%} breached kill switch "
            f"{limits['total_drawdown_kill_switch_pct']:.2%}"
        )

    equity = portfolio.equity
    delta_pct = abs(portfolio.net_delta) / equity if equity > 0 else 0.0
    vega_pct = abs(portfolio.net_vega) / equity if equity > 0 else 0.0
    hedge_needed = (
        delta_pct > limits["portfolio_delta_band_pct"]
        or vega_pct > limits["portfolio_vega_cap_pct"]
    )
    if hedge_needed:
        reasons.append(
            f"portfolio delta/vega band breached (delta {delta_pct:.2%}, vega {vega_pct:.2%}) "
            f"-- protective SPY put hedge indicated"
        )

    return CircuitBreakerState(
        halt_new_entries=daily_halt or kill_switch,
        manage_only=kill_switch,
        hedge_needed=hedge_needed,
        reasons=tuple(reasons),
    )
