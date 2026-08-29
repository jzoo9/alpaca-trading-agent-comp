"""Deterministic, unit-tested pre-trade risk gates.

These are pure functions over plain dataclasses -- no network, no LLM calls.
The agent's LLM layer (agent/brain.py) only ever sees candidates that have
already passed every gate here; it cannot place an order that wasn't first
approved by this module. This is the enforcement boundary described in the
plan's "LLM's bounded role" section.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProposedTrade:
    symbol: str
    sleeve: str  # "A" | "B"
    strategy_type: str  # "put_credit_spread" | "call_credit_spread" | "iron_condor"
    contracts: int
    max_loss_per_contract: float  # dollars, per contract (already x100)
    credit_per_contract: float  # dollars, per contract (already x100)
    net_delta_per_contract: float
    net_vega_per_contract: float
    open_interest: int
    bid_ask_spread_pct: float
    days_to_earnings: int | None  # None if no known upcoming earnings

    @property
    def total_max_loss(self) -> float:
        return self.contracts * self.max_loss_per_contract

    @property
    def total_delta(self) -> float:
        return self.contracts * self.net_delta_per_contract

    @property
    def total_vega(self) -> float:
        return self.contracts * self.net_vega_per_contract


@dataclass(frozen=True)
class OpenPosition:
    symbol: str
    sleeve: str
    max_loss: float
    net_delta: float
    net_vega: float


@dataclass(frozen=True)
class PortfolioState:
    equity: float
    equity_peak: float
    daily_pnl_pct: float  # e.g. -0.02 == down 2% today
    open_positions: tuple[OpenPosition, ...] = field(default_factory=tuple)

    @property
    def portfolio_heat(self) -> float:
        return sum(p.max_loss for p in self.open_positions)

    @property
    def net_delta(self) -> float:
        return sum(p.net_delta for p in self.open_positions)

    @property
    def net_vega(self) -> float:
        return sum(p.net_vega for p in self.open_positions)

    @property
    def sleeve_b_heat(self) -> float:
        return sum(p.max_loss for p in self.open_positions if p.sleeve == "B")

    def positions_in(self, symbol: str) -> int:
        return sum(1 for p in self.open_positions if p.symbol == symbol)

    @property
    def total_positions(self) -> int:
        return len(self.open_positions)

    @property
    def drawdown_from_peak_pct(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return (self.equity - self.equity_peak) / self.equity_peak


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def evaluate_gates(trade: ProposedTrade, portfolio: PortfolioState, config: dict) -> list[GateCheck]:
    limits = config["risk_gates"]
    checks: list[GateCheck] = []

    equity = portfolio.equity

    max_risk_pct = limits["max_risk_per_trade_pct"]
    risk_pct = trade.total_max_loss / equity if equity > 0 else float("inf")
    checks.append(GateCheck(
        "max_risk_per_trade",
        risk_pct <= max_risk_pct,
        f"trade risk {risk_pct:.4f} vs cap {max_risk_pct:.4f}",
    ))

    max_heat_pct = limits["max_portfolio_heat_pct"]
    projected_heat_pct = (portfolio.portfolio_heat + trade.total_max_loss) / equity if equity > 0 else float("inf")
    checks.append(GateCheck(
        "max_portfolio_heat",
        projected_heat_pct <= max_heat_pct,
        f"projected heat {projected_heat_pct:.4f} vs cap {max_heat_pct:.4f}",
    ))

    max_per_underlying = limits["max_positions_per_underlying"]
    existing = portfolio.positions_in(trade.symbol)
    checks.append(GateCheck(
        "max_positions_per_underlying",
        existing < max_per_underlying,
        f"{existing} existing positions in {trade.symbol} vs cap {max_per_underlying}",
    ))

    max_total = limits["max_total_positions"]
    checks.append(GateCheck(
        "max_total_positions",
        portfolio.total_positions < max_total,
        f"{portfolio.total_positions} open positions vs cap {max_total}",
    ))

    if trade.sleeve == "B":
        sleeve_b_cap_pct = limits["sleeve_b_max_allocation_pct"]
        projected_sleeve_b_pct = (portfolio.sleeve_b_heat + trade.total_max_loss) / equity if equity > 0 else float("inf")
        checks.append(GateCheck(
            "sleeve_b_allocation_cap",
            projected_sleeve_b_pct <= sleeve_b_cap_pct,
            f"projected sleeve B heat {projected_sleeve_b_pct:.4f} vs cap {sleeve_b_cap_pct:.4f}",
        ))

    delta_band_pct = limits["portfolio_delta_band_pct"]
    projected_delta_pct = abs(portfolio.net_delta + trade.total_delta) / equity if equity > 0 else float("inf")
    checks.append(GateCheck(
        "portfolio_delta_band",
        projected_delta_pct <= delta_band_pct,
        f"projected |net delta|/equity {projected_delta_pct:.4f} vs band {delta_band_pct:.4f}",
    ))

    vega_cap_pct = limits["portfolio_vega_cap_pct"]
    projected_vega_pct = abs(portfolio.net_vega + trade.total_vega) / equity if equity > 0 else float("inf")
    checks.append(GateCheck(
        "portfolio_vega_cap",
        projected_vega_pct <= vega_cap_pct,
        f"projected |net vega|/equity {projected_vega_pct:.4f} vs cap {vega_cap_pct:.4f}",
    ))

    min_oi = limits["min_open_interest"]
    checks.append(GateCheck(
        "min_open_interest",
        trade.open_interest >= min_oi,
        f"open interest {trade.open_interest} vs floor {min_oi}",
    ))

    max_spread_pct = limits["max_bid_ask_spread_pct"]
    checks.append(GateCheck(
        "max_bid_ask_spread",
        trade.bid_ask_spread_pct <= max_spread_pct,
        f"bid-ask spread {trade.bid_ask_spread_pct:.4f} vs cap {max_spread_pct:.4f}",
    ))

    if trade.sleeve == "A":
        blackout_days = config["strategy"]["sleeve_a"]["earnings_blackout_days"]
        earnings_ok = trade.days_to_earnings is None or trade.days_to_earnings > blackout_days
        checks.append(GateCheck(
            "earnings_blackout",
            earnings_ok,
            f"days_to_earnings={trade.days_to_earnings} vs blackout {blackout_days}",
        ))

    checks.append(GateCheck(
        "daily_loss_halt",
        portfolio.daily_pnl_pct > limits["daily_loss_halt_pct"],
        f"daily pnl {portfolio.daily_pnl_pct:.4f} vs halt {limits['daily_loss_halt_pct']:.4f}",
    ))

    checks.append(GateCheck(
        "total_drawdown_kill_switch",
        portfolio.drawdown_from_peak_pct > limits["total_drawdown_kill_switch_pct"],
        f"drawdown {portfolio.drawdown_from_peak_pct:.4f} vs kill switch {limits['total_drawdown_kill_switch_pct']:.4f}",
    ))

    return checks


def gate_result(trade: ProposedTrade, portfolio: PortfolioState, config: dict) -> tuple[bool, list[GateCheck]]:
    checks = evaluate_gates(trade, portfolio, config)
    return all(c.passed for c in checks), checks
