"""Deterministic exit rules for open spread/condor positions, evaluated every
cycle. Pure function core (`evaluate_exit`) is unit tested; the surrounding
async functions pull current quotes via AlpacaMcpClient and submit closing
orders for any position that should exit -- no LLM judgment involved, per
the plan (position management is fully automatic; Claude only gets pulled in
if a close can't be executed cleanly, which is handled in agent/brain.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ManagedPosition:
    position_id: str
    symbol: str
    sleeve: str
    strategy_type: str
    contracts: int
    credit_received_per_contract: float  # positive, dollars, x100
    opened_at: date
    expiration: date
    days_to_earnings: int | None = None


@dataclass(frozen=True)
class ExitDecision:
    should_close: bool
    reason: str | None


def evaluate_exit(
    position: ManagedPosition,
    current_value_per_contract: float,  # cost to close the spread now, dollars x100
    today: date,
    config: dict,
) -> ExitDecision:
    """current_value_per_contract is what it would cost to buy back the
    spread right now (i.e. the debit to close). Profit captured so far is
    the drop in value from the credit originally received.
    """
    credit = position.credit_received_per_contract
    if credit <= 0:
        return ExitDecision(False, None)

    profit_captured_pct = (credit - current_value_per_contract) / credit
    loss = current_value_per_contract - credit

    if position.sleeve == "B":
        cfg = config["strategy"]["sleeve_b_earnings"]
        if cfg.get("close_next_session") and position.days_to_earnings is not None and position.days_to_earnings <= 0:
            return ExitDecision(True, "earnings_sleeve_close_next_session")
        return ExitDecision(False, None)

    cfg = config["strategy"]["sleeve_a"]

    if profit_captured_pct >= cfg["take_profit_pct_of_credit"]:
        return ExitDecision(True, "take_profit")

    if loss >= cfg["stop_loss_multiple_of_credit"] * credit:
        return ExitDecision(True, "stop_loss")

    dte = (position.expiration - today).days
    if dte <= cfg["time_stop_dte"]:
        return ExitDecision(True, "time_stop")

    if position.days_to_earnings is not None and position.days_to_earnings <= cfg["earnings_blackout_days"]:
        return ExitDecision(True, "earnings_blackout_close")

    return ExitDecision(False, None)
