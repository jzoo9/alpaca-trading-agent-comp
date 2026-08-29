"""Locally-defined, OpenAI-function-calling-style tools -- the LLM's *only*
write surface each cycle.

This is the structural enforcement of the plan's "LLM's bounded role":
the reasoning model is never given the raw Alpaca `place_option_order` tool.
It can only call `submit_approved_trade(candidate_id, rationale)` against a
candidate that the deterministic screener + risk gates already fully
specified (exact legs, strikes, contracts, credit) in strategy/screener.py
and risk/gates.py. This module looks up the pre-computed order server-side --
the model cannot pass its own strikes/qty/price.

`submit_approved_trade` also re-validates the risk gates against the
portfolio *plus whatever else was already executed earlier in this same
cycle*, so multiple approvals in one cycle can't jointly blow through the
portfolio-heat / delta / vega caps even though each candidate individually
passed the gate at screening time.

Each tool is a plain (OpenAI function-calling JSON schema, async handler)
pair -- agent/brain.py's manual tool-calling loop dispatches to these by
name, independent of any particular LLM provider's SDK.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from alpaca_quant_agent import ledger
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient
from alpaca_quant_agent.risk.gates import OpenPosition, PortfolioState, gate_result
from alpaca_quant_agent.strategy.screener import CandidateTrade


@dataclass
class CycleState:
    candidates: dict[str, CandidateTrade]
    portfolio: PortfolioState
    config: dict
    alpaca: AlpacaMcpClient
    db_path: str
    dry_run: bool = False
    executed_ids: set[str] = field(default_factory=set)
    skipped_ids: set[str] = field(default_factory=set)
    executed_trades: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolDef:
    schema: dict[str, Any]  # OpenAI-style {"type": "function", "function": {...}}
    handler: Callable[[dict[str, Any]], Awaitable[str]]


def _candidate_summary(c: CandidateTrade) -> dict[str, Any]:
    return {
        "candidate_id": c.candidate_id,
        "symbol": c.symbol,
        "sleeve": c.sleeve,
        "strategy_type": c.strategy_type,
        "expiration": c.expiration.isoformat(),
        "contracts": c.contracts,
        "credit_per_contract": c.credit_per_contract,
        "max_loss_per_contract": c.max_loss_per_contract,
        "max_loss_total": round(c.contracts * c.max_loss_per_contract, 2),
        "days_to_earnings": c.days_to_earnings,
        "legs": [
            {"occ_symbol": leg.occ_symbol, "side": leg.side, "option_type": leg.option_type, "strike": leg.strike}
            for leg in c.legs
        ],
        "rationale_hint": c.rationale_hint,
    }


def build_quant_tools(state: CycleState) -> dict[str, ToolDef]:
    async def list_candidates(args: dict[str, Any]) -> str:
        summaries = [
            _candidate_summary(c)
            for c in state.candidates.values()
            if c.candidate_id not in state.executed_ids and c.candidate_id not in state.skipped_ids
        ]
        return json.dumps({"candidates": summaries})

    async def get_portfolio_state(args: dict[str, Any]) -> str:
        p = state.portfolio
        return json.dumps(
            {
                "equity": p.equity,
                "equity_peak": p.equity_peak,
                "daily_pnl_pct": p.daily_pnl_pct,
                "open_positions": p.total_positions,
                "portfolio_heat": p.portfolio_heat,
                "net_delta": p.net_delta,
                "net_vega": p.net_vega,
            }
        )

    async def submit_approved_trade(args: dict[str, Any]) -> str:
        candidate_id = args["candidate_id"]
        rationale = args.get("rationale", "")
        candidate = state.candidates.get(candidate_id)

        if candidate is None:
            return json.dumps({"error": f"Unknown candidate_id {candidate_id}"})
        if candidate_id in state.executed_ids:
            return json.dumps({"error": "Already executed this cycle"})

        already_open = list(state.portfolio.open_positions) + [
            OpenPosition(symbol=t["symbol"], sleeve=t["sleeve"], max_loss=t["max_loss"],
                         net_delta=t["net_delta"], net_vega=t["net_vega"])
            for t in state.executed_trades
        ]
        projected_portfolio = PortfolioState(
            equity=state.portfolio.equity,
            equity_peak=state.portfolio.equity_peak,
            daily_pnl_pct=state.portfolio.daily_pnl_pct,
            open_positions=tuple(already_open),
        )
        allowed, checks = gate_result(candidate.to_proposed_trade(), projected_portfolio, state.config)
        if not allowed:
            reasons = [c.detail for c in checks if not c.passed]
            ledger.log_decision(
                state.db_path, candidate_id=candidate_id, symbol=candidate.symbol,
                decision="gate_rejected_at_submit", detail="; ".join(reasons),
            )
            return json.dumps({"error": "Rejected by risk gates", "reasons": reasons})

        legs_payload = [{"symbol": leg.occ_symbol, "side": leg.side, "ratio_qty": 1} for leg in candidate.legs]
        client_order_id = f"vrp-{candidate_id[:8]}"

        proposed = candidate.to_proposed_trade()
        state.executed_ids.add(candidate_id)
        state.executed_trades.append(
            {
                "symbol": candidate.symbol,
                "sleeve": candidate.sleeve,
                "max_loss": proposed.total_max_loss,
                "net_delta": proposed.total_delta,
                "net_vega": proposed.total_vega,
            }
        )

        if state.dry_run:
            ledger.log_decision(
                state.db_path, candidate_id=candidate_id, symbol=candidate.symbol,
                decision="dry_run_would_open",
                detail=f"{candidate.strategy_type} x{candidate.contracts}, credit={candidate.credit_per_contract}, rationale={rationale}",
            )
            return json.dumps({"status": "dry_run", "message": f"would submit {candidate.strategy_type} on {candidate.symbol}, no order placed"})

        order = await state.alpaca.place_option_order(legs_payload, candidate.contracts, client_order_id)
        order_id = order.get("id") if isinstance(order, dict) else None
        ledger.log_trade(
            state.db_path,
            candidate_id=candidate_id,
            position_group=candidate_id,
            symbol=candidate.symbol,
            sleeve=candidate.sleeve,
            strategy_type=candidate.strategy_type,
            action="open",
            contracts=candidate.contracts,
            credit_or_debit=candidate.credit_per_contract,
            max_loss=proposed.total_max_loss,
            net_delta=proposed.total_delta,
            net_vega=proposed.total_vega,
            expiration=candidate.expiration.isoformat(),
            days_to_earnings=candidate.days_to_earnings,
            legs=legs_payload,
            rationale=rationale,
            order_id=order_id,
        )
        return json.dumps({"status": "submitted", "order_id": order_id, "symbol": candidate.symbol, "strategy_type": candidate.strategy_type})

    async def skip_candidate(args: dict[str, Any]) -> str:
        candidate_id = args["candidate_id"]
        reason = args.get("reason", "")
        candidate = state.candidates.get(candidate_id)
        state.skipped_ids.add(candidate_id)
        ledger.log_decision(
            state.db_path, candidate_id=candidate_id,
            symbol=candidate.symbol if candidate else None,
            decision="llm_skipped", detail=reason,
        )
        return json.dumps({"status": "skipped", "candidate_id": candidate_id})

    return {
        "list_candidates": ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": "list_candidates",
                    "description": (
                        "List every risk-gate-approved candidate option trade available this cycle, "
                        "with exact strikes, contracts, credit, and max loss already computed. You may "
                        "only take-or-skip these exactly as given; you cannot alter strikes, quantity, or price."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            handler=list_candidates,
        ),
        "get_portfolio_state": ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": "get_portfolio_state",
                    "description": (
                        "Get current portfolio equity, daily P&L, open position count, portfolio heat "
                        "(dollars at risk), and net delta/vega -- for judging remaining risk budget."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            handler=get_portfolio_state,
        ),
        "submit_approved_trade": ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": "submit_approved_trade",
                    "description": (
                        "Execute exactly one pre-approved candidate trade as-is (no parameter changes "
                        "are possible). Re-validates remaining risk budget server-side before submitting "
                        "the real order to Alpaca, and can reject even an already-approved candidate if "
                        "earlier trades this cycle used up the available risk budget."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "rationale": {"type": "string", "description": "2-3 sentence trade rationale for the ledger."},
                        },
                        "required": ["candidate_id", "rationale"],
                    },
                },
            },
            handler=submit_approved_trade,
        ),
        "skip_candidate": ToolDef(
            schema={
                "type": "function",
                "function": {
                    "name": "skip_candidate",
                    "description": (
                        "Decline a pre-approved candidate this cycle (e.g. a macro/earnings catalyst makes "
                        "it unwise right now) and record why, for the ledger."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_id", "reason"],
                    },
                },
            },
            handler=skip_candidate,
        ),
    }
