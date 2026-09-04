"""One full trading cycle: reconcile account state, manage exits on existing
positions (fully deterministic), screen + risk-gate new candidates across the
universe, and hand the survivors to the bounded Claude reasoning layer.
This is what scheduler.py calls every `scheduler.cycle_interval_minutes`.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

from alpaca_quant_agent import control, ledger, universe
from alpaca_quant_agent.agent.brain import run_cycle as run_llm_cycle
from alpaca_quant_agent.agent.tools import CycleState
from alpaca_quant_agent.config import Config
from alpaca_quant_agent.data.snapshot import SymbolSnapshot, build_symbol_snapshot
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient
from alpaca_quant_agent.execution.position_manager import ManagedPosition, evaluate_exit
from alpaca_quant_agent.risk import circuit_breaker
from alpaca_quant_agent.risk.gates import OpenPosition, PortfolioState, gate_result
from alpaca_quant_agent.risk.hedge import compute_delta_hedge
from alpaca_quant_agent.strategy import earnings_sleeve
from alpaca_quant_agent.strategy.signals import vol_term_structure_regime
from alpaca_quant_agent.strategy.screener import (
    CandidateTrade,
    build_credit_spread_candidate,
    build_iron_condor_candidate,
    select_expiration,
)

logger = logging.getLogger(__name__)


def _parse_account(raw: dict) -> tuple[float, float]:
    """(equity, daily_pnl_pct) from Alpaca's account payload (`equity`,
    `last_equity` = prior session's closing equity)."""
    equity = float(raw.get("equity", 0.0))
    last_equity = float(raw.get("last_equity", equity)) or equity
    daily_pnl_pct = (equity - last_equity) / last_equity if last_equity else 0.0
    return equity, daily_pnl_pct


def _load_portfolio_state(db_path: str, equity: float, daily_pnl_pct: float) -> PortfolioState:
    rows = ledger.open_positions(db_path)
    positions = tuple(
        OpenPosition(
            symbol=r["symbol"], sleeve=r["sleeve"],
            max_loss=r["max_loss"] or 0.0, net_delta=r["net_delta"] or 0.0, net_vega=r["net_vega"] or 0.0,
        )
        for r in rows
    )
    equity_peak = max(ledger.latest_equity_peak(db_path, fallback=equity), equity)
    return PortfolioState(equity=equity, equity_peak=equity_peak, daily_pnl_pct=daily_pnl_pct, open_positions=positions)


def _extract_bid_ask(snap: dict) -> tuple[float, float]:
    """get_option_snapshot's real response shape (verified live against
    alpaca-mcp-server 2.3.0): quotes live under `latestQuote.bp`/`.ap`
    (bid/ask price), not top-level `bid_price`/`ask_price`.
    """
    if not isinstance(snap, dict):
        return 0.0, 0.0
    quote = snap.get("latestQuote") or {}
    bid = quote.get("bp")
    ask = quote.get("ap")
    if bid is None or ask is None:
        return 0.0, 0.0
    return float(bid or 0.0), float(ask or 0.0)


async def _current_close_value(client: AlpacaMcpClient, legs_json: str) -> float | None:
    legs = json.loads(legs_json)
    symbols = [leg["symbol"] for leg in legs]
    snapshots = await client.get_option_snapshot(symbols)
    total = 0.0
    for leg in legs:
        bid, ask = _extract_bid_ask(snapshots.get(leg["symbol"], {}))
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None  # can't safely value this leg right now (e.g. no quote) -- skip this cycle
        total += mid if leg["side"] == "sell" else -mid
    return total * 100.0


async def _close_position_row(
    client: AlpacaMcpClient, db_path: str, row: dict, current_value: float, reason: str, dry_run: bool = False
) -> str:
    """Places the closing order for one open-position ledger row and logs it.
    Shared by the deterministic per-cycle exit check and the dashboard's
    manual-close action -- both just need a reason string and a valuation."""
    legs = json.loads(row["legs_json"])
    closing_legs = [{"symbol": leg["symbol"], "side": "buy" if leg["side"] == "sell" else "sell", "ratio_qty": 1} for leg in legs]

    if dry_run:
        ledger.log_decision(
            db_path, candidate_id=None, symbol=row["symbol"],
            decision="dry_run_would_close", detail=f"reason={reason}, current_value={current_value}",
        )
        logger.info("[DRY RUN] would close %s (%s) reason=%s", row["symbol"], row["position_group"], reason)
        return f"[DRY RUN] would close {row['symbol']} ({row['position_group']}) reason={reason}"

    client_order_id = f"close-{row['position_group'][:8]}"
    order = await client.place_option_order(closing_legs, row["contracts"], client_order_id)
    ledger.log_trade(
        db_path,
        candidate_id=None,
        position_group=row["position_group"],
        symbol=row["symbol"],
        sleeve=row["sleeve"],
        strategy_type=row["strategy_type"],
        action="close",
        contracts=row["contracts"],
        credit_or_debit=current_value,
        legs=closing_legs,
        rationale=reason,
        order_id=order.get("id") if isinstance(order, dict) else None,
    )
    logger.info("closed %s (%s) reason=%s", row["symbol"], row["position_group"], reason)
    return f"Closed {row['symbol']} ({row['position_group']}) reason={reason}"


async def _manage_open_positions(client: AlpacaMcpClient, db_path: str, today: date, config: dict, dry_run: bool = False) -> None:
    for row in ledger.open_positions(db_path):
        expiration = datetime.strptime(row["expiration"], "%Y-%m-%d").date() if row["expiration"] else today
        position = ManagedPosition(
            position_id=row["position_group"],
            symbol=row["symbol"],
            sleeve=row["sleeve"],
            strategy_type=row["strategy_type"],
            contracts=row["contracts"],
            credit_received_per_contract=row["credit_received"],
            opened_at=datetime.fromisoformat(row["opened_at"]).date(),
            expiration=expiration,
            days_to_earnings=row["days_to_earnings"],
        )
        current_value = await _current_close_value(client, row["legs_json"])
        if current_value is None:
            logger.warning("skipping exit check for %s: no live quote", row["position_group"])
            continue

        decision = evaluate_exit(position, current_value, today, config)
        if not decision.should_close:
            continue

        await _close_position_row(client, db_path, row, current_value, f"deterministic exit: {decision.reason}", dry_run=dry_run)


async def close_position_manually(config: Config, position_group: str, dry_run: bool = True) -> str:
    """Dashboard-triggered manual close of one specific open position,
    bypassing the deterministic exit rules (evaluate_exit) entirely -- this
    is a human override, so no reason beyond "operator requested" is needed.
    Raises ValueError if the position isn't open (already closed / unknown)."""
    row = ledger.open_position_by_group(config.db_path, position_group)
    if row is None:
        raise ValueError(f"no open position found for position_group={position_group!r}")

    async with AlpacaMcpClient(config) as client:
        current_value = await _current_close_value(client, row["legs_json"])
        if current_value is None:
            raise RuntimeError(f"no live quote available for {row['symbol']} ({position_group}); cannot value the close")
        return await _close_position_row(client, config.db_path, row, current_value, "manual close (dashboard)", dry_run=dry_run)


def _screen_symbol(
    symbol: str,
    snapshot: SymbolSnapshot,
    equity: float,
    today: date,
    config: dict,
    exposure_multiplier: float = 1.0,
) -> list[CandidateTrade]:
    signals_cfg = config["signals"]
    sleeve_a_cfg = config["strategy"]["sleeve_a"]
    sizing_kelly = config["sizing"]["kelly_fraction"]
    vrp_edge = config["sizing"]["assumed_win_prob_edge"]
    risk_cfg = config["risk_gates"]

    candidates: list[CandidateTrade] = []

    if snapshot.iv_rank_value >= signals_cfg["iv_rank_entry_threshold"]:
        expirations = sorted({q.expiration for q in snapshot.chain})
        expiration = select_expiration(expirations, today, sleeve_a_cfg["target_dte_min"], sleeve_a_cfg["target_dte_max"])
        if expiration is not None:
            width = sleeve_a_cfg["spread_width_etf"] if universe.is_etf(symbol) else sleeve_a_cfg["spread_width_single_name"]
            common_kwargs = dict(
                symbol=symbol, sleeve="A", chain=snapshot.chain, expiration=expiration, equity=equity,
                kelly_multiplier=sizing_kelly, max_risk_per_trade_pct=risk_cfg["max_risk_per_trade_pct"],
                short_delta_target=sleeve_a_cfg["short_delta_target"], short_delta_band=sleeve_a_cfg["short_delta_band"],
                width=width, days_to_earnings=snapshot.days_to_earnings, vrp_edge=vrp_edge,
                exposure_multiplier=exposure_multiplier,
            )
            if snapshot.regime.is_trending and snapshot.regime.direction in ("bullish", "bearish"):
                rationale = (
                    f"VRP entry: IV rank {snapshot.iv_rank_value:.0f} >= {signals_cfg['iv_rank_entry_threshold']}; "
                    f"trending {snapshot.regime.direction} (ADX {snapshot.regime.adx_value:.1f}, "
                    f"12-1 momentum {snapshot.regime.momentum_12_1:.2%}) -> directional credit spread."
                )
                candidate = build_credit_spread_candidate(regime=snapshot.regime, rationale_hint=rationale, **common_kwargs)
            else:
                rationale = (
                    f"VRP entry: IV rank {snapshot.iv_rank_value:.0f} >= {signals_cfg['iv_rank_entry_threshold']}; "
                    f"no clear trend (ADX {snapshot.regime.adx_value:.1f}) -> iron condor."
                )
                candidate = build_iron_condor_candidate(rationale_hint=rationale, **common_kwargs)
            if candidate is not None:
                candidates.append(candidate)

    if snapshot.days_to_earnings is not None:
        earnings_date = today + timedelta(days=snapshot.days_to_earnings)
        b_candidate = earnings_sleeve.screen_earnings_candidate(
            symbol=symbol, chain=snapshot.chain, today=today, earnings_date=earnings_date,
            iv_rank_value=snapshot.iv_rank_value, equity=equity, config=config,
            exposure_multiplier=exposure_multiplier,
        )
        if b_candidate is not None:
            candidates.append(b_candidate)

    return candidates


async def _latest_close(client: AlpacaMcpClient, symbol: str, today: date) -> float | None:
    """Most recent daily close for an index/symbol, or None if unavailable."""
    start = (today - timedelta(days=10)).isoformat()
    end = today.isoformat()
    try:
        bars = await client.get_stock_bars(symbol, "1Day", start, end)
    except Exception:  # noqa: BLE001 -- a missing/failed vol feed must not break the cycle
        return None
    if not bars:
        return None
    last = bars[-1]
    close = last.get("c") if isinstance(last, dict) else None  # bar field is "c" (verified live), not "close"
    return float(close) if close else None


async def _exposure_multiplier(client: AlpacaMcpClient, today: date, config: dict) -> float:
    """Global short-premium exposure throttle from the vol term structure
    (signals.vol_term_structure_regime). Returns 1.0 (no throttle) whenever
    the feature is disabled or a vol reading can't be obtained -- a missing
    signal should never *increase* risk, only fail open to normal sizing.
    """
    cfg = config.get("vol_term_structure") or {}
    if not cfg.get("enabled", False):
        return 1.0

    front = await _latest_close(client, cfg.get("front_symbol", "VIX"), today)
    back = await _latest_close(client, cfg.get("back_symbol", "VIX3M"), today)
    if front is None or back is None:
        logger.warning("vol term-structure feed unavailable; defaulting exposure multiplier to 1.0")
        return 1.0

    signal = vol_term_structure_regime(
        front, back,
        contango_ratio=cfg.get("contango_ratio", 0.95),
        backwardation_ratio=cfg.get("backwardation_ratio", 1.00),
        floor_ratio=cfg.get("floor_ratio", 1.10),
        min_multiplier=cfg.get("min_exposure_multiplier", 0.0),
    )
    logger.info(
        "vol term structure: %s (ratio %.3f) -> exposure multiplier %.2f",
        signal.regime, signal.ratio, signal.exposure_multiplier,
    )
    return signal.exposure_multiplier


async def _apply_delta_hedge(
    client: AlpacaMcpClient, db_path: str, today: date, portfolio: PortfolioState, config: dict, dry_run: bool = False
) -> None:
    """Turn a circuit-breaker delta/vega breach into a concrete SPY share
    hedge that pulls net delta back toward neutral. Best-effort: an order
    failure is logged, never fatal to the cycle."""
    order = compute_delta_hedge(portfolio, config)
    if not order.needed:
        if order.vega_breach:
            ledger.log_decision(db_path, candidate_id=None, symbol=order.symbol, decision="hedge_noted", detail=order.reason)
        return

    shares = order.shares
    if shares <= 0:
        return

    if dry_run:
        ledger.log_decision(db_path, candidate_id=None, symbol=order.symbol, decision="dry_run_would_hedge", detail=f"{order.side} {shares} {order.symbol}; {order.reason}")
        logger.info("[DRY RUN] would hedge: %s %d %s (%s)", order.side, shares, order.symbol, order.reason)
        return

    client_order_id = f"hedge-{today.isoformat()}-{order.side}"
    try:
        await client.place_stock_order(order.symbol, order.side, shares, client_order_id)
    except Exception as exc:  # noqa: BLE001 -- hedge failure must not crash the cycle
        logger.warning("hedge order failed: %s", exc)
        ledger.log_decision(db_path, candidate_id=None, symbol=order.symbol, decision="hedge_failed", detail=str(exc))
        return
    ledger.log_decision(db_path, candidate_id=None, symbol=order.symbol, decision="hedge_executed", detail=f"{order.side} {shares} {order.symbol}; {order.reason}")
    logger.info("hedge executed: %s %d %s (%s)", order.side, shares, order.symbol, order.reason)


async def run_one_cycle(config: Config, dry_run: bool = False) -> str:
    today = date.today()

    async with AlpacaMcpClient(config) as client:
        await _manage_open_positions(client, config.db_path, today, config.raw, dry_run=dry_run)

        account_raw = await client.get_account()
        equity, daily_pnl_pct = _parse_account(account_raw)
        portfolio = _load_portfolio_state(config.db_path, equity, daily_pnl_pct)

        ledger.record_equity_snapshot(
            config.db_path, snapshot_date=today, equity=portfolio.equity, equity_peak=portfolio.equity_peak,
            daily_pnl_pct=portfolio.daily_pnl_pct, open_position_count=portfolio.total_positions,
            portfolio_heat=portfolio.portfolio_heat,
        )

        cb_state = circuit_breaker.evaluate(portfolio, config.raw)

        if cb_state.hedge_needed:
            await _apply_delta_hedge(client, config.db_path, today, portfolio, config.raw, dry_run=dry_run)

        if cb_state.halt_new_entries:
            reason = "; ".join(cb_state.reasons)
            ledger.log_decision(config.db_path, candidate_id=None, symbol=None, decision="no_candidates", detail=f"circuit breaker halt: {reason}")
            logger.warning("circuit breaker halting new entries: %s", reason)
            return f"No new entries this cycle -- circuit breaker: {reason}"

        halt_state = control.get_halt_state(config.db_path)
        if halt_state.halted:
            detail = f"manual kill switch active (dashboard): {halt_state.reason}" if halt_state.reason else "manual kill switch active (dashboard)"
            ledger.log_decision(config.db_path, candidate_id=None, symbol=None, decision="no_candidates", detail=detail)
            logger.warning(detail)
            return f"No new entries this cycle -- {detail}"

        exposure_multiplier = await _exposure_multiplier(client, today, config.raw)
        if exposure_multiplier <= 0.0:
            ledger.log_decision(config.db_path, candidate_id=None, symbol=None, decision="no_candidates", detail="vol term structure in backwardation -- new premium selling halted this cycle")
            logger.warning("vol term structure inverted; halting new entries this cycle")
            return "No new entries this cycle -- vol term structure in backwardation (exposure throttled to zero)."

        approved: dict[str, CandidateTrade] = {}
        for entry in universe.ALL_ENTRIES:
            symbol = entry.symbol
            snapshot = await build_symbol_snapshot(client, config.db_path, symbol, today, config.raw)
            if snapshot is None:
                continue
            for candidate in _screen_symbol(symbol, snapshot, equity, today, config.raw, exposure_multiplier):
                allowed, checks = gate_result(candidate.to_proposed_trade(), portfolio, config.raw)
                if allowed:
                    approved[candidate.candidate_id] = candidate
                else:
                    reasons = "; ".join(c.detail for c in checks if not c.passed)
                    ledger.log_decision(config.db_path, candidate_id=candidate.candidate_id, symbol=symbol, decision="gate_rejected", detail=reasons)

        if not approved:
            ledger.log_decision(config.db_path, candidate_id=None, symbol=None, decision="no_candidates", detail="no candidates passed screening + gates this cycle")
            return "No candidates passed screening and risk gates this cycle."

        state = CycleState(
            candidates=approved, portfolio=portfolio, config=config.raw, alpaca=client,
            db_path=config.db_path, dry_run=dry_run,
        )
        summary = await run_llm_cycle(config, state)
        return summary
