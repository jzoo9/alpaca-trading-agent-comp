"""Turns market data (option chain quotes + regime/IV signals) into fully
specified candidate trades: exact legs, strikes, expiration, and a proposed
contract count. Nothing here talks to the network -- callers hand in plain
OptionQuote objects (typically built from an Alpaca option chain snapshot in
data/snapshot.py), which keeps this module unit-testable with fixtures.

A CandidateTrade is deliberately "fully specified": the LLM layer in
agent/brain.py can only choose to take-or-skip a CandidateTrade as-is, never
edit its strikes/qty/price (see agent/tools.py).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from alpaca_quant_agent.risk.gates import ProposedTrade
from alpaca_quant_agent.strategy.signals import RegimeSignal
from alpaca_quant_agent.strategy.sizing import contracts_for_trade, win_probability_estimate


@dataclass(frozen=True)
class OptionQuote:
    occ_symbol: str
    underlying: str
    strike: float
    expiration: date
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    delta: float  # signed: puts negative, calls positive
    vega: float
    open_interest: int
    iv: float = 0.0  # implied volatility, used for ATM IV-rank tracking (data/snapshot.py)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m if m > 0 else float("inf")


@dataclass(frozen=True)
class OptionLeg:
    occ_symbol: str
    side: str  # "buy" | "sell"
    option_type: str
    strike: float


@dataclass(frozen=True)
class CandidateTrade:
    candidate_id: str
    symbol: str
    sleeve: str
    strategy_type: str
    expiration: date
    legs: tuple[OptionLeg, ...]
    contracts: int
    credit_per_contract: float
    max_loss_per_contract: float
    net_delta_per_contract: float
    net_vega_per_contract: float
    open_interest: int
    bid_ask_spread_pct: float
    days_to_earnings: int | None
    rationale_hint: str

    def to_proposed_trade(self) -> ProposedTrade:
        return ProposedTrade(
            symbol=self.symbol,
            sleeve=self.sleeve,
            strategy_type=self.strategy_type,
            contracts=self.contracts,
            max_loss_per_contract=self.max_loss_per_contract,
            credit_per_contract=self.credit_per_contract,
            net_delta_per_contract=self.net_delta_per_contract,
            net_vega_per_contract=self.net_vega_per_contract,
            open_interest=self.open_interest,
            bid_ask_spread_pct=self.bid_ask_spread_pct,
            days_to_earnings=self.days_to_earnings,
        )


def select_expiration(expirations: list[date], today: date, dte_min: int, dte_max: int) -> date | None:
    in_range = [e for e in expirations if dte_min <= (e - today).days <= dte_max]
    if not in_range:
        return None
    # Prefer the expiration closest to the midpoint of the target DTE window.
    target_dte = (dte_min + dte_max) / 2
    return min(in_range, key=lambda e: abs((e - today).days - target_dte))


def select_short_strike(
    chain: list[OptionQuote], option_type: str, target_delta: float, band: float
) -> OptionQuote | None:
    candidates = [
        q for q in chain
        if q.option_type == option_type and abs(abs(q.delta) - target_delta) <= band
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda q: abs(abs(q.delta) - target_delta))


def select_long_strike(
    chain: list[OptionQuote], short: OptionQuote, width: float, option_type: str
) -> OptionQuote | None:
    # Put credit spread: long strike is BELOW short strike (protection below).
    # Call credit spread: long strike is ABOVE short strike (protection above).
    target_strike = short.strike - width if option_type == "put" else short.strike + width
    candidates = [q for q in chain if q.option_type == option_type and q.strike != short.strike]
    if not candidates:
        return None
    return min(candidates, key=lambda q: abs(q.strike - target_strike))


def _spread_economics(short: OptionQuote, long: OptionQuote) -> tuple[float, float, float, float]:
    """Returns (credit_per_contract, max_loss_per_contract, net_delta, net_vega),
    all in per-contract dollar terms (x100 for credit/max_loss)."""
    credit = (short.mid - long.mid) * 100.0
    width = abs(short.strike - long.strike)
    max_loss = width * 100.0 - credit
    # Short leg is sold (negative delta/vega exposure to us), long leg is bought.
    net_delta = (-short.delta + long.delta) * 100.0
    net_vega = (-short.vega + long.vega) * 100.0
    return credit, max_loss, net_delta, net_vega


def build_credit_spread_candidate(
    *,
    symbol: str,
    sleeve: str,
    chain: list[OptionQuote],
    regime: RegimeSignal,
    expiration: date,
    equity: float,
    kelly_multiplier: float,
    max_risk_per_trade_pct: float,
    short_delta_target: float,
    short_delta_band: float,
    width: float,
    days_to_earnings: int | None,
    rationale_hint: str,
    vrp_edge: float = 0.0,
    exposure_multiplier: float = 1.0,
) -> CandidateTrade | None:
    option_type = "put" if regime.direction == "bullish" else "call"
    exp_chain = [q for q in chain if q.expiration == expiration]

    short = select_short_strike(exp_chain, option_type, short_delta_target, short_delta_band)
    if short is None:
        return None
    long = select_long_strike(exp_chain, short, width, option_type)
    if long is None:
        return None

    credit, max_loss, net_delta, net_vega = _spread_economics(short, long)
    if credit <= 0 or max_loss <= 0:
        return None

    win_prob = win_probability_estimate(short.delta, vrp_edge)
    sizing = contracts_for_trade(
        equity=equity,
        win_prob=win_prob,
        credit_per_contract=credit,
        max_loss_per_contract=max_loss,
        kelly_multiplier=kelly_multiplier,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        exposure_multiplier=exposure_multiplier,
    )
    if sizing.contracts <= 0:
        return None

    strategy_type = "put_credit_spread" if option_type == "put" else "call_credit_spread"
    legs = (
        OptionLeg(short.occ_symbol, "sell", option_type, short.strike),
        OptionLeg(long.occ_symbol, "buy", option_type, long.strike),
    )
    bid_ask = max(short.spread_pct, long.spread_pct)
    open_interest = min(short.open_interest, long.open_interest)

    return CandidateTrade(
        candidate_id=str(uuid.uuid4()),
        symbol=symbol,
        sleeve=sleeve,
        strategy_type=strategy_type,
        expiration=expiration,
        legs=legs,
        contracts=sizing.contracts,
        credit_per_contract=credit,
        max_loss_per_contract=max_loss,
        net_delta_per_contract=net_delta,
        net_vega_per_contract=net_vega,
        open_interest=open_interest,
        bid_ask_spread_pct=bid_ask,
        days_to_earnings=days_to_earnings,
        rationale_hint=rationale_hint,
    )


def build_iron_condor_candidate(
    *,
    symbol: str,
    sleeve: str,
    chain: list[OptionQuote],
    expiration: date,
    equity: float,
    kelly_multiplier: float,
    max_risk_per_trade_pct: float,
    short_delta_target: float,
    short_delta_band: float,
    width: float,
    days_to_earnings: int | None,
    rationale_hint: str,
    vrp_edge: float = 0.0,
    exposure_multiplier: float = 1.0,
) -> CandidateTrade | None:
    exp_chain = [q for q in chain if q.expiration == expiration]

    put_short = select_short_strike(exp_chain, "put", short_delta_target, short_delta_band)
    call_short = select_short_strike(exp_chain, "call", short_delta_target, short_delta_band)
    if put_short is None or call_short is None:
        return None
    put_long = select_long_strike(exp_chain, put_short, width, "put")
    call_long = select_long_strike(exp_chain, call_short, width, "call")
    if put_long is None or call_long is None:
        return None

    put_credit, put_max_loss, put_delta, put_vega = _spread_economics(put_short, put_long)
    call_credit, call_max_loss, call_delta, call_vega = _spread_economics(call_short, call_long)
    credit = put_credit + call_credit
    # Iron condor max loss is the worse of the two wings (only one side can be breached).
    max_loss = max(put_max_loss, call_max_loss) - min(put_credit, call_credit)
    if credit <= 0 or max_loss <= 0:
        return None

    win_prob = win_probability_estimate(put_short.delta, vrp_edge) * win_probability_estimate(call_short.delta, vrp_edge)
    sizing = contracts_for_trade(
        equity=equity,
        win_prob=win_prob,
        credit_per_contract=credit,
        max_loss_per_contract=max_loss,
        kelly_multiplier=kelly_multiplier,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        exposure_multiplier=exposure_multiplier,
    )
    if sizing.contracts <= 0:
        return None

    legs = (
        OptionLeg(put_short.occ_symbol, "sell", "put", put_short.strike),
        OptionLeg(put_long.occ_symbol, "buy", "put", put_long.strike),
        OptionLeg(call_short.occ_symbol, "sell", "call", call_short.strike),
        OptionLeg(call_long.occ_symbol, "buy", "call", call_long.strike),
    )
    bid_ask = max(put_short.spread_pct, put_long.spread_pct, call_short.spread_pct, call_long.spread_pct)
    open_interest = min(put_short.open_interest, put_long.open_interest, call_short.open_interest, call_long.open_interest)

    return CandidateTrade(
        candidate_id=str(uuid.uuid4()),
        symbol=symbol,
        sleeve=sleeve,
        strategy_type="iron_condor",
        expiration=expiration,
        legs=legs,
        contracts=sizing.contracts,
        credit_per_contract=credit,
        max_loss_per_contract=max_loss,
        net_delta_per_contract=put_delta + call_delta,
        net_vega_per_contract=put_vega + call_vega,
        open_interest=open_interest,
        bid_ask_spread_pct=bid_ask,
        days_to_earnings=days_to_earnings,
        rationale_hint=rationale_hint,
    )
