from datetime import date

import pytest

from alpaca_quant_agent.strategy.screener import (
    OptionQuote,
    build_credit_spread_candidate,
    build_iron_condor_candidate,
    select_expiration,
    select_long_strike,
    select_short_strike,
)
from alpaca_quant_agent.strategy.signals import RegimeSignal

EXP = date(2026, 10, 16)


def make_chain():
    # SPY-like put wing (deltas negative) and call wing (deltas positive),
    # strikes spaced $5 apart around a $500 underlying.
    # Priced so the ~0.22-delta short / $5-wide long has a realistic
    # credit/max_loss ratio (~0.28), matching typical live OTM spread pricing.
    puts = [
        OptionQuote("SPY261016P00485000", "SPY", 485.0, EXP, "put", 0.80, 0.90, -0.12, 0.30, 800),
        OptionQuote("SPY261016P00490000", "SPY", 490.0, EXP, "put", 1.90, 2.00, -0.22, 0.35, 900),
        OptionQuote("SPY261016P00495000", "SPY", 495.0, EXP, "put", 3.20, 3.30, -0.35, 0.40, 700),
    ]
    calls = [
        OptionQuote("SPY261016C00505000", "SPY", 505.0, EXP, "call", 3.10, 3.20, 0.35, 0.40, 700),
        OptionQuote("SPY261016C00510000", "SPY", 510.0, EXP, "call", 1.85, 1.95, 0.22, 0.35, 900),
        OptionQuote("SPY261016C00515000", "SPY", 515.0, EXP, "call", 0.80, 0.90, 0.12, 0.30, 800),
    ]
    return puts + calls


def test_select_expiration_prefers_midpoint():
    expirations = [date(2026, 9, 10), date(2026, 10, 5), date(2026, 11, 1)]
    today = date(2026, 9, 1)
    chosen = select_expiration(expirations, today, dte_min=30, dte_max=45)
    assert chosen == date(2026, 10, 5)


def test_select_expiration_none_when_out_of_range():
    expirations = [date(2026, 9, 5), date(2026, 12, 1)]
    today = date(2026, 9, 1)
    assert select_expiration(expirations, today, dte_min=30, dte_max=45) is None


def test_select_short_strike_finds_closest_to_target_delta():
    chain = make_chain()
    short = select_short_strike(chain, "put", target_delta=0.20, band=0.10)
    assert short.strike == 490.0  # delta -0.22 is closest to 0.20


def test_select_short_strike_none_outside_band():
    chain = make_chain()
    short = select_short_strike(chain, "put", target_delta=0.70, band=0.05)
    assert short is None


def test_select_long_strike_put_is_below_short():
    chain = make_chain()
    short = next(q for q in chain if q.strike == 490.0 and q.option_type == "put")
    long = select_long_strike(chain, short, width=5.0, option_type="put")
    assert long.strike == 485.0


def test_select_long_strike_call_is_above_short():
    chain = make_chain()
    short = next(q for q in chain if q.strike == 510.0 and q.option_type == "call")
    long = select_long_strike(chain, short, width=5.0, option_type="call")
    assert long.strike == 515.0


def test_build_credit_spread_candidate_bullish_selects_put_spread():
    chain = make_chain()
    regime = RegimeSignal(direction="bullish", is_trending=True, adx_value=30.0, momentum_12_1=0.1)
    candidate = build_credit_spread_candidate(
        symbol="SPY",
        sleeve="A",
        chain=chain,
        regime=regime,
        expiration=EXP,
        equity=100_000.0,
        kelly_multiplier=0.3,
        max_risk_per_trade_pct=0.02,
        short_delta_target=0.22,
        short_delta_band=0.05,
        width=5.0,
        days_to_earnings=None,
        rationale_hint="test",
        vrp_edge=0.05,
    )
    assert candidate is not None
    assert candidate.strategy_type == "put_credit_spread"
    assert candidate.symbol == "SPY"
    # short=490 (bid 1.90/ask 2.00 mid 1.95), long=485 (bid 0.80/ask 0.90 mid 0.85)
    # credit = (1.95 - 0.85) * 100 = 110
    assert candidate.credit_per_contract == pytest.approx(110.0)
    assert candidate.max_loss_per_contract == pytest.approx(500.0 - 110.0)
    assert candidate.contracts >= 0


def test_build_credit_spread_candidate_bearish_selects_call_spread():
    chain = make_chain()
    regime = RegimeSignal(direction="bearish", is_trending=True, adx_value=30.0, momentum_12_1=-0.1)
    candidate = build_credit_spread_candidate(
        symbol="SPY",
        sleeve="A",
        chain=chain,
        regime=regime,
        expiration=EXP,
        equity=100_000.0,
        kelly_multiplier=0.3,
        max_risk_per_trade_pct=0.02,
        short_delta_target=0.22,
        short_delta_band=0.05,
        width=5.0,
        days_to_earnings=None,
        rationale_hint="test",
        vrp_edge=0.05,
    )
    assert candidate is not None
    assert candidate.strategy_type == "call_credit_spread"


def test_build_credit_spread_candidate_none_when_no_strike_in_band():
    chain = make_chain()
    regime = RegimeSignal(direction="bullish", is_trending=True, adx_value=30.0, momentum_12_1=0.1)
    candidate = build_credit_spread_candidate(
        symbol="SPY",
        sleeve="A",
        chain=chain,
        regime=regime,
        expiration=EXP,
        equity=100_000.0,
        kelly_multiplier=0.3,
        max_risk_per_trade_pct=0.02,
        short_delta_target=0.90,  # nothing this deep ITM in the fixture
        short_delta_band=0.02,
        width=5.0,
        days_to_earnings=None,
        rationale_hint="test",
    )
    assert candidate is None


def test_build_iron_condor_candidate_has_four_legs():
    chain = make_chain()
    candidate = build_iron_condor_candidate(
        symbol="SPY",
        sleeve="A",
        chain=chain,
        expiration=EXP,
        equity=100_000.0,
        kelly_multiplier=0.3,
        max_risk_per_trade_pct=0.02,
        short_delta_target=0.22,
        short_delta_band=0.05,
        width=5.0,
        days_to_earnings=None,
        rationale_hint="test",
        vrp_edge=0.05,
    )
    assert candidate is not None
    assert candidate.strategy_type == "iron_condor"
    assert len(candidate.legs) == 4
    assert candidate.credit_per_contract > 0
    assert candidate.max_loss_per_contract > 0
