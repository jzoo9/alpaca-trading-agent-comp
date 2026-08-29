import pytest

from alpaca_quant_agent.strategy.sizing import (
    contracts_for_trade,
    full_kelly_fraction,
    win_probability_estimate,
    win_probability_from_delta,
)


def test_win_probability_from_delta():
    assert win_probability_from_delta(-0.20) == pytest.approx(0.80)
    assert win_probability_from_delta(0.30) == pytest.approx(0.70)


def test_win_probability_estimate_applies_edge_and_clamps():
    assert win_probability_estimate(-0.20, 0.05) == pytest.approx(0.85)
    assert win_probability_estimate(-0.01, 0.5) == pytest.approx(0.999)  # clamped
    assert win_probability_estimate(-0.99, -1.0) == pytest.approx(0.0)  # clamped


def test_full_kelly_fraction_zero_edge_case():
    # Fairly priced 20-delta spread (win_prob purely delta-implied, no VRP edge)
    # has ~zero edge by construction: p=0.8, b=credit/max_loss=100/400=0.25
    # f* = 0.8 - 0.2/0.25 = 0.0
    f = full_kelly_fraction(win_prob=0.80, credit_per_contract=100.0, max_loss_per_contract=400.0)
    assert f == pytest.approx(0.0, abs=1e-9)


def test_full_kelly_fraction_positive_with_vrp_edge():
    # Same spread economics, but win_prob bumped by a VRP edge -> positive edge.
    f = full_kelly_fraction(win_prob=0.85, credit_per_contract=100.0, max_loss_per_contract=400.0)
    assert f > 0
    # Hand-computed: b=0.25, p=0.85 -> f* = 0.85 - 0.15/0.25 = 0.25
    assert f == pytest.approx(0.25, abs=1e-9)


def test_full_kelly_fraction_negative_edge_clamped_to_zero():
    f = full_kelly_fraction(win_prob=0.5, credit_per_contract=50.0, max_loss_per_contract=450.0)
    assert f == 0.0


def test_full_kelly_fraction_rejects_nonpositive_max_loss():
    with pytest.raises(ValueError):
        full_kelly_fraction(win_prob=0.8, credit_per_contract=100.0, max_loss_per_contract=0.0)


def test_contracts_for_trade_capped_by_max_risk_pct():
    # Full Kelly here is large (25%), but the hard per-trade cap is 2%.
    result = contracts_for_trade(
        equity=100_000.0,
        win_prob=0.85,
        credit_per_contract=100.0,
        max_loss_per_contract=400.0,
        kelly_multiplier=1.0,  # full Kelly, to prove the cap (not the multiplier) binds
        max_risk_per_trade_pct=0.02,
    )
    # dollars_budget = 100_000 * 0.02 = 2_000 -> 2_000 / 400 = 5 contracts
    assert result.contracts == 5
    assert result.dollars_at_risk == pytest.approx(2000.0)
    assert result.kelly_fraction_used == pytest.approx(0.02)


def test_contracts_for_trade_uses_fractional_kelly_when_below_cap():
    result = contracts_for_trade(
        equity=100_000.0,
        win_prob=0.85,
        credit_per_contract=100.0,
        max_loss_per_contract=400.0,
        kelly_multiplier=0.3,  # 0.3 * 0.25 full kelly = 0.075, still capped by 2%? no: 0.075 > 0.02
        max_risk_per_trade_pct=0.02,
    )
    # 0.3 * 0.25 = 0.075, which exceeds the 2% cap, so cap still binds.
    assert result.kelly_fraction_used == pytest.approx(0.02)


def test_contracts_for_trade_zero_equity():
    result = contracts_for_trade(
        equity=0.0,
        win_prob=0.85,
        credit_per_contract=100.0,
        max_loss_per_contract=400.0,
        kelly_multiplier=0.3,
        max_risk_per_trade_pct=0.02,
    )
    assert result.contracts == 0
