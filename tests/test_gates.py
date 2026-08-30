import pytest

from alpaca_quant_agent.risk.circuit_breaker import evaluate as evaluate_circuit_breaker
from alpaca_quant_agent.risk.gates import OpenPosition, PortfolioState, ProposedTrade, gate_result

CONFIG = {
    "risk_gates": {
        "max_risk_per_trade_pct": 0.02,
        "max_portfolio_heat_pct": 0.20,
        "max_bucket_heat_pct": 0.10,
        "max_positions_per_underlying": 1,
        "max_total_positions": 12,
        "sleeve_b_max_allocation_pct": 0.15,
        "portfolio_delta_band_pct": 0.05,
        "portfolio_vega_cap_pct": 0.03,
        "daily_loss_halt_pct": -0.03,
        "total_drawdown_kill_switch_pct": -0.10,
        "min_open_interest": 100,
        "max_bid_ask_spread_pct": 0.15,
    },
    "strategy": {
        "sleeve_a": {"earnings_blackout_days": 3},
    },
}


def make_trade(**overrides):
    defaults = dict(
        symbol="SPY",
        sleeve="A",
        strategy_type="put_credit_spread",
        contracts=5,
        max_loss_per_contract=400.0,
        credit_per_contract=100.0,
        net_delta_per_contract=8.0,
        net_vega_per_contract=-2.0,
        open_interest=500,
        bid_ask_spread_pct=0.05,
        days_to_earnings=None,
    )
    defaults.update(overrides)
    return ProposedTrade(**defaults)


def make_portfolio(**overrides):
    defaults = dict(
        equity=100_000.0,
        equity_peak=100_000.0,
        daily_pnl_pct=0.0,
        open_positions=(),
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def test_clean_trade_passes_all_gates():
    trade = make_trade()
    portfolio = make_portfolio()
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert allowed
    assert all(c.passed for c in checks)


def test_max_risk_per_trade_boundary():
    # 5 contracts * 400 max loss = 2000 = exactly 2% of 100_000 -> should pass.
    trade = make_trade(contracts=5, max_loss_per_contract=400.0)
    portfolio = make_portfolio(equity=100_000.0)
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert allowed

    # 6 contracts -> 2400 > 2% cap -> should fail on that specific gate.
    trade_over = make_trade(contracts=6, max_loss_per_contract=400.0)
    allowed_over, checks_over = gate_result(trade_over, portfolio, CONFIG)
    assert not allowed_over
    failed_names = {c.name for c in checks_over if not c.passed}
    assert "max_risk_per_trade" in failed_names


def test_max_positions_per_underlying_blocks_duplicate():
    trade = make_trade(symbol="SPY")
    portfolio = make_portfolio(
        open_positions=(OpenPosition(symbol="SPY", sleeve="A", max_loss=100.0, net_delta=1.0, net_vega=-0.5),)
    )
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    assert any(c.name == "max_positions_per_underlying" and not c.passed for c in checks)


def test_earnings_blackout_blocks_sleeve_a():
    trade = make_trade(sleeve="A", days_to_earnings=1)
    portfolio = make_portfolio()
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    assert any(c.name == "earnings_blackout" and not c.passed for c in checks)


def test_earnings_blackout_does_not_apply_to_sleeve_b():
    trade = make_trade(sleeve="B", days_to_earnings=1)
    portfolio = make_portfolio()
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not any(c.name == "earnings_blackout" for c in checks)


def test_liquidity_gates():
    trade = make_trade(open_interest=10, bid_ask_spread_pct=0.5)
    portfolio = make_portfolio()
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    failed = {c.name for c in checks if not c.passed}
    assert {"min_open_interest", "max_bid_ask_spread"} <= failed


def test_daily_loss_halt_blocks_new_entry():
    trade = make_trade()
    portfolio = make_portfolio(daily_pnl_pct=-0.04)
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    assert any(c.name == "daily_loss_halt" and not c.passed for c in checks)


def test_drawdown_kill_switch_blocks_new_entry():
    trade = make_trade()
    portfolio = make_portfolio(equity=89_000.0, equity_peak=100_000.0)
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    assert any(c.name == "total_drawdown_kill_switch" and not c.passed for c in checks)


def test_circuit_breaker_halts_on_daily_loss():
    portfolio = make_portfolio(daily_pnl_pct=-0.05)
    state = evaluate_circuit_breaker(portfolio, CONFIG)
    assert state.halt_new_entries
    assert not state.manage_only


def test_circuit_breaker_manage_only_on_drawdown():
    portfolio = make_portfolio(equity=85_000.0, equity_peak=100_000.0)
    state = evaluate_circuit_breaker(portfolio, CONFIG)
    assert state.halt_new_entries
    assert state.manage_only


def test_circuit_breaker_hedge_needed_on_delta_breach():
    portfolio = make_portfolio(
        open_positions=(
            OpenPosition(symbol="SPY", sleeve="A", max_loss=100.0, net_delta=6000.0, net_vega=0.0),
        )
    )
    state = evaluate_circuit_breaker(portfolio, CONFIG)
    assert state.hedge_needed
    assert not state.halt_new_entries


def test_circuit_breaker_clean_state():
    portfolio = make_portfolio()
    state = evaluate_circuit_breaker(portfolio, CONFIG)
    assert not state.halt_new_entries
    assert not state.manage_only
    assert not state.hedge_needed
    assert state.reasons == ()


# --- concentration gate (idea 2) -------------------------------------------

def test_concentration_gate_blocks_when_bucket_full():
    # Existing tech-bucket exposure: 9000 (QQQ) + tries to add NVDA 2000 -> 11000
    # = 11% of 100k, over the 10% bucket cap. NVDA and QQQ share the megacap_tech
    # bucket, so this must be blocked even though the new trade alone is fine.
    trade = make_trade(symbol="NVDA", contracts=5, max_loss_per_contract=400.0)
    portfolio = make_portfolio(
        open_positions=(
            OpenPosition(symbol="QQQ", sleeve="A", max_loss=9000.0, net_delta=0.0, net_vega=0.0),
        )
    )
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert not allowed
    assert any(c.name == "max_bucket_heat" and not c.passed for c in checks)


def test_concentration_gate_allows_diversifying_across_buckets():
    # Same 9000 of tech exposure, but the new trade is IWM (small_cap bucket),
    # so the tech bucket doesn't grow -- adding to a *different* bucket is fine.
    trade = make_trade(symbol="IWM", contracts=5, max_loss_per_contract=400.0)
    portfolio = make_portfolio(
        open_positions=(
            OpenPosition(symbol="QQQ", sleeve="A", max_loss=9000.0, net_delta=0.0, net_vega=0.0),
        )
    )
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert allowed
    assert any(c.name == "max_bucket_heat" and c.passed for c in checks)


def test_concentration_gate_boundary_exactly_at_cap_passes():
    # 8000 existing tech + 2000 new = 10000 = exactly 10% of 100k -> passes (<=).
    trade = make_trade(symbol="AAPL", contracts=5, max_loss_per_contract=400.0)
    portfolio = make_portfolio(
        open_positions=(
            OpenPosition(symbol="MSFT", sleeve="A", max_loss=8000.0, net_delta=0.0, net_vega=0.0),
        )
    )
    allowed, checks = gate_result(trade, portfolio, CONFIG)
    assert any(c.name == "max_bucket_heat" and c.passed for c in checks)


def test_concentration_gate_skipped_when_unconfigured():
    config = {k: dict(v) for k, v in CONFIG.items()}
    config["risk_gates"] = dict(CONFIG["risk_gates"])
    del config["risk_gates"]["max_bucket_heat_pct"]
    trade = make_trade(symbol="NVDA")
    portfolio = make_portfolio(
        open_positions=(
            OpenPosition(symbol="QQQ", sleeve="A", max_loss=9000.0, net_delta=0.0, net_vega=0.0),
        )
    )
    allowed, checks = gate_result(trade, portfolio, config)
    assert not any(c.name == "max_bucket_heat" for c in checks)
