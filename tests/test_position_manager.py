from datetime import date

import pytest

from alpaca_quant_agent.execution.position_manager import ManagedPosition, evaluate_exit

CONFIG = {
    "strategy": {
        "sleeve_a": {
            "take_profit_pct_of_credit": 0.50,
            "stop_loss_multiple_of_credit": 2.0,
            "time_stop_dte": 21,
            "earnings_blackout_days": 3,
        },
        "sleeve_b_earnings": {
            "close_next_session": True,
        },
    }
}

TODAY = date(2026, 9, 1)


def make_position(**overrides):
    defaults = dict(
        position_id="p1",
        symbol="SPY",
        sleeve="A",
        strategy_type="put_credit_spread",
        contracts=5,
        credit_received_per_contract=110.0,
        opened_at=date(2026, 8, 1),
        expiration=date(2026, 10, 16),
        days_to_earnings=None,
    )
    defaults.update(overrides)
    return ManagedPosition(**defaults)


def test_no_exit_when_nothing_triggered():
    position = make_position()
    decision = evaluate_exit(position, current_value_per_contract=90.0, today=TODAY, config=CONFIG)
    assert not decision.should_close


def test_take_profit_at_50_pct_captured():
    position = make_position(credit_received_per_contract=100.0)
    # value now 50 -> captured 50% of credit -> exactly at threshold, should close.
    decision = evaluate_exit(position, current_value_per_contract=50.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "take_profit"


def test_no_take_profit_just_below_threshold():
    position = make_position(credit_received_per_contract=100.0)
    decision = evaluate_exit(position, current_value_per_contract=51.0, today=TODAY, config=CONFIG)
    assert not decision.should_close


def test_stop_loss_at_2x_credit():
    position = make_position(credit_received_per_contract=100.0)
    # loss = 300 - 100 = 200 = 2x credit -> exactly at threshold, should close.
    decision = evaluate_exit(position, current_value_per_contract=300.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "stop_loss"


def test_time_stop_at_21_dte():
    position = make_position(expiration=date(2026, 9, 22))  # 21 days from TODAY
    decision = evaluate_exit(position, current_value_per_contract=90.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "time_stop"


def test_earnings_blackout_forces_close():
    position = make_position(days_to_earnings=2)
    decision = evaluate_exit(position, current_value_per_contract=90.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "earnings_blackout_close"


def test_take_profit_checked_before_earnings_blackout():
    # Both conditions true -> take_profit should win since it's checked first
    # (economically the more specific/urgent reason to lock in the win).
    position = make_position(credit_received_per_contract=100.0, days_to_earnings=1)
    decision = evaluate_exit(position, current_value_per_contract=40.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "take_profit"


def test_sleeve_b_closes_next_session_after_earnings():
    position = make_position(sleeve="B", strategy_type="iron_condor", days_to_earnings=0)
    decision = evaluate_exit(position, current_value_per_contract=90.0, today=TODAY, config=CONFIG)
    assert decision.should_close
    assert decision.reason == "earnings_sleeve_close_next_session"


def test_sleeve_b_holds_before_earnings():
    position = make_position(sleeve="B", strategy_type="iron_condor", days_to_earnings=1)
    decision = evaluate_exit(position, current_value_per_contract=90.0, today=TODAY, config=CONFIG)
    assert not decision.should_close


def test_zero_credit_never_closes():
    position = make_position(credit_received_per_contract=0.0)
    decision = evaluate_exit(position, current_value_per_contract=0.0, today=TODAY, config=CONFIG)
    assert not decision.should_close
