from datetime import date

import pytest

from alpaca_quant_agent.data.option_quotes import build_option_quotes
from alpaca_quant_agent.strategy.black_scholes import bs_price, bs_vega

TODAY = date(2026, 9, 1)
EXPIRATION = "2026-10-16"  # 45 days out


def make_contract(symbol, strike, option_type, open_interest=500):
    return {
        "symbol": symbol,
        "underlying_symbol": "SPY",
        "strike_price": str(strike),
        "expiration_date": EXPIRATION,
        "type": option_type,
        "open_interest": str(open_interest),
    }


def make_snapshot(bid, ask):
    return {"latestQuote": {"bp": bid, "ap": ask}}


def test_builds_quote_with_recovered_iv_and_sane_delta():
    S = 500.0
    K = 490.0
    T = 45 / 365.0
    r = 0.045
    true_iv = 0.20
    fair_price = bs_price("put", S, K, T, r, true_iv)

    contracts = {"SPY261016P00490000": make_contract("SPY261016P00490000", 490.0, "put")}
    chain = {"SPY261016P00490000": make_snapshot(fair_price - 0.02, fair_price + 0.02)}

    quotes = build_option_quotes(contracts, chain, S, TODAY, risk_free_rate=r)
    assert len(quotes) == 1
    q = quotes[0]
    assert q.iv == pytest.approx(true_iv, abs=0.01)
    assert -1.0 < q.delta < 0.0  # put delta is negative
    assert q.strike == 490.0
    assert q.option_type == "put"
    assert q.open_interest == 500


def test_vega_is_scaled_per_1_vol_point_not_per_100pct_move():
    # Regression test: bs_vega() returns $ sensitivity per 1.00 (100%)
    # absolute vol change; build_option_quotes must convert this to the
    # conventional "per 1 vol point (1%)" figure that risk_gates.py's
    # portfolio_vega_cap_pct is calibrated against -- i.e. divide by 100.
    # A prior bug skipped this conversion, making every portfolio vega
    # check ~100x too strict and silently rejecting every real candidate.
    S, K, T, r = 500.0, 490.0, 45 / 365.0, 0.045
    true_iv = 0.20
    fair_price = bs_price("put", S, K, T, r, true_iv)
    raw_vega = bs_vega(S, K, T, r, true_iv)

    contracts = {"X": make_contract("X", 490.0, "put")}
    chain = {"X": make_snapshot(fair_price - 0.02, fair_price + 0.02)}

    quotes = build_option_quotes(contracts, chain, S, TODAY, risk_free_rate=r)
    assert len(quotes) == 1
    assert quotes[0].vega == pytest.approx(raw_vega / 100.0, rel=0.05)
    assert quotes[0].vega == pytest.approx(raw_vega * 0.01, rel=0.05)  # not raw_vega itself


def test_skips_contract_with_no_matching_chain_entry():
    contracts = {"X": make_contract("X", 100.0, "call")}
    chain = {}  # no market data for this contract
    quotes = build_option_quotes(contracts, chain, 100.0, TODAY)
    assert quotes == []


def test_skips_contract_with_zero_or_crossed_quote():
    contracts = {"X": make_contract("X", 100.0, "call")}
    chain_zero = {"X": make_snapshot(0.0, 0.0)}
    assert build_option_quotes(contracts, chain_zero, 100.0, TODAY) == []

    chain_crossed = {"X": make_snapshot(5.0, 4.0)}  # bid > ask
    assert build_option_quotes(contracts, chain_crossed, 100.0, TODAY) == []


def test_skips_already_expired_contract():
    contract = make_contract("X", 100.0, "call")
    contract["expiration_date"] = "2026-08-01"  # before TODAY
    contracts = {"X": contract}
    chain = {"X": make_snapshot(1.0, 1.2)}
    assert build_option_quotes(contracts, chain, 100.0, TODAY) == []


def test_skips_contract_with_unachievable_price():
    # A wildly mispriced quote (price far exceeds any achievable BS value at
    # reasonable vol) should be skipped, not silently produce a garbage IV.
    contracts = {"X": make_contract("X", 100.0, "call")}
    chain = {"X": make_snapshot(999.0, 999.5)}
    assert build_option_quotes(contracts, chain, 100.0, TODAY) == []


def test_multiple_contracts_only_matching_ones_returned():
    contracts = {
        "A": make_contract("A", 490.0, "put"),
        "B": make_contract("B", 495.0, "put"),
    }
    fair_a = bs_price("put", 500.0, 490.0, 45 / 365.0, 0.045, 0.20)
    chain = {
        "A": make_snapshot(fair_a - 0.02, fair_a + 0.02),
        # "B" has no chain entry -> skipped
    }
    quotes = build_option_quotes(contracts, chain, 500.0, TODAY)
    assert len(quotes) == 1
    assert quotes[0].occ_symbol == "A"
