import pytest

from alpaca_quant_agent.strategy.black_scholes import (
    bs_delta,
    bs_price,
    bs_vega,
    implied_volatility,
)

# Textbook reference case (Hull-style): S=100, K=100, T=1y, r=5%, sigma=20%.
# Known closed-form call price ~= 10.4506, put price ~= 5.5735 (put-call parity).
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20


def test_call_price_matches_known_value():
    assert bs_price("call", S, K, T, R, SIGMA) == pytest.approx(10.4506, abs=1e-3)


def test_put_price_matches_known_value():
    assert bs_price("put", S, K, T, R, SIGMA) == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    call = bs_price("call", S, K, T, R, SIGMA)
    put = bs_price("put", S, K, T, R, SIGMA)
    import math
    assert call - put == pytest.approx(S - K * math.exp(-R * T), abs=1e-6)


def test_call_delta_between_0_and_1():
    d = bs_delta("call", S, K, T, R, SIGMA)
    assert 0.0 < d < 1.0
    assert d == pytest.approx(0.6368, abs=1e-3)


def test_put_delta_between_minus1_and_0():
    d = bs_delta("put", S, K, T, R, SIGMA)
    assert -1.0 < d < 0.0
    assert d == pytest.approx(-0.3632, abs=1e-3)


def test_deep_itm_call_delta_near_1():
    d = bs_delta("call", S=200.0, K=100.0, T=0.25, r=R, sigma=SIGMA)
    assert d > 0.95


def test_deep_otm_call_delta_near_0():
    d = bs_delta("call", S=50.0, K=100.0, T=0.25, r=R, sigma=SIGMA)
    assert d < 0.05


def test_vega_positive():
    assert bs_vega(S, K, T, R, SIGMA) > 0


def test_implied_volatility_recovers_input_sigma():
    price = bs_price("call", S, K, T, R, SIGMA)
    iv = implied_volatility("call", price, S, K, T, R)
    assert iv == pytest.approx(SIGMA, abs=1e-4)


def test_implied_volatility_recovers_input_sigma_put():
    price = bs_price("put", S, K, T, R, SIGMA)
    iv = implied_volatility("put", price, S, K, T, R)
    assert iv == pytest.approx(SIGMA, abs=1e-4)


def test_implied_volatility_none_for_unachievable_price():
    # A price far above any achievable value at low vol bounds -> None.
    iv = implied_volatility("call", price=99.0, S=100.0, K=100.0, T=0.01, r=R, high=1.0)
    assert iv is None


def test_implied_volatility_none_for_nonpositive_inputs():
    assert implied_volatility("call", price=0.0, S=100.0, K=100.0, T=1.0, r=R) is None
    assert implied_volatility("call", price=5.0, S=100.0, K=100.0, T=0.0, r=R) is None
