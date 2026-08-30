import pytest

from alpaca_quant_agent.risk.gates import OpenPosition, PortfolioState
from alpaca_quant_agent.risk.hedge import HedgeOrder, compute_delta_hedge, hedge_shares

CONFIG = {
    "risk_gates": {
        "portfolio_delta_band_pct": 0.05,
        "portfolio_vega_cap_pct": 0.03,
    }
}


def make_portfolio(**overrides):
    defaults = dict(
        equity=100_000.0,
        equity_peak=100_000.0,
        daily_pnl_pct=0.0,
        open_positions=(),
    )
    defaults.update(overrides)
    return PortfolioState(**defaults)


def test_no_hedge_when_delta_within_band():
    # net delta 4000 = 4% of 100k, band is 5% -> no hedge.
    portfolio = make_portfolio(
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=4000.0, net_vega=0.0),)
    )
    order = compute_delta_hedge(portfolio, CONFIG)
    assert not order.needed
    assert order.shares == 0
    assert not order.delta_breach


def test_hedge_sells_spy_when_book_is_net_long_delta():
    # net delta 8000 = 8% > 5% band -> breach, book is long -> sell SPY.
    portfolio = make_portfolio(
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=8000.0, net_vega=0.0),)
    )
    order = compute_delta_hedge(portfolio, CONFIG)
    assert order.needed
    assert order.delta_breach
    assert order.side == "sell"
    # net_delta is share-equivalents; 8000 -> 8000 shares (no price division).
    assert order.shares == 8000


def test_hedge_buys_spy_when_book_is_net_short_delta():
    portfolio = make_portfolio(
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=-8000.0, net_vega=0.0),)
    )
    order = compute_delta_hedge(portfolio, CONFIG)
    assert order.needed
    assert order.side == "buy"


def test_vega_breach_flagged_even_when_delta_ok():
    # delta fine (0), but vega 4000 = 4% > 3% cap -> vega_breach True, no share hedge.
    portfolio = make_portfolio(
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=0.0, net_vega=4000.0),)
    )
    order = compute_delta_hedge(portfolio, CONFIG)
    assert not order.needed  # shares don't hedge vega
    assert order.vega_breach
    assert not order.delta_breach


def test_hedge_shares_offsets_share_equivalents_directly():
    # net_delta is already in share-equivalents -> one share per unit.
    assert hedge_shares(8000.0) == 8000
    # negative net delta uses magnitude.
    assert hedge_shares(-8000.0) == 8000
    # rounds to nearest whole share.
    assert hedge_shares(8000.4) == 8000
    assert hedge_shares(8000.6) == 8001


def test_no_hedge_on_nonpositive_equity():
    portfolio = make_portfolio(equity=0.0,
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=8000.0, net_vega=0.0),))
    order = compute_delta_hedge(portfolio, CONFIG)
    assert not order.needed


def test_hedge_boundary_exactly_at_band_no_hedge():
    # exactly 5% -> not a breach (uses strict > like the circuit breaker).
    portfolio = make_portfolio(
        open_positions=(OpenPosition("SPY", "A", max_loss=0.0, net_delta=5000.0, net_vega=0.0),)
    )
    order = compute_delta_hedge(portfolio, CONFIG)
    assert not order.needed
