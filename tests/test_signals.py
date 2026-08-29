import numpy as np
import pandas as pd
import pytest

from alpaca_quant_agent.strategy.signals import (
    adx,
    classify_regime,
    ema_crossover_bias,
    iv_percentile,
    iv_rank,
    momentum_12_1,
)


def _uptrend_series(n=300, start=100.0, daily_drift=0.003, seed=1):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.001, n)
    returns = daily_drift + noise
    close = pd.Series(start * np.cumprod(1 + returns))
    return close


def _flat_noisy_series(n=300, start=100.0, seed=2):
    # Cumulative-product / smooth-oscillation series both have persistent
    # local directional runs, which is exactly what ADX is designed to pick
    # up -- so they aren't a reliable "no trend" fixture. Stationary i.i.d.
    # noise around a constant level is: direction reverses roughly every
    # other day, so no rolling window shows sustained directional movement.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.3, n)
    close = pd.Series(start + noise)
    return close


def test_momentum_12_1_positive_for_uptrend():
    close = _uptrend_series()
    mom = momentum_12_1(close, lookback_days=252, skip_days=21)
    assert mom > 0


def test_momentum_12_1_raises_on_insufficient_history():
    close = pd.Series(np.linspace(100, 110, 50))
    with pytest.raises(ValueError):
        momentum_12_1(close, lookback_days=252, skip_days=21)


def test_ema_crossover_bias_bullish_for_uptrend():
    close = _uptrend_series(n=100)
    assert ema_crossover_bias(close, fast=20, slow=50) == "bullish"


def test_ema_crossover_bias_bearish_for_downtrend():
    close = _uptrend_series(n=100, daily_drift=-0.003)
    assert ema_crossover_bias(close, fast=20, slow=50) == "bearish"


def test_adx_higher_for_trending_than_flat():
    n = 100
    trend_close = _uptrend_series(n=n, daily_drift=0.004)
    flat_close = _flat_noisy_series(n=n)

    # Build synthetic high/low bands around close for both series.
    def bands(close):
        high = close * 1.003
        low = close * 0.997
        return high, low

    trend_high, trend_low = bands(trend_close)
    flat_high, flat_low = bands(flat_close)

    trend_adx = adx(trend_high, trend_low, trend_close, period=14).iloc[-1]
    flat_adx = adx(flat_high, flat_low, flat_close, period=14).iloc[-1]

    assert trend_adx > flat_adx


def test_iv_rank_boundaries():
    history = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert iv_rank(history, 50.0) == 100.0
    assert iv_rank(history, 10.0) == 0.0
    assert iv_rank(history, 30.0) == pytest.approx(50.0)


def test_iv_rank_empty_history_is_neutral():
    assert iv_rank([], 25.0) == 50.0


def test_iv_rank_zero_range_is_neutral():
    assert iv_rank([20.0, 20.0, 20.0], 20.0) == 50.0


def test_iv_percentile_counts_below_or_equal():
    history = [10.0, 20.0, 30.0, 40.0]
    assert iv_percentile(history, 25.0) == 50.0  # 2 of 4 are <= 25
    assert iv_percentile(history, 5.0) == 0.0
    assert iv_percentile(history, 40.0) == 100.0


def test_classify_regime_bullish_trending():
    n = 300
    close = _uptrend_series(n=n, daily_drift=0.004)
    high = close * 1.003
    low = close * 0.997
    regime = classify_regime(high, low, close)
    assert regime.direction == "bullish"
    assert regime.momentum_12_1 > 0


def test_classify_regime_neutral_when_flat():
    n = 300
    close = _flat_noisy_series(n=n)
    high = close * 1.003
    low = close * 0.997
    regime = classify_regime(high, low, close)
    assert regime.is_trending is False
