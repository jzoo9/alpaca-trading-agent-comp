"""Pure, unit-testable signal functions: momentum, trend, regime, IV rank.

None of these functions touch the network or a database -- they take plain
pandas/numpy inputs and return plain values, so they can be exercised with
synthetic fixtures in tests/test_signals.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def momentum_12_1(close: pd.Series, lookback_days: int = 252, skip_days: int = 21) -> float:
    """Time-series momentum a la Moskowitz-Ooi-Pedersen (2012): total return
    over the trailing `lookback_days`, excluding the most recent `skip_days`
    (avoids the well-documented short-term reversal effect).

    Returns the cumulative return as a float, e.g. 0.15 == +15%.
    Raises ValueError if there isn't enough history.
    """
    if len(close) < lookback_days + 1:
        raise ValueError(
            f"need at least {lookback_days + 1} closes, got {len(close)}"
        )
    window = close.iloc[-(lookback_days + 1):]
    end_idx = len(window) - 1 - skip_days
    if end_idx <= 0:
        raise ValueError("skip_days too large relative to lookback_days")
    start_price = window.iloc[0]
    end_price = window.iloc[end_idx]
    return float(end_price / start_price - 1.0)


def ema_crossover_bias(close: pd.Series, fast: int = 20, slow: int = 50) -> str:
    """Returns 'bullish' | 'bearish' | 'neutral' based on fast vs slow EMA."""
    if len(close) < slow:
        raise ValueError(f"need at least {slow} closes, got {len(close)}")
    fast_ema = ema(close, fast).iloc[-1]
    slow_ema = ema(close, slow).iloc[-1]
    if fast_ema > slow_ema:
        return "bullish"
    if fast_ema < slow_ema:
        return "bearish"
    return "neutral"


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index. Returns a Series aligned to input index
    (leading `period` entries are NaN, matching standard ADX warm-up behavior).
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_series = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_series


def iv_rank(iv_history: list[float], current_iv: float) -> float:
    """Percentile rank (0-100) of current_iv within the trailing iv_history sample
    (history should NOT include current_iv; it's the lookback window).
    Uses a standard IV-Rank definition: (current - min) / (max - min) * 100,
    clamped to [0, 100]. Falls back to 50.0 (neutral) if history is empty or
    has zero range, since there is no basis to time entries yet.
    """
    if not iv_history:
        return 50.0
    lo = min(iv_history)
    hi = max(iv_history)
    if hi <= lo:
        return 50.0
    pct = (current_iv - lo) / (hi - lo) * 100.0
    return float(min(100.0, max(0.0, pct)))


def iv_percentile(iv_history: list[float], current_iv: float) -> float:
    """Alternative IV-Rank definition: percentage of trailing observations
    the current IV exceeds. Less sensitive to single outlier extremes than
    iv_rank(). Both are exposed; screener.py uses iv_rank() by default.
    """
    if not iv_history:
        return 50.0
    n_below = sum(1 for v in iv_history if v <= current_iv)
    return float(n_below / len(iv_history) * 100.0)


@dataclass(frozen=True)
class VolRegimeSignal:
    """Volatility-term-structure regime, used to scale gross short-premium
    exposure globally (strategy/screener sizing + heat budget).

    `ratio` is front-month vol / longer-dated vol (e.g. VIX / VIX3M, or the
    front two VIX futures). Short-volatility strategies are safe to press in
    *contango* (ratio < 1: near-term calm relative to later) and are the most
    dangerous in *backwardation* (ratio > 1: acute near-term stress), which
    is exactly the setup that has historically produced the catastrophic
    losses for premium sellers. `exposure_multiplier` in [0, 1] throttles new
    exposure accordingly.
    """
    ratio: float
    regime: str  # "contango" | "flat" | "backwardation"
    exposure_multiplier: float


def vol_term_structure_regime(
    front_vol: float,
    back_vol: float,
    *,
    contango_ratio: float = 0.95,
    backwardation_ratio: float = 1.00,
    floor_ratio: float = 1.10,
    min_multiplier: float = 0.0,
) -> VolRegimeSignal:
    """Maps a front/back volatility ratio to a global exposure multiplier via a
    single continuous, monotonically non-increasing linear ramp:

    - ratio <= `contango_ratio`  -> full exposure (multiplier 1.0), "contango"
    - ratio >= `floor_ratio`     -> `min_multiplier` (deepest throttle)
    - in between                 -> linearly interpolated 1.0 -> min_multiplier

    The regime *label* is purely cosmetic (for logs): "contango" below
    `contango_ratio`, "backwardation" at/above `backwardation_ratio`, "flat"
    between. The multiplier itself is one ramp with no discontinuities, so a
    slightly rising ratio never causes a jump in exposure.

    Defensive fallback (multiplier 1.0, "flat") if inputs are non-positive or
    the ramp is misconfigured (floor_ratio <= contango_ratio): a bad reading
    must never *increase* risk, and full-but-still-gated sizing is the safe
    default.
    """
    if front_vol <= 0 or back_vol <= 0:
        return VolRegimeSignal(ratio=0.0, regime="flat", exposure_multiplier=1.0)

    ratio = front_vol / back_vol

    # Misconfigured ramp (floor not strictly above contango) -> fail open,
    # checked up front so it can't be pre-empted by the boundary branches below.
    span = floor_ratio - contango_ratio
    if span <= 0:
        return VolRegimeSignal(ratio=ratio, regime="flat", exposure_multiplier=1.0)

    regime = _vol_regime_label(ratio, contango_ratio, backwardation_ratio)

    if ratio <= contango_ratio:
        return VolRegimeSignal(ratio=ratio, regime=regime, exposure_multiplier=1.0)
    if ratio >= floor_ratio:
        return VolRegimeSignal(ratio=ratio, regime=regime, exposure_multiplier=_clamp01(min_multiplier))

    frac = (ratio - contango_ratio) / span
    mult = 1.0 - frac * (1.0 - min_multiplier)
    return VolRegimeSignal(ratio=ratio, regime=regime, exposure_multiplier=_clamp01(mult))


def _vol_regime_label(ratio: float, contango_ratio: float, backwardation_ratio: float) -> str:
    if ratio <= contango_ratio:
        return "contango"
    if ratio >= backwardation_ratio:
        return "backwardation"
    return "flat"


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class RegimeSignal:
    direction: str  # "bullish" | "bearish" | "neutral"
    is_trending: bool  # True if ADX >= trend_threshold
    adx_value: float
    momentum_12_1: float


def classify_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    ema_fast: int = 20,
    ema_slow: int = 50,
    adx_period: int = 14,
    adx_trend_threshold: float = 20.0,
    momentum_lookback_days: int = 252,
    momentum_skip_days: int = 21,
) -> RegimeSignal:
    bias = ema_crossover_bias(close, ema_fast, ema_slow)
    adx_series = adx(high, low, close, adx_period)
    adx_value = float(adx_series.iloc[-1])
    is_trending = adx_value >= adx_trend_threshold

    try:
        mom = momentum_12_1(close, momentum_lookback_days, momentum_skip_days)
    except ValueError:
        mom = 0.0

    # Combine EMA-crossover bias and 12-1 momentum sign; disagreement -> neutral.
    mom_bias = "bullish" if mom > 0 else "bearish" if mom < 0 else "neutral"
    direction = bias if bias == mom_bias else "neutral"

    return RegimeSignal(
        direction=direction,
        is_trending=is_trending,
        adx_value=adx_value,
        momentum_12_1=mom,
    )
