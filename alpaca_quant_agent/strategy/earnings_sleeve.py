"""Sleeve B: earnings IV-crush micro-sleeve.

Sells a small defined-risk iron condor on curated single names when earnings
are imminent (0-2 trading days) and IV rank is very elevated (>=80th
percentile), harvesting the well-documented collapse in implied volatility
that follows an earnings announcement (Patell & Wolfson 1979, 1981, on
post-announcement volatility behavior). Position is closed the next session
regardless of outcome (config: sleeve_b_earnings.close_next_session) --
this sleeve never carries directional earnings risk overnight past the print.
"""
from __future__ import annotations

from datetime import date

from alpaca_quant_agent.strategy.screener import CandidateTrade, OptionQuote, build_iron_condor_candidate, select_expiration


def screen_earnings_candidate(
    *,
    symbol: str,
    chain: list[OptionQuote],
    today: date,
    earnings_date: date,
    iv_rank_value: float,
    equity: float,
    config: dict,
) -> CandidateTrade | None:
    cfg = config["strategy"]["sleeve_b_earnings"]
    days_to_earnings = (earnings_date - today).days

    if days_to_earnings < 0 or days_to_earnings > cfg["max_days_to_earnings"]:
        return None
    if iv_rank_value < cfg["min_iv_rank"]:
        return None

    expirations = sorted({q.expiration for q in chain})
    # Nearest expiration on/after the earnings date -- the condor must still
    # be open through the print to capture the IV crush.
    post_earnings_expirations = [e for e in expirations if e >= earnings_date]
    if not post_earnings_expirations:
        return None
    expiration = min(post_earnings_expirations)

    sizing_cfg = config["strategy"]["sleeve_a"]  # reuse delta target / width conventions
    sizing_kelly = config["sizing"]["kelly_fraction"]
    vrp_edge = config["sizing"]["assumed_win_prob_edge"]
    risk_cfg = config["risk_gates"]

    return build_iron_condor_candidate(
        symbol=symbol,
        sleeve="B",
        chain=chain,
        expiration=expiration,
        equity=equity,
        kelly_multiplier=sizing_kelly,
        max_risk_per_trade_pct=risk_cfg["max_risk_per_trade_pct"],
        short_delta_target=sizing_cfg["short_delta_target"],
        short_delta_band=sizing_cfg["short_delta_band"],
        width=sizing_cfg["spread_width_single_name"],
        days_to_earnings=days_to_earnings,
        vrp_edge=vrp_edge,
        rationale_hint=(
            f"Earnings IV-crush sleeve: {symbol} reports in {days_to_earnings}d, "
            f"IV rank {iv_rank_value:.0f} >= threshold {cfg['min_iv_rank']}; "
            f"selling iron condor expiring {expiration.isoformat()}, closes next session."
        ),
    )
