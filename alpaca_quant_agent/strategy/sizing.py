"""Fractional-Kelly position sizing for defined-risk credit spreads.

Kelly (1956) / Thorp's binary-bet formula: f* = p - (1-p)/b, where p is the
win probability and b is the reward/risk ratio (credit / max_loss for a
defined-risk spread). We use the standard options-desk approximation
p ~= 1 - |delta| of the short strike (probability the short strike expires
OTM), and always apply a fractional multiplier (config: sizing.kelly_fraction,
default 0.3) on top of full Kelly, since full Kelly is well known to be far
too aggressive under estimation error in p and b (Thorp's own practitioner
writing recommends 1/4-1/2 Kelly).

The final contract count is additionally hard-capped by the deterministic
max-risk-per-trade gate in risk/gates.py -- this module never has the final
say on size, only a starting proposal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def win_probability_from_delta(short_strike_delta: float) -> float:
    """Standard approximation: probability a short option expires OTM
    (and thus the spread reaches max profit) is ~= 1 - |delta|.

    Note: delta is a risk-neutral (implied-vol-derived) probability, so
    pricing built purely from it has ~zero edge by construction -- credit
    received and delta-implied loss probability are mutually consistent
    under the same implied-vol measure. The actual edge in premium-selling
    comes from the volatility risk premium (IV empirically exceeds
    subsequent realized vol on average -- Carr & Wu 2009; Bakshi & Kapadia
    2003), meaning realized win rates for short premium historically run
    above what delta alone implies. Callers should use
    win_probability_estimate() below, not this raw estimate, when sizing.
    """
    return 1.0 - abs(short_strike_delta)


def win_probability_estimate(short_strike_delta: float, vrp_edge: float) -> float:
    """Delta-implied win probability, adjusted upward by `vrp_edge` (config:
    sizing.assumed_win_prob_edge) to reflect the historically observed
    volatility risk premium -- i.e. that realized win rates for short
    premium exceed the risk-neutral (delta-implied) estimate on average.
    Clamped to [0, 0.999] to keep Kelly well-defined.
    """
    base = win_probability_from_delta(short_strike_delta)
    return min(0.999, max(0.0, base + vrp_edge))


def full_kelly_fraction(win_prob: float, credit_per_contract: float, max_loss_per_contract: float) -> float:
    """Returns the full-Kelly fraction of bankroll to risk on this bet.
    Clamped to [0, 1] -- a negative result means the bet has negative edge
    and should not be sized at all (screener should have already filtered
    this out, but sizing stays defensive).
    """
    if max_loss_per_contract <= 0:
        raise ValueError("max_loss_per_contract must be positive")
    b = credit_per_contract / max_loss_per_contract
    if b <= 0:
        return 0.0
    p = win_prob
    f_star = p - (1 - p) / b
    return max(0.0, min(1.0, f_star))


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    kelly_fraction_used: float
    dollars_at_risk: float


def contracts_for_trade(
    *,
    equity: float,
    win_prob: float,
    credit_per_contract: float,
    max_loss_per_contract: float,
    kelly_multiplier: float,
    max_risk_per_trade_pct: float,
) -> SizingResult:
    """Proposes a contract count sized to `kelly_multiplier` * full Kelly,
    hard-capped so dollars-at-risk never exceeds `max_risk_per_trade_pct`
    of equity regardless of what Kelly suggests.
    """
    if equity <= 0:
        return SizingResult(0, 0.0, 0.0)

    f_star = full_kelly_fraction(win_prob, credit_per_contract, max_loss_per_contract)
    fraction_used = f_star * kelly_multiplier
    fraction_used = min(fraction_used, max_risk_per_trade_pct)

    dollars_budget = equity * fraction_used
    contracts = math.floor(dollars_budget / max_loss_per_contract) if max_loss_per_contract > 0 else 0
    contracts = max(0, contracts)

    return SizingResult(
        contracts=contracts,
        kelly_fraction_used=fraction_used,
        dollars_at_risk=contracts * max_loss_per_contract,
    )
