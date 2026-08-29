"""Static, curated trading universe.

The core ETFs are the primary book: maximal liquidity, no idiosyncratic
earnings/news risk, tight option spreads. Single names are a small,
hand-picked "quality / low volatility" tilt (Asness 2019 "Quality Minus
Junk"; Ang, Hodrick, Xing & Zhang 2006 low-volatility anomaly) used only
for Sleeve A diversification and the Sleeve B earnings-IV-crush trade.
This list is intentionally static and reviewed manually, not algorithmically
screened, because Alpaca's MCP/API surface does not expose deep fundamentals
(balance sheet, earnings quality) data -- the live news check in agent/tools.py
is the dynamic complement to this static quality screen.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseEntry:
    symbol: str
    kind: str  # "etf" | "single_name"
    rationale: str


CORE_ETFS: tuple[UniverseEntry, ...] = (
    UniverseEntry("SPY", "etf", "Broad market beta, deepest options liquidity on the exchange."),
    UniverseEntry("QQQ", "etf", "Large-cap tech/growth beta, very liquid options chain."),
    UniverseEntry("IWM", "etf", "Small-cap beta, diversifies away from large-cap-only exposure."),
)

QUALITY_SINGLE_NAMES: tuple[UniverseEntry, ...] = (
    UniverseEntry("AAPL", "single_name", "Profitable, low leverage, dominant liquid options chain."),
    UniverseEntry("MSFT", "single_name", "High-quality balance sheet, durable cash flows, diversified revenue."),
    UniverseEntry("NVDA", "single_name", "Structural growth leader; included for premium/IV richness despite higher vol."),
    UniverseEntry("GOOGL", "single_name", "High margin, net-cash balance sheet, deep liquidity."),
    UniverseEntry("AMZN", "single_name", "Market leader in two durable franchises (retail + AWS cash engine)."),
    UniverseEntry("META", "single_name", "High margin, strong free cash flow, buyback discipline."),
    UniverseEntry("AVGO", "single_name", "Profitable, diversified semis/software cash flows, low relative volatility."),
)

ALL_ENTRIES: tuple[UniverseEntry, ...] = CORE_ETFS + QUALITY_SINGLE_NAMES
SYMBOLS: tuple[str, ...] = tuple(e.symbol for e in ALL_ENTRIES)


def is_etf(symbol: str) -> bool:
    return any(e.symbol == symbol and e.kind == "etf" for e in ALL_ENTRIES)


def rationale_for(symbol: str) -> str | None:
    for e in ALL_ENTRIES:
        if e.symbol == symbol:
            return e.rationale
    return None
