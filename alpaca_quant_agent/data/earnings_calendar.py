"""Upcoming-earnings lookup for the earnings blackout (Sleeve A) and the
earnings IV-crush sleeve (Sleeve B).

Alpaca's MCP/CLI surface exposes historical corporate actions but not a
verified, reliably-schemad *forward* earnings calendar tool, so rather than
build on a guessed tool name that might not exist, this reads a small
hand-maintained YAML file (earnings_calendar.yaml at repo root). This is a
deliberate, documented scope limitation -- see WRITEUP.md -- and is the one
piece of the pipeline that needs a weekly manual touch during the
competition. Swapping in a live data source later only requires changing
this one function.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "earnings_calendar.yaml"


def _load(path: Path) -> dict[str, list[date]]:
    if not path.exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {
        symbol: [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
        for symbol, dates in raw.items()
    }


def next_earnings_date(symbol: str, today: date, path: Path = DEFAULT_PATH) -> date | None:
    calendar = _load(path)
    upcoming = sorted(d for d in calendar.get(symbol, []) if d >= today)
    return upcoming[0] if upcoming else None


def days_to_earnings(symbol: str, today: date, path: Path = DEFAULT_PATH) -> int | None:
    next_date = next_earnings_date(symbol, today, path)
    return (next_date - today).days if next_date else None
