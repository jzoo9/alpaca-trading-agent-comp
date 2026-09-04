"""Market-hours-aware autonomous loop. Runs a full cycle (cycle.run_one_cycle)
every `scheduler.cycle_interval_minutes` while the market is open, and sleeps
between checks otherwise. Meant to be started once and left running -- see
launchd/ for macOS persistence across logout/reboot.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from alpaca_quant_agent.config import Config
from alpaca_quant_agent.cycle import run_one_cycle
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient

logger = logging.getLogger(__name__)


async def market_is_open(config: Config) -> bool:
    """Public so streamlit_app/app.py can reuse the same market-hours check
    the real daemon uses, instead of re-implementing it."""
    async with AlpacaMcpClient(config) as client:
        clock = await client.get_clock()
        return bool(clock.get("is_open"))


async def run_daemon(config: Config) -> None:
    interval_seconds = config.get("scheduler", "cycle_interval_minutes", default=15) * 60
    logger.info("starting autonomous daemon, cycle interval=%ss", interval_seconds)

    while True:
        try:
            if await market_is_open(config):
                logger.info("cycle starting at %s", datetime.now().isoformat())
                summary = await run_one_cycle(config)
                logger.info("cycle summary: %s", summary)
            else:
                logger.info("market closed, skipping cycle")
        except Exception:
            logger.exception("cycle failed -- will retry next interval rather than crash the daemon")
        await asyncio.sleep(interval_seconds)
