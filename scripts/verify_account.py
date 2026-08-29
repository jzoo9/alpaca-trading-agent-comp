"""Confirms the Alpaca paper account is reachable through the MCP server and
reports equity / options trading level, so you can verify the $100,000
starting-balance requirement before the daemon starts trading.

Usage:
    python -m scripts.verify_account
"""
from __future__ import annotations

import asyncio

from alpaca_quant_agent.config import load_config
from alpaca_quant_agent.execution.alpaca_mcp import AlpacaMcpClient


async def main() -> None:
    config = load_config()
    async with AlpacaMcpClient(config) as client:
        account = await client.get_account()
        clock = await client.get_clock()

    print("=== Alpaca paper account ===")
    print(f"  status:            {account.get('status')}")
    print(f"  equity:            {account.get('equity')}")
    print(f"  cash:              {account.get('cash')}")
    print(f"  buying_power:      {account.get('buying_power')}")
    print(f"  options_trading_level: {account.get('options_trading_level')}")
    print(f"  pattern_day_trader: {account.get('pattern_day_trader')}")
    print("=== Market clock ===")
    print(f"  is_open:           {clock.get('is_open')}")
    print(f"  next_open:         {clock.get('next_open')}")
    print(f"  next_close:        {clock.get('next_close')}")

    equity = float(account.get("equity") or 0)
    if abs(equity - 100_000.0) > 1.0:
        print(
            "\n WARNING: equity is not $100,000. Reset the paper account balance "
            "from the Alpaca dashboard (Paper Trading -> Reset Account -> set to "
            "$100,000) before starting the competition run."
        )
    else:
        print("\n OK: starting balance is $100,000 as required.")

    level = account.get("options_trading_level")
    if level is not None and int(level) < 3:
        print(
            " WARNING: options trading level is below 3 -- multi-leg spreads/condors "
            "will be rejected. Paper accounts should auto-approve level 3; if not, "
            "check the Alpaca dashboard options settings."
        )


if __name__ == "__main__":
    asyncio.run(main())
