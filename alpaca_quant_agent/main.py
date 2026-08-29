"""CLI entrypoint.

    python -m alpaca_quant_agent.main --mode dry-run    # one cycle, no orders submitted
    python -m alpaca_quant_agent.main --mode run-once    # one cycle, real paper orders
    python -m alpaca_quant_agent.main --mode daemon      # autonomous market-hours loop
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from alpaca_quant_agent.config import load_config
from alpaca_quant_agent.cycle import run_one_cycle
from alpaca_quant_agent.scheduler import run_daemon


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous options-selling VRP-harvesting agent")
    parser.add_argument("--mode", choices=["dry-run", "run-once", "daemon"], required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()

    if args.mode == "dry-run":
        summary = asyncio.run(run_one_cycle(config, dry_run=True))
        print(summary)
    elif args.mode == "run-once":
        summary = asyncio.run(run_one_cycle(config, dry_run=False))
        print(summary)
    else:
        asyncio.run(run_daemon(config))


if __name__ == "__main__":
    main()
