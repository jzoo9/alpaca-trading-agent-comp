"""CLI entrypoint.

    python -m alpaca_quant_agent.main --mode dry-run    # one cycle, no orders submitted
    python -m alpaca_quant_agent.main --mode run-once    # one cycle, real paper orders
    python -m alpaca_quant_agent.main --mode daemon      # autonomous market-hours loop
    python -m alpaca_quant_agent.main --mode dashboard   # local read-only monitoring UI
"""
from __future__ import annotations

import argparse
import asyncio
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous options-selling VRP-harvesting agent")
    parser.add_argument("--mode", choices=["dry-run", "run-once", "daemon", "dashboard"], required=True)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--port", type=int, default=8787, help="dashboard mode only")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="dashboard mode only; use 0.0.0.0 to accept connections from other machines (e.g. a teammate on the same network, or a hosted server)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.mode == "dashboard":
        # The dashboard itself needs no Alpaca/Featherless credentials for
        # viewing (read-only over the local ledger) -- only its write actions
        # (kill switch, run cycle, close position) call load_config() lazily.
        from alpaca_quant_agent.dashboard import run as run_dashboard
        run_dashboard(host=args.host, port=args.port)
        return

    from alpaca_quant_agent.config import load_config
    from alpaca_quant_agent.cycle import run_one_cycle

    config = load_config()

    if args.mode == "dry-run":
        summary = asyncio.run(run_one_cycle(config, dry_run=True))
        print(summary)
    elif args.mode == "run-once":
        summary = asyncio.run(run_one_cycle(config, dry_run=False))
        print(summary)
    else:
        from alpaca_quant_agent.scheduler import run_daemon
        asyncio.run(run_daemon(config))


if __name__ == "__main__":
    main()
