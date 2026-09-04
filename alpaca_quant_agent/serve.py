"""Combined entrypoint for hosting: runs the autonomous trading daemon and
the monitoring dashboard in one process, sharing one disk. Built for
platforms that expect a single process listening on a single port (Railway,
Render, Fly.io all inject $PORT) -- see README.md "Hosting" section.

    python -m alpaca_quant_agent.serve

Unlike `main.py --mode daemon` (daemon only) or `--mode dashboard`
(dashboard only, no credentials needed), this needs the full .env / platform
environment variables since it actually runs the trading loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from alpaca_quant_agent.dashboard import run as run_dashboard

    port = int(os.environ.get("PORT", 8787))
    dashboard_thread = threading.Thread(
        target=run_dashboard, kwargs={"host": "0.0.0.0", "port": port}, daemon=True,
    )
    dashboard_thread.start()
    logger.info("dashboard serving on 0.0.0.0:%d", port)

    from alpaca_quant_agent.config import load_config
    from alpaca_quant_agent.scheduler import run_daemon

    config = load_config()
    asyncio.run(run_daemon(config))


if __name__ == "__main__":
    main()
