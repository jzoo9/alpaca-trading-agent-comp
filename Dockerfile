# Single-process deploy: the trading daemon and the dashboard run together
# (alpaca_quant_agent/serve.py) so they share one disk (/data) for the
# SQLite ledger. Portable across Railway / Render / Fly.io -- all three can
# build straight from this file.
FROM python:3.11-slim

# alpaca_quant_agent/execution/alpaca_mcp.py shells out to `uvx` to launch
# alpaca-mcp-server -- `uv` provides that binary.
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml ./
COPY alpaca_quant_agent ./alpaca_quant_agent
COPY config.yaml earnings_calendar.yaml ./

RUN pip install --no-cache-dir .

# The ledger (data/agent.db) lives here -- mount a persistent volume at this
# path on whichever platform you deploy to, or trades/decisions/equity
# history are lost on every restart.
ENV AGENT_DB_PATH=/data/agent.db
VOLUME ["/data"]

EXPOSE 8787
CMD ["python", "-m", "alpaca_quant_agent.serve"]
