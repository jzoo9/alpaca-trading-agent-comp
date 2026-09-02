# Autonomous Options-Selling Agent (Alpaca Paper Trading)

A volatility-risk-premium-harvesting options agent: sells defined-risk credit
spreads and iron condors on a curated, sector-diversified universe of liquid
ETFs and quality large-caps (22 symbols across tech, financials, healthcare,
consumer, energy, and gold), gated by deterministic risk rules, with a bounded
LLM reasoning layer for catalyst awareness and trade rationale. See
`WRITEUP.md` for the full one-page summary of strategy, risk gates, and
infrastructure.

## 1. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (provides `uvx`, used to launch `alpaca-mcp-server`):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- An Alpaca account with a **paper trading** API key pair, and a
  [Featherless AI](https://featherless.ai) API key (OpenAI-compatible;
  powers the trade-selection reasoning layer).

## 2. Create your Alpaca paper account + API keys

1. Sign up / log in at https://app.alpaca.markets
2. Go to the **Paper Trading** dashboard (top-left environment switch).
3. Under "Reset Account", set the starting balance to **$100,000** (required
   for the competition) if it isn't already.
4. Go to **API Keys** (still in the paper environment) and generate a key
   pair. Copy the Key ID and Secret -- the secret is only shown once.

## 3. Configure

```bash
cp .env.example .env
# edit .env: ALPACA_API_KEY, ALPACA_SECRET_KEY, FEATHERLESS_API_KEY
```

Get a Featherless API key at https://featherless.ai (Settings -> API Keys).
`FEATHERLESS_MODEL` defaults to `moonshotai/Kimi-K2-Instruct`, verified
live to handle this project's tool-calling shape reliably in ~2-3s per
call. A Qwen3-30B-A3B-Instruct-2507 model, despite being in Featherless's
documented tool-calling-capable family, either 500'd or hung indefinitely
once a realistic tool schema was attached -- if you switch models, re-verify
tool-calling latency/reliability directly before trusting it in the daemon
(a plain completion succeeding is not evidence tool-calling works).

Strategy/universe/risk-limit constants live in `config.yaml` -- all the
numeric risk gates described in `WRITEUP.md` are there, not hardcoded.
`earnings_calendar.yaml` needs occasional manual updates (see the comment
at the top of that file / the "Known limitations" section of `WRITEUP.md`).

## 4. Install and test

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q                      # 97 unit tests: signals/gates/sizing/screener/exits/hedge/black-scholes/universe
```

## 5. Verify the account is wired up correctly

```bash
python -m scripts.verify_account
```

Confirms the MCP server can reach your paper account, prints equity /
options trading level, and warns if equity isn't $100,000 or options level
isn't 3 (multi-leg spreads).

## 6. Dry run (no orders placed)

```bash
python -m alpaca_quant_agent.main --mode dry-run
```

Runs one full cycle against live paper-market data -- real option chains,
real IV, real news -- but every "would open" / "would close" decision is
only logged to the SQLite ledger (`./data/agent.db`), not sent to Alpaca.
Inspect the `decisions` table before going live:

```bash
sqlite3 data/agent.db "select decision, symbol, detail from decisions order by created_at desc limit 20;"
```

## 7. One live cycle

```bash
python -m alpaca_quant_agent.main --mode run-once
```

Places real (paper) orders. Check the Alpaca paper dashboard for the
resulting positions.

## 8. Run the autonomous daemon

Foreground (simplest, good for the first day of the competition so you can
watch the logs):

```bash
python -m alpaca_quant_agent.main --mode daemon
```

Persistent across logout/reboot (macOS `launchd`):

```bash
mkdir -p logs
# edit launchd/com.alpacaquantagent.daemon.plist if your repo path differs
cp launchd/com.alpacaquantagent.daemon.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.alpacaquantagent.daemon.plist
# to stop:
launchctl unload ~/Library/LaunchAgents/com.alpacaquantagent.daemon.plist
```

The daemon checks the market clock every cycle and only trades during
regular market hours; it sleeps otherwise and never crashes out on a
single cycle's exception (logs and retries next interval).

## Project layout

See `WRITEUP.md` for the strategy/risk narrative, and the module
docstrings in `alpaca_quant_agent/` for how each piece fits together --
`cycle.py` is the best starting point to trace one full cycle end to end.
