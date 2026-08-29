# VRP-Harvesting Options Agent — Write-up

**Account:** Alpaca paper trading, $100,000 starting balance, options level 3 (auto-approved on paper).

## AI logic

**Core thesis.** Implied volatility systematically exceeds subsequently realized
volatility — the volatility risk premium (Bakshi & Kapadia 2003; Carr & Wu
2009; the long-run outperformance of CBOE's BXM/PUT premium-selling indices).
The agent harvests this by selling defined-risk credit spreads and iron
condors, but only when three literature-grounded filters agree — naive
constant short-vol is well known to blow up:

1. **IV Rank timing** — enter only when a symbol's 63-day IV percentile is
   elevated (≥40th, ≥80th for the earnings sleeve); otherwise stand aside.
2. **Regime-gated direction** — 12-1 month time-series momentum
   (Moskowitz–Ooi–Pedersen 2012) plus a 20/50 EMA crossover sets directional
   bias; ADX(14) < 20 (no clear trend) switches the trade from a directional
   credit spread to an iron condor, avoiding naked-directional premium
   selling into a real trend.
3. **Fundamental/quality tilt** — beyond SPY/QQQ/IWM, single names are
   limited to a small static curated quality/low-vol large-cap list (Asness
   2019 "Quality Minus Junk"; Ang et al. low-vol anomaly), each with a
   documented rationale (`universe.py`). A live news check before every
   entry is the dynamic complement.

Two sleeves: **Sleeve A** (core) sells 30–45 DTE spreads/condors at ~0.22
short delta, exits at 50% of max profit captured / 2× credit stop-loss / 21
DTE time-stop / earnings blackout. **Sleeve B** (small, capped) sells a
tight iron condor ahead of earnings when IV rank ≥80th, closing the next
session to harvest the well-documented post-earnings IV crush (Patell &
Wolfson 1979/1981). Sizing is fractional Kelly (Thorp) off a VRP-adjusted
win probability, always capped by the hard per-trade risk gate below.

**The LLM's role is deliberately narrow.** Each cycle, deterministic code
(`strategy/screener.py`, `risk/gates.py`) produces a list of already-approved,
fully-specified candidates. A Featherless AI-hosted open-weight model
(Qwen3 family / Kimi-K2, OpenAI-compatible tool calling, `agent/brain.py`)
can only: check live news/macro catalysts on candidate underlyings via a
read-only subset of `alpaca-mcp-server`'s tools, select which candidates to
take within the remaining risk budget, write the trade rationale, and skip
anything it judges unwise. Its *only* write tool is
`submit_approved_trade(candidate_id, rationale)` (`agent/tools.py`) — it
cannot author strikes, quantity, or price, and never receives Alpaca's raw
order-placement tool. `submit_approved_trade` re-validates every risk gate
server-side before submitting, including against trades already taken
earlier in the same cycle.

## Risk gates (deterministic, unit-tested — `risk/gates.py`, `risk/circuit_breaker.py`)

| Gate | Limit |
|---|---|
| Max risk per trade | 2% of equity |
| Max portfolio heat | 20% of equity |
| Positions per underlying / total | 1 / 12 |
| Sleeve B allocation cap | 15% of equity |
| Portfolio delta / vega bands | 5% / 3% of equity (breach → protective SPY put hedge) |
| Daily loss halt | −3% → no new entries today |
| Total drawdown kill switch | −10% from peak → manage-to-close only |
| Liquidity floor | min open interest, max bid-ask spread |
| Earnings blackout | Sleeve A skips/closes within 3 days of earnings |

All 52 unit tests (`tests/`) exercise these — including exact-boundary cases
— plus the momentum/ADX/IV-rank signal math, Kelly sizing, and exit rules,
independent of any live connection. A `--dry-run` mode runs the full cycle
against live paper-market data (real chains, IV, news) with no orders sent,
for pre-flight validation.

## Alpaca infrastructure

`alpaca-mcp-server` is the sole execution/data surface — no raw Alpaca SDK
calls. Two uses of the same server: (1) a deterministic MCP client
(`execution/alpaca_mcp.py`) drives all data pulls and every order
submission, invoked by the scheduler and by the bounded tool handlers in
`agent/tools.py`; (2) a restricted read-only subset of the same server's
tools (news, account, positions, clock — `LLM_DATA_TOOL_NAMES`) is exposed
to the Featherless reasoning model as OpenAI-style function-calling tools —
it can look things up live but never place an order directly. A
market-hours-aware daemon (`scheduler.py`) runs a full cycle
every 15 minutes, independent of any interactive session (macOS `launchd`
unit provided for persistence). All trades, gate/skip decisions, IV history,
and daily equity snapshots are logged to SQLite (`ledger.py`) — the system
of record for this write-up's eventual performance section.

**Known limitation:** Alpaca's API/MCP surface doesn't expose a verified
forward earnings calendar, so the earnings blackout/sleeve keys off a small
hand-maintained `earnings_calendar.yaml` rather than a live feed.
