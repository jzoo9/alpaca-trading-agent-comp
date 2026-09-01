"""Wiring for the external Alpaca MCP server (alpaca-mcp-server).

Two distinct uses of this same server, per the plan's "LLM's bounded role":

1. `AlpacaMcpClient` -- a raw MCP client session (no LLM involved) used by
   the deterministic scheduler/screener/position-manager to pull market data
   and, crucially, to submit orders. This is invoked directly by our own
   Python code, and also from inside agent/tools.py's `submit_approved_trade`
   handler -- i.e. order placement always goes through here, never through
   an LLM-authored tool call with free-form parameters.
2. The same server, via the same `AlpacaMcpClient`, is used by agent/brain.py
   to expose a small read-only subset of tools (news, account, positions,
   clock -- see LLM_DATA_TOOL_NAMES) to the Featherless-hosted reasoning
   model as OpenAI-style function-calling tools. Order-placement tools are
   never included in that subset -- the LLM's only path to placing an order
   is agent/tools.py's `submit_approved_trade`, a locally-defined tool that
   looks up a pre-computed, pre-gated order server-side.

Every tool name and request/response shape below was verified against a
live paper account (alpaca-mcp-server 2.3.0 / fastmcp 3.4.7), not guessed --
see the module-level notes on each method for what the real payload looks
like, since several diverge from what their tool descriptions alone would
suggest (e.g. every response is wrapped in `{"data": {...}}`, and
`get_option_chain`/`get_option_snapshot` carry no greeks or IV at all on
the free indicative feed -- see strategy/black_scholes.py).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alpaca_quant_agent.config import Config


def _alpaca_env(config: Config) -> dict[str, str]:
    return {
        "ALPACA_API_KEY": config.creds.alpaca_api_key,
        "ALPACA_SECRET_KEY": config.creds.alpaca_secret_key,
        "ALPACA_PAPER_TRADE": "true" if config.creds.alpaca_paper else "false",
    }


def alpaca_stdio_params(config: Config) -> StdioServerParameters:
    # alpaca-mcp-server (as of 2.3.0) declares `fastmcp>=3.1.0` with no upper
    # bound; fastmcp 4.0.0 renamed/removed fastmcp.tools.tool.ToolResult,
    # which alpaca-mcp-server imports at startup, so an unpinned `uvx` install
    # resolves a fastmcp that breaks it. Pin fastmcp <4 in the ephemeral uvx
    # environment until upstream tightens its own constraint.
    return StdioServerParameters(
        command="uvx",
        args=["--with", "fastmcp<4.0.0", "alpaca-mcp-server"],
        env=_alpaca_env(config),
    )


# Read-only tool names the reasoning model (agent/brain.py) is allowed to
# call directly during its turn. Order-placement tools are deliberately
# excluded -- see module docstring. (Real tool names, verified live --
# NOT get_account/get_positions/cancel_order as an earlier draft assumed.)
LLM_DATA_TOOL_NAMES = ["get_account_info", "get_all_positions", "get_news", "get_clock"]


def _extract_result(result: Any) -> Any:
    """Unwraps an mcp.types.CallToolResult into plain Python data, then
    unwraps alpaca-mcp-server's own response envelope. Every verified
    response from this server has the shape
    `{"_alpaca_mcp_security": {...}, "data": <actual payload>}` -- the
    security block is a static trust-annotation, not data, so callers
    throughout this codebase should only ever see `<actual payload>`.
    """
    structured = getattr(result, "structured_content", None)
    parsed = structured
    if parsed is None:
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    parsed = text
                break

    if isinstance(parsed, dict) and "data" in parsed and "_alpaca_mcp_security" in parsed:
        return parsed["data"]
    return parsed


class AlpacaMcpClient:
    """Deterministic (non-LLM) MCP client session against alpaca-mcp-server.
    Use as `async with AlpacaMcpClient(config) as client: ...`.
    """

    def __init__(self, config: Config):
        self._config = config
        self._stdio_cm = None
        self._session_cm = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "AlpacaMcpClient":
        self._stdio_cm = stdio_client(alpaca_stdio_params(self._config))
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)

    async def list_tool_defs(self, names: list[str]) -> list[dict[str, Any]]:
        """Fetches the live tool schemas from the MCP server for the given
        tool names and translates them into OpenAI-style function-calling
        definitions for agent/brain.py. `mcp.types.Tool.input_schema` is
        already JSON Schema (note: snake_case attribute, not `inputSchema`
        -- the mcp package's Python bindings differ from the wire format's
        camelCase), so this is a near-direct passthrough into the
        `{"type": "function", "function": {...}}` shape Featherless expects.
        """
        assert self.session is not None, "use `async with AlpacaMcpClient(...)`"
        listed = await self.session.list_tools()
        by_name = {t.name: t for t in listed.tools}
        defs: list[dict[str, Any]] = []
        for name in names:
            tool = by_name.get(name)
            if tool is None:
                continue
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema or {"type": "object", "properties": {}},
                    },
                }
            )
        return defs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Calls the MCP tool and returns plain Python data (dict/list/str),
        with alpaca-mcp-server's `{"data": ...}` envelope already unwrapped --
        every typed wrapper method below, and every caller elsewhere in the
        codebase, assumes `.get()`/indexing on the returned value works
        directly against the real payload.
        """
        assert self.session is not None, "use `async with AlpacaMcpClient(...)`"
        result = await self.session.call_tool(name, arguments)
        if getattr(result, "is_error", False):
            raise RuntimeError(f"alpaca-mcp-server tool '{name}' failed: {result.content}")
        return _extract_result(result)

    async def get_account(self) -> dict:
        return await self.call_tool("get_account_info", {})

    async def get_positions(self) -> list[dict]:
        # data shape: {"result": [...]}
        data = await self.call_tool("get_all_positions", {})
        return (data or {}).get("result", []) if isinstance(data, dict) else []

    async def get_option_contracts(
        self, underlying: str, expiration_gte: str, expiration_lte: str,
        strike_gte: float | None = None, strike_lte: float | None = None, limit: int = 1000,
    ) -> dict[str, dict]:
        """Contract metadata (strike, expiration, type, open_interest) keyed
        by OCC symbol -- distinct from get_option_chain, which carries market
        data (quotes) but no strike/expiration/OI fields.

        A strike band is required in practice, not just an expiration window:
        liquid underlyings like SPY have *daily* expirations, so a 45-day
        window can hold thousands of contracts across all strikes -- more
        than the API `limit` -- and results get silently truncated to only
        the first couple of expiration dates (verified live: an unbounded
        SPY query returned only 2 distinct expirations despite spanning 49
        days). Restricting to strikes near the underlying, which is all a
        delta-targeted strategy would ever use anyway, keeps each request
        well under the limit.
        """
        params: dict[str, Any] = {
            "underlying_symbols": underlying,
            "expiration_date_gte": expiration_gte,
            "expiration_date_lte": expiration_lte,
            "limit": limit,
        }
        if strike_gte is not None:
            params["strike_price_gte"] = strike_gte
        if strike_lte is not None:
            params["strike_price_lte"] = strike_lte
        data = await self.call_tool("get_option_contracts", params)
        contracts = (data or {}).get("option_contracts", []) if isinstance(data, dict) else []
        return {c["symbol"]: c for c in contracts}

    async def get_option_chain(
        self, underlying: str, expiration_gte: str, expiration_lte: str,
        strike_gte: float | None = None, strike_lte: float | None = None, limit: int = 1000,
    ) -> dict[str, dict]:
        """Market-data snapshots (latestQuote/latestTrade/dailyBar) keyed by
        OCC symbol. No greeks/IV on the free indicative feed -- see
        data/option_quotes.py for how these get combined with
        get_option_contracts + Black-Scholes into a full OptionQuote. See
        get_option_contracts for why the strike band matters."""
        params: dict[str, Any] = {
            "underlying_symbol": underlying,
            "expiration_date_gte": expiration_gte,
            "expiration_date_lte": expiration_lte,
            "limit": limit,
        }
        if strike_gte is not None:
            params["strike_price_gte"] = strike_gte
        if strike_lte is not None:
            params["strike_price_lte"] = strike_lte
        data = await self.call_tool("get_option_chain", params)
        return (data or {}).get("snapshots", {}) if isinstance(data, dict) else {}

    async def get_option_snapshot(self, occ_symbols: list[str]) -> dict[str, dict]:
        data = await self.call_tool("get_option_snapshot", {"symbols": ",".join(occ_symbols)})
        return (data or {}).get("snapshots", {}) if isinstance(data, dict) else {}

    async def get_stock_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]:
        # data shape: {"bars": {"<SYMBOL>": [...]}, "next_page_token": ...}.
        # feed="iex" is required: the default feed is SIP, and a paper
        # account with no market-data subscription gets a 403
        # ("subscription does not permit querying recent SIP data") on any
        # request touching recent dates -- verified live. IEX is the free
        # tier feed and is what this whole codebase should assume throughout.
        data = await self.call_tool(
            "get_stock_bars",
            {"symbols": symbol, "timeframe": timeframe, "start": start, "end": end, "feed": "iex"},
        )
        return (data or {}).get("bars", {}).get(symbol, []) if isinstance(data, dict) else []

    async def get_news(self, symbols: list[str], limit: int = 10) -> list[dict]:
        data = await self.call_tool("get_news", {"symbols": ",".join(symbols), "limit": limit})
        return (data or {}).get("news", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

    async def get_clock(self) -> dict:
        return await self.call_tool("get_clock", {})

    async def get_calendar(self, start: str, end: str) -> list[dict]:
        return await self.call_tool("get_calendar", {"start": start, "end": end})

    async def place_option_order(self, legs: list[dict], qty: int, client_order_id: str) -> dict:
        # qty and each leg's ratio_qty must be strings per the tool's schema
        # (verified live); order_class is auto-inferred when legs are given,
        # so it's deliberately omitted rather than guessed.
        string_legs = [{**leg, "ratio_qty": str(leg.get("ratio_qty", 1))} for leg in legs]
        return await self.call_tool(
            "place_option_order",
            {"legs": string_legs, "qty": str(qty), "client_order_id": client_order_id},
        )

    async def place_stock_order(self, symbol: str, side: str, qty: int, client_order_id: str) -> dict:
        """Simple market stock order -- used only by the protective delta-hedge
        overlay (risk/hedge.py) to trade SPY shares back to delta-neutral. This
        is deterministic code, not an LLM-exposed tool."""
        return await self.call_tool(
            "place_stock_order",
            {"symbol": symbol, "side": side, "qty": str(qty), "type": "market",
             "time_in_force": "day", "client_order_id": client_order_id},
        )

    async def get_orders(self, status: str = "open") -> list[dict]:
        data = await self.call_tool("get_orders", {"status": status})
        return (data or {}).get("result", []) if isinstance(data, dict) else []

    async def cancel_order(self, order_id: str) -> dict:
        return await self.call_tool("cancel_order_by_id", {"order_id": order_id})
