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
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from alpaca_quant_agent.config import Config
from alpaca_quant_agent.strategy.screener import OptionQuote


def _alpaca_env(config: Config) -> dict[str, str]:
    return {
        "ALPACA_API_KEY": config.creds.alpaca_api_key,
        "ALPACA_SECRET_KEY": config.creds.alpaca_secret_key,
        "ALPACA_PAPER_TRADE": "true" if config.creds.alpaca_paper else "false",
    }


def alpaca_stdio_params(config: Config) -> StdioServerParameters:
    return StdioServerParameters(command="uvx", args=["alpaca-mcp-server"], env=_alpaca_env(config))


# Read-only tool names the reasoning model (agent/brain.py) is allowed to
# call directly during its turn. Order-placement tools are deliberately
# excluded -- see module docstring.
LLM_DATA_TOOL_NAMES = ["get_account", "get_positions", "get_news", "get_clock"]


def _extract_result(result: Any) -> Any:
    """Unwraps an mcp.types.CallToolResult into plain Python data. Modern MCP
    tool results may carry `structured_content` (already parsed JSON-like
    data); otherwise we fall back to parsing the first text content block
    (alpaca-mcp-server, like most MCP servers, returns JSON-serialized data
    as a text block).
    """
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
    return None


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
        definitions for agent/brain.py. MCP tool `inputSchema` is already
        JSON Schema, so this is a near-direct passthrough into the
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
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }
            )
        return defs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Calls the MCP tool and returns plain Python data (dict/list/str),
        not the raw CallToolResult -- every typed wrapper method below, and
        every caller elsewhere in the codebase, assumes `.get()`/indexing on
        the returned value works directly.
        """
        assert self.session is not None, "use `async with AlpacaMcpClient(...)`"
        result = await self.session.call_tool(name, arguments)
        if getattr(result, "is_error", False):
            raise RuntimeError(f"alpaca-mcp-server tool '{name}' failed: {result.content}")
        return _extract_result(result)

    async def get_account(self) -> dict:
        return await self.call_tool("get_account", {})

    async def get_positions(self) -> list[dict]:
        return await self.call_tool("get_positions", {})

    async def get_option_chain(self, underlying: str) -> list[dict]:
        return await self.call_tool("get_option_chain", {"underlying_symbol": underlying})

    async def get_option_snapshot(self, occ_symbol: str) -> dict:
        return await self.call_tool("get_option_snapshot", {"symbol_or_symbols": occ_symbol})

    async def get_stock_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]:
        return await self.call_tool(
            "get_stock_bars", {"symbol": symbol, "timeframe": timeframe, "start": start, "end": end}
        )

    async def get_news(self, symbols: list[str], limit: int = 10) -> list[dict]:
        return await self.call_tool("get_news", {"symbols": symbols, "limit": limit})

    async def get_clock(self) -> dict:
        return await self.call_tool("get_clock", {})

    async def get_calendar(self, start: str, end: str) -> list[dict]:
        return await self.call_tool("get_calendar", {"start": start, "end": end})

    async def place_option_order(self, legs: list[dict], qty: int, client_order_id: str) -> dict:
        return await self.call_tool(
            "place_option_order",
            {"legs": legs, "qty": qty, "order_class": "mleg" if len(legs) > 1 else "simple",
             "client_order_id": client_order_id},
        )

    async def place_stock_order(self, symbol: str, side: str, qty: int, client_order_id: str) -> dict:
        """Simple market stock order -- used only by the protective delta-hedge
        overlay (risk/hedge.py) to trade SPY shares back to delta-neutral. This
        is deterministic code, not an LLM-exposed tool."""
        return await self.call_tool(
            "place_stock_order",
            {"symbol": symbol, "side": side, "qty": qty, "order_type": "market",
             "time_in_force": "day", "client_order_id": client_order_id},
        )

    async def get_orders(self, status: str = "open") -> list[dict]:
        return await self.call_tool("get_orders", {"status": status})

    async def cancel_order(self, order_id: str) -> dict:
        return await self.call_tool("cancel_order", {"order_id": order_id})


def parse_option_chain(raw_chain: list[dict], underlying: str) -> list[OptionQuote]:
    """Converts the alpaca-mcp-server get_option_chain response into our
    internal OptionQuote type used throughout strategy/screener.py.
    Kept in one place so a server-side response-shape change only touches
    this function.
    """
    quotes: list[OptionQuote] = []
    for row in raw_chain:
        exp = row["expiration_date"]
        if isinstance(exp, str):
            exp = datetime.strptime(exp, "%Y-%m-%d").date()
        greeks = row.get("greeks") or {}
        quotes.append(
            OptionQuote(
                occ_symbol=row["symbol"],
                underlying=underlying,
                strike=float(row["strike_price"]),
                expiration=exp,
                option_type="call" if str(row["type"]).lower().startswith("c") else "put",
                bid=float(row.get("bid_price") or 0.0),
                ask=float(row.get("ask_price") or 0.0),
                delta=float(greeks.get("delta") or 0.0),
                vega=float(greeks.get("vega") or 0.0),
                open_interest=int(row.get("open_interest") or 0),
                iv=float(row.get("implied_volatility") or 0.0),
            )
        )
    return quotes
