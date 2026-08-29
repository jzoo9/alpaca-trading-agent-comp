"""Runs one reasoning cycle over a set of already-screened, already
risk-gated candidate trades, using a Featherless AI-hosted open-weight
model (OpenAI-compatible chat completions + tool calling) as the reasoning
layer. See agent/tools.py for how the model's write surface is bounded, and
agent/prompts.py for its exact mandate.

This is a small hand-rolled agentic loop rather than a provider SDK, since
Featherless hosts arbitrary open-weight models via a plain OpenAI-compatible
API -- there's no equivalent of a first-party "agent SDK" to lean on here.
Tool schemas for the small read-only Alpaca data-tool subset
(execution/alpaca_mcp.py::LLM_DATA_TOOL_NAMES) are pulled live from the MCP
server's own tool definitions (already JSON Schema) and combined with the
locally-defined quant tools from agent/tools.py.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from alpaca_quant_agent.agent.prompts import SYSTEM_PROMPT
from alpaca_quant_agent.agent.tools import CycleState, build_quant_tools
from alpaca_quant_agent.config import Config
from alpaca_quant_agent.execution.alpaca_mcp import LLM_DATA_TOOL_NAMES

logger = logging.getLogger(__name__)

FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

CYCLE_PROMPT = (
    "Begin this cycle's trade review. Call list_candidates and get_portfolio_state "
    "first, then check news for any candidate underlyings before deciding."
)

MAX_TOOL_ROUNDS = 12


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    """Builds a minimal, explicit assistant-message dict to append back into
    the conversation, rather than trusting the SDK's pydantic model_dump()
    to produce a shape every OpenAI-compatible provider accepts unmodified.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return msg


async def run_cycle(config: Config, state: CycleState) -> str:
    """Drives one bounded tool-calling turn over `state.candidates`. Returns
    the final text summary the model produced (also logged by the caller).
    All actual state mutation (orders submitted, decisions logged) happens
    as a side effect of the tool calls in agent/tools.py, not from parsing
    this return value -- it's purely for the run log / write-up.
    """
    quant_tools = build_quant_tools(state)
    alpaca_tool_defs = await state.alpaca.list_tool_defs(LLM_DATA_TOOL_NAMES)
    alpaca_tool_names = {d["function"]["name"] for d in alpaca_tool_defs}
    tool_schemas = [t.schema for t in quant_tools.values()] + alpaca_tool_defs

    client = AsyncOpenAI(base_url=FEATHERLESS_BASE_URL, api_key=config.creds.featherless_api_key)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CYCLE_PROMPT},
    ]

    final_text = ""
    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.chat.completions.create(
            model=config.creds.featherless_model,
            messages=messages,
            tools=tool_schemas,
        )
        message = response.choices[0].message
        messages.append(_assistant_message_dict(message))

        if not message.tool_calls:
            final_text = message.content or ""
            break

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name in quant_tools:
                result_text = await quant_tools[name].handler(args)
            elif name in alpaca_tool_names:
                try:
                    result = await state.alpaca.call_tool(name, args)
                    result_text = result if isinstance(result, str) else json.dumps(result)
                except Exception as exc:  # noqa: BLE001 -- surface any tool failure back to the model, don't crash the cycle
                    result_text = json.dumps({"error": str(exc)})
            else:
                result_text = json.dumps({"error": f"unknown tool {name}"})

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
    else:
        logger.warning("cycle hit max tool-call rounds (%s) without a final answer", MAX_TOOL_ROUNDS)

    return final_text.strip()
