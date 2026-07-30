"""Two-turn Tic-Tac-Toe agent for the issue #199 GPU ablation.

The first turn chooses a square with the existing Tic-Tac-Toe tool. The
environment executes that call and returns a deterministic tool result. The
second turn produces a plain-text final answer. This makes every successful
trajectory valid for all three trainable-turn modes without synthesizing
missing tool calls or rewriting model output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

TICTACTOE_DIR = Path(__file__).resolve().parents[1] / "tictactoe"
sys.path.insert(0, str(TICTACTOE_DIR))
import game  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a careful Tic-Tac-Toe player. You play X. "
    "Choose exactly one legal empty square by calling the choose_square tool. "
    "Digits on the board are empty square labels, not marks. "
    "Win immediately if possible; otherwise block any immediate O win."
)
FINAL_ANSWER_PROMPT = (
    "State the square you chose and briefly explain the move. "
    "Do not call another tool."
)
CHOOSE_SQUARE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_square",
        "description": "Choose the next Tic-Tac-Toe square for X.",
        "parameters": {
            "type": "object",
            "properties": {
                "square": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 9,
                    "description": "The square number to place X in.",
                }
            },
            "required": ["square"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run one tool-call turn and one final-answer turn for each board."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The trainable-turn ablation requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    logger.info(
        "Issue #199 ablation agent start requests=%d max_running_prompts=%d",
        len(items),
        ctx.max_running_prompts,
    )
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    async def run_one(item):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        tool_choice = {"type": "function", "function": {"name": "choose_square"}}
        choice_response = await client.chat.completions.create(
            model="policy",
            messages=messages,
            tools=[CHOOSE_SQUARE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        choice_turn = AgentTrajectoryTurn(
            item=item,
            messages=messages,
            response=choice_response,
            tools=[CHOOSE_SQUARE_TOOL],
            tool_choice=tool_choice,
        )

        choice_message = _assistant_message(choice_response)
        final_messages = [*messages, choice_message]
        final_messages.extend(_tool_result_messages(choice_message, item.record))
        final_messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
        final_response = await client.chat.completions.create(
            model="policy",
            messages=final_messages,
            stream=False,
        )
        final_turn = AgentTrajectoryTurn(
            item=item,
            messages=final_messages,
            response=final_response,
        )
        return [choice_turn, final_turn]

    try:
        grouped_turns = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for turns in grouped_turns for turn in turns])
    finally:
        await client.close()


def _assistant_message(response: Any) -> dict[str, Any]:
    """Preserve the exact assistant content and parsed tool calls."""

    message = response.choices[0].message
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _tool_result_messages(
    assistant_message: dict[str, Any],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute each returned tool call and preserve one result per call ID."""

    board = game.normalize_board(record["board"])
    results = []
    for call in assistant_message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name")
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        if name != "choose_square":
            payload = {"ok": False, "error": f"unknown tool: {name}"}
        elif not isinstance(arguments, dict):
            payload = {"ok": False, "error": "invalid JSON arguments"}
        else:
            try:
                square = int(arguments.get("square"))
            except (TypeError, ValueError):
                square = None
            reward = game.score_move(board, square)
            payload = {
                "ok": reward >= 0.0,
                "square": square,
                "reward": reward,
            }
        results.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": str(name),
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            }
        )
    return results
