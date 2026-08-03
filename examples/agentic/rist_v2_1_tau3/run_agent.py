"""Drive the frozen Tau3 gym environment through the AReno rollout proxy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

RUNTIME_KEY = "_rist_tau3_runtime_by_sample"
MAX_AGENT_TURNS = 20


def _parse_simulation_payload(value: Any, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Tau3 {context} produced invalid simulation evidence") from exc
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Tau3 {context} produced no simulation evidence")
    return payload


def _assistant_message(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in message.tool_calls or []
    ]
    result = {"role": "assistant", "content": message.content}
    if calls:
        result["tool_calls"] = calls
    return result


def response_to_action(response: Any) -> tuple[str, dict[str, Any]]:
    """Convert exactly one policy message to the upstream gym action string."""

    assistant = _assistant_message(response)
    calls = assistant.get("tool_calls") or []
    if len(calls) > 1:
        raise ValueError("Tau3 policy response may contain at most one tool call")
    if calls:
        function = calls[0]["function"]
        try:
            arguments = json.loads(function["arguments"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Tau3 policy tool arguments must be valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tau3 policy tool arguments must be an object")
        action = json.dumps(
            {"name": function["name"], "arguments": arguments},
            sort_keys=True,
        )
        return action, assistant
    content = assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Tau3 policy response requires text or one tool call")
    return content, assistant


def store_runtime_result(item: Any, result: dict[str, Any]) -> None:
    """Store one immutable sample result on the shared prompt record."""

    mapping = item.record.setdefault(RUNTIME_KEY, {})
    if not isinstance(mapping, dict):
        raise ValueError("Tau3 runtime metadata container is invalid")
    key = str(int(item.sample_index))
    if key in mapping:
        raise ValueError(f"duplicate Tau3 runtime result for sample {key}")
    mapping[key] = result


def _append_raw_event(item: Any, payload: dict[str, Any]) -> None:
    journal = os.environ.get("RIST_RAW_JOURNAL_PATH")
    if not journal:
        if os.environ.get("RIST_REQUIRE_EVIDENCE_JOURNALS") == "1":
            raise RuntimeError("RIST_RAW_JOURNAL_PATH is required by the frozen Tau3 run")
        return
    row = {
        "task_id": item.record["id"],
        "prompt_index": int(item.prompt_index),
        "sample_index": int(item.sample_index),
        **payload,
    }
    descriptor = os.open(Path(journal), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
    finally:
        os.close(descriptor)


async def run_agent(ctx, batch):
    """Run fresh Tau3 episodes without policy or user-simulator retry."""

    try:
        import httpx
        from openai import AsyncOpenAI
        from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
        from tau2.gym.gym_agent import AgentGymEnv
    except ImportError as exc:
        raise RuntimeError("Tau3 rollout requires the frozen Tau3 and AReno environments") from exc

    user_llm = os.environ.get("RIST_TAU3_USER_LLM")
    if not user_llm:
        raise RuntimeError("RIST_TAU3_USER_LLM must pin the authorized user simulator")
    items = list(batch.iter_samples())
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(len(items), ctx.max_running_prompts),
            max_keepalive_connections=max(len(items), ctx.max_running_prompts),
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
        if item.record.get("domain") != "airline":
            raise RuntimeError("X3 permits only the strict airline reward domain")
        episode_seed = ctx.request_seed(item.prompt_index, item.sample_index, "tau3_user")
        if episode_seed is None:
            raise RuntimeError("Tau3 rollout requires an explicit training seed")
        env = AgentGymEnv(
            domain=item.record["domain"],
            task_id=item.record["task_id"],
            max_steps=100,
            solo_mode=False,
            user_llm=user_llm,
            user_llm_args={
                "temperature": 0.0,
                "seed": int(episode_seed),
                "num_retries": 0,
            },
            all_messages_as_observation=False,
        )
        turns = []
        evidence = []
        try:
            observation, info = await asyncio.wait_for(
                asyncio.to_thread(env.reset, seed=int(episode_seed)), timeout=900.0
            )
            tools = [tool.openai_schema for tool in info["tools"]]
            messages = [
                {"role": "system", "content": str(info["policy"])},
                {"role": "user", "content": str(observation)},
            ]
            final_reward = 0.0
            terminated = False
            truncated = False
            invalid_reason = None
            for turn_index in range(MAX_AGENT_TURNS):
                request = {
                    "model": "policy",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": False,
                }
                request_seed = ctx.request_seed(
                    item.prompt_index, item.sample_index, turn_index
                )
                if request_seed is not None:
                    request["seed"] = request_seed
                response = await client.chat.completions.create(**request)
                turns.append(
                    AgentTrajectoryTurn(
                        item=item,
                        messages=messages,
                        response=response,
                        tools=tools,
                        tool_choice="auto",
                    )
                )
                response_event = {
                    "turn_index": turn_index,
                    "phase": "policy_response",
                    "raw_response": response.model_dump(mode="json"),
                }
                evidence.append(response_event)
                _append_raw_event(item, response_event)
                try:
                    action, assistant = response_to_action(response)
                except ValueError as exc:
                    invalid_reason = str(exc)
                    break
                observation, reward, terminated, truncated, step_info = (
                    await asyncio.wait_for(
                        asyncio.to_thread(env.step, action), timeout=900.0
                    )
                )
                if terminated:
                    simulation_payload = _parse_simulation_payload(
                        step_info.get("simulation_run"), "terminated episode"
                    )
                event = {
                    "turn_index": turn_index,
                    "phase": "environment_step",
                    "observation": observation,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "reward_info": step_info.get("reward_info"),
                }
                if terminated:
                    event["simulation_run"] = simulation_payload
                evidence.append(event)
                _append_raw_event(item, event)
                final_reward = float(reward)
                if terminated or truncated:
                    break
                messages = [*messages, assistant]
                calls = assistant.get("tool_calls") or []
                if calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": calls[0]["id"],
                            "name": calls[0]["function"]["name"],
                            "content": str(observation),
                        }
                    )
                else:
                    messages.append({"role": "user", "content": str(observation)})
            if not terminated:
                cleanup_action = json.dumps(
                    {"name": "done", "arguments": {}}, sort_keys=True
                )
                _, _, cleanup_terminated, _, cleanup_info = await asyncio.wait_for(
                    asyncio.to_thread(env.step, cleanup_action), timeout=900.0
                )
                cleanup_simulation = _parse_simulation_payload(
                    cleanup_info.get("simulation_run"), "cleanup"
                )
                cleanup_event = {
                    "turn_index": len(turns),
                    "phase": "cleanup_not_policy",
                    "terminated": bool(cleanup_terminated),
                    "invalid_reason": invalid_reason or "MAX_AGENT_TURNS",
                    "simulation_run": cleanup_simulation,
                }
                evidence.append(cleanup_event)
                _append_raw_event(item, cleanup_event)
                if not cleanup_terminated:
                    raise RuntimeError("Tau3 cleanup action did not terminate the episode")
                truncated = True
                final_reward = 0.0
            if final_reward not in {0.0, 1.0}:
                raise RuntimeError("Tau3 upstream evaluator returned non-binary reward")
            evidence_bytes = json.dumps(
                evidence, sort_keys=True, separators=(",", ":")
            ).encode()
            store_runtime_result(
                item,
                {
                    "domain": item.record["domain"],
                    "task_id": item.record["task_id"],
                    "reward": final_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated or not terminated),
                    "invalid_reason": invalid_reason,
                    "evaluator": "tau2.EvaluationType.ALL",
                    "runtime_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                    "user_simulator": user_llm,
                    "user_seed": int(episode_seed),
                    "policy_retry_count": 0,
                    "user_retry_count": 0,
                },
            )
            return turns
        finally:
            await asyncio.to_thread(env.close)

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for group in grouped for turn in group])
    finally:
        await client.close()
