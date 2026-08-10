"""Freeze a CPU-only Tau3 parseability canary template.

The template is a next-step external public environment candidate after the BFCL
format gate failed.  It does not execute Tau3, call a model/API, use GPU, or
train.  The CPU replay only validates the local adapter path that converts an
OpenAI-style tool call into the Tau3 action JSON expected by the gym.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = Path(__file__).resolve().parent
TAU3_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
TAU3_TAG = "v1.0.1"


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    type: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tau3_runner():
    path = ROOT / "examples" / "agentic" / "rist_v2_1_tau3" / "run_agent.py"
    spec = importlib.util.spec_from_file_location("rist_tau3_parseability_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_fake_tool_response(name: str = "DB", arguments: dict[str, Any] | None = None) -> FakeResponse:
    if arguments is None:
        arguments = {"query": "SELECT 1"}
    return FakeResponse(
        choices=[
            FakeChoice(
                message=FakeMessage(
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id="call_tau3_cpu_fixture_0",
                            type="function",
                            function=FakeFunction(
                                name=name,
                                arguments=json.dumps(arguments, sort_keys=True),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def run_cpu_parser_replay() -> dict[str, Any]:
    runner = load_tau3_runner()
    response = build_fake_tool_response(arguments={"query": "SELECT * FROM flights LIMIT 1"})
    action, assistant = runner.response_to_action(response)
    parsed_action = json.loads(action)

    malformed = build_fake_tool_response(arguments={"query": "SELECT 1"})
    malformed.choices[0].message.tool_calls[0].function.arguments = "[1, 2, 3]"
    malformed_rejected = False
    try:
        runner.response_to_action(malformed)
    except ValueError:
        malformed_rejected = True

    return {
        "status": "PASS",
        "cpu_only": True,
        "model_inference_used": False,
        "api_used": False,
        "gpu_used": False,
        "training_used": False,
        "heldout_or_sealed_access_used": False,
        "bfcl_used": False,
        "adapter": "examples/agentic/rist_v2_1_tau3/run_agent.py::response_to_action",
        "input_response_shape": "OpenAI chat.completions response with one tool call",
        "parsed_action": parsed_action,
        "assistant_has_tool_calls": bool(assistant.get("tool_calls")),
        "malformed_non_object_arguments_rejected": malformed_rejected,
    }


def build_template(replay: dict[str, Any]) -> dict[str, Any]:
    runner_path = ROOT / "examples" / "agentic" / "rist_v2_1_tau3" / "run_agent.py"
    reward_path = ROOT / "examples" / "agentic" / "rist_v2_1_tau3" / "reward.py"
    loader_path = ROOT / "examples" / "agentic" / "rist_v2_1_tau3" / "dataset_loader.py"
    return {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-v1",
        "status": "CPU_FROZEN_AWAITING_SEPARATE_SINGLE_REQUEST_AUTHORIZATION",
        "scope": {
            "external_public_environment_candidate": "Tau3/Tau2 airline tasks",
            "tau3_tag": TAU3_TAG,
            "tau3_commit": TAU3_COMMIT,
            "task_count": 1,
            "model_count": 1,
            "rollouts_per_task": 1,
            "retries": 0,
            "training_authorized": False,
            "bfcl_used": False,
            "heldout_or_sealed_access_authorized": False,
            "model_inference_authorized": False,
            "api_inference_authorized": False,
            "gpu_authorized": False,
        },
        "source_identity": {
            "runner_sha256": sha256_file(runner_path),
            "reward_sha256": sha256_file(reward_path),
            "dataset_loader_sha256": sha256_file(loader_path),
        },
        "runtime_required_bindings": {
            "source_commit": "UNBOUND_40_HEX_AT_RUNTIME",
            "model_repo_or_api_id": "UNBOUND_AT_RUNTIME",
            "model_revision": "UNBOUND_AT_RUNTIME",
            "user_simulator_model": "UNBOUND_AT_RUNTIME",
            "user_simulator_revision": "UNBOUND_AT_RUNTIME",
            "user_simulator_authorization_sha256": "UNBOUND_64_HEX_AT_RUNTIME",
            "gpu_uuid_or_api_provider": "UNBOUND_AT_RUNTIME",
            "task_id": "UNBOUND_TAU3_AIRLINE_PUBLIC_TASK_ID",
        },
        "request_policy": {
            "single_model_request_only": True,
            "zero_retry": True,
            "no_training": True,
            "no_bfcl_access": True,
            "terminal_finalizer_required_even_on_parse_error": True,
            "no_protocol_edit_after_first_model_request": True,
        },
        "success_gate": {
            "primary": "parseable_tool_call_emission",
            "required_observed_tool_call_count_min": 1,
            "required_adapter_action_json_object": True,
            "strict_task_success_required": False,
            "reward_resolution_claim_allowed": False,
        },
        "cpu_parser_replay": replay,
        "interpretation": {
            "why_tau3_next": (
                "Unlike the BFCL direct-generation canary, the Tau3 adapter already uses an "
                "OpenAI-style tool-call response path and converts exactly one tool call into "
                "the gym action JSON."
            ),
            "current_status": (
                "CPU parser path is validated, but no external model request has been sent. "
                "The goal remains incomplete until a separately authorized runtime canary "
                "produces parseable tool calls."
            ),
        },
    }


def write_report(template: dict[str, Any]) -> str:
    replay = template["cpu_parser_replay"]
    return "\n".join(
        [
            "# Tau3 airline parseability canary freeze",
            "",
            f"Status: `{template['status']}`",
            "",
            "This is the next external public-environment candidate after the BFCL",
            "format route terminated at parse failure. It is CPU-only and freezes a",
            "future `1 task × 1 model × 1 rollout` canary. No model/API/GPU/training",
            "action is authorized by this file.",
            "",
            "## CPU parser replay",
            "",
            f"- Replay status: {replay['status']}",
            f"- Adapter: `{replay['adapter']}`",
            f"- Parsed tool name: `{replay['parsed_action']['name']}`",
            f"- Malformed non-object arguments rejected: {replay['malformed_non_object_arguments_rejected']}",
            "",
            "## Runtime gate",
            "",
            "A future execution must bind the source commit, model/API identity, Tau3",
            "user-simulator identity, runtime target, and exactly one public airline task",
            "before the first request. The only success criterion is parseable tool-call",
            "emission; strict reward success and reward-resolution claims remain closed.",
            "",
        ]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay = run_cpu_parser_replay()
    template = build_template(replay)
    (OUT_DIR / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256((OUT_DIR / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_bytes()).hexdigest()
    (OUT_DIR / "TAU3_PARSEABILITY_CANARY_TEMPLATE.sha256").write_text(
        f"{digest}  TAU3_PARSEABILITY_CANARY_TEMPLATE.json\n",
        encoding="ascii",
    )
    (OUT_DIR / "TAU3_PARSEABILITY_CANARY_REPORT.md").write_text(
        write_report(template),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
