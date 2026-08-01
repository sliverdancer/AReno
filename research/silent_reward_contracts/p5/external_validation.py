"""Generate source-pinned ARCA P5 external-validation artifacts on CPU."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

from research.silent_reward_contracts.evaluate_auditor import evaluate


PROTOCOL_ID = "ARCA-P5-EXTERNAL-v0.1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_function(path: Path, name: str, globals_: dict[str, Any]):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level {name} in {path}, found {len(matches)}")
    module = ast.Module(body=matches, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class _EvalOutput:
    def __init__(
        self,
        reward: float,
        is_correct: bool,
        signals: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.reward = reward
        self.is_correct = is_correct
        self.signals = signals or []
        self.metadata = metadata or {}


class _Signal:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value


def _probe_rllm(paths: list[Path]) -> dict[str, Any]:
    outputs = []
    for path in paths[:2]:
        coerce = _extract_function(
            path,
            "_coerce_eval_result",
            {"EvalOutput": _EvalOutput, "Signal": _Signal},
        )
        missing_false = coerce({"is_correct": False})
        missing_true = coerce({"is_correct": True})
        positive = coerce({"reward": 1.0, "is_correct": True})
        outputs.append(
            {
                "source_file": path.name,
                "missing_false_reward": float(missing_false.reward),
                "missing_true_reward": float(missing_true.reward),
                "missing_true_is_correct": bool(missing_true.is_correct),
                "missing_true_metadata": missing_true.metadata,
                "positive_reward": float(positive.reward),
            }
        )
    comparable = [
        {key: value for key, value in row.items() if key != "source_file"}
        for row in outputs
    ]
    if len(comparable) != 2 or comparable[0] != comparable[1]:
        raise AssertionError("rLLM coercion paths do not reproduce the same behavior")
    docs = paths[2].read_text(encoding="utf-8")
    required = (
        "def evaluate(metadata: dict, trajectory: dict) -> dict",
        "Returns are coerced: `float`, `bool`, `dict`",
    )
    if any(fragment not in docs for fragment in required):
        raise AssertionError("rLLM lightweight dict-return contract is not present in pinned docs")
    first = outputs[0]
    if not (
        first["missing_false_reward"] == 0.0
        and first["missing_true_reward"] == 0.0
        and first["missing_true_is_correct"] is True
        and first["missing_true_metadata"] == {}
        and first["positive_reward"] == 1.0
    ):
        raise AssertionError("rLLM silent default probe no longer matches the frozen observation")
    return {"duplicated_production_paths": outputs, "documented_dict_return": True}


class _SampleStatus:
    TRUNCATED = "truncated"


class _Sample:
    Status = _SampleStatus

    @staticmethod
    def from_dict(value: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**value)


class _RolloutFnEvalOutput:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class _Logger:
    def info(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _run_slime_forge(path: Path, reward: float | None) -> dict[str, Any]:
    torch_stub = SimpleNamespace(
        load=lambda _path, weights_only=False: {
            "samples": [
                {
                    "reward": reward,
                    "status": "completed",
                    "index": 0,
                }
            ]
        }
    )
    generate_rollout = _extract_function(
        path,
        "generate_rollout",
        {
            "_resolve_path": lambda *args, **kwargs: "fixture.pt",
            "torch": torch_stub,
            "Sample": _Sample,
            "RolloutFnEvalOutput": _RolloutFnEvalOutput,
            "logger": _Logger(),
        },
    )
    args = SimpleNamespace(eval_reward_key=None, reward_key=None)
    output = generate_rollout(args, 0, None, evaluation=True)
    result = output.data["forge_eval"]
    return {
        "rewards": result["rewards"],
        "truncated": result["truncated"],
        "has_reward_field_present": "has_reward" in result,
    }


def _probe_slime(path: Path) -> dict[str, Any]:
    missing = _run_slime_forge(path, None)
    positive = _run_slime_forge(path, 1.0)
    if missing["rewards"] != [0.0] or missing["truncated"] != [False]:
        raise AssertionError("slime missing-reward probe no longer matches the frozen observation")
    if missing["has_reward_field_present"] or positive["rewards"] != [1.0]:
        raise AssertionError("slime positive/provenance control failed")
    return {"missing_reward": missing, "positive_control": positive}


def _probe_negative_controls(paths: dict[str, list[Path]]) -> dict[str, Any]:
    required = {
        "Agent-R1": (
            'step_info = {"error": str(exc)}',
            '"reward_extra_info": reward_extra_info',
        ),
        "RAGEN": (
            'status = "timeout" if',
            'message_objects=[{"severity": "error", "data": str(exc)}]',
        ),
        "Agent-Lightning": (
            "Warning: Reward is None for rollout",
            '"has_reward": final_reward_raw is not None',
        ),
    }
    results = {}
    for framework, fragments in required.items():
        source = paths[framework][0].read_text(encoding="utf-8")
        missing = [fragment for fragment in fragments if fragment not in source]
        if missing:
            raise AssertionError(f"{framework} negative-control fragments missing: {missing}")
        results[framework] = {
            "structured_status_fragments_verified": len(fragments),
            "qualifying_natural_case": False,
        }
    return results


def _verify_sources(
    manifest: dict[str, Any],
    upstreams_root: Path,
    require_git: bool,
) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    paths: dict[str, list[Path]] = {}
    records = []
    for candidate in manifest["candidates"]:
        root = upstreams_root / candidate["local_dir"]
        if require_git:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if head != candidate["commit"]:
                raise ValueError(f"{candidate['name']} HEAD {head} != {candidate['commit']}")
        candidate_paths = []
        file_records = []
        for item in candidate["audit_files"]:
            path = root / item["path"]
            observed = _sha256(path)
            if observed != item["sha256"]:
                raise ValueError(f"source hash mismatch for {candidate['name']}:{item['path']}")
            candidate_paths.append(path)
            file_records.append({**item, "observed_sha256": observed})
        paths[candidate["name"]] = candidate_paths
        records.append(
            {
                "framework": candidate["name"],
                "repository": candidate["repository"],
                "commit": candidate["commit"],
                "files": file_records,
            }
        )
    return paths, records


def _cases(slime_probe: dict[str, Any], rllm_probe: dict[str, Any]) -> dict[str, Any]:
    rllm_first = rllm_probe["duplicated_production_paths"][0]
    cases = [
        {
            "case_id": "P5-SLIME-F6-MISSING-EVAL-REWARD",
            "framework": "slime",
            "workload": "forge-eval-replay",
            "seed": 0,
            "natural_case": True,
            "severity": "medium",
            "conclusion_flip": False,
            "expected_violations": ["ARCA-F6-STATUS"],
            "payload": {
                "execution_failures": [
                    {
                        "failure": "reward missing at replay boundary",
                        "numeric_reward": slime_probe["missing_reward"]["rewards"][0],
                        "structured_status": None,
                    }
                ]
            },
        },
        {
            "case_id": "P5-SLIME-CLEAN-PRESENT-EVAL-REWARD",
            "framework": "slime",
            "workload": "forge-eval-replay",
            "seed": 1,
            "natural_case": False,
            "severity": "control",
            "conclusion_flip": False,
            "expected_violations": [],
            "payload": {
                "execution_failures": [
                    {
                        "failure": "reward present",
                        "numeric_reward": slime_probe["positive_control"]["rewards"][0],
                        "structured_status": "reward_present",
                    }
                ]
            },
        },
        {
            "case_id": "P5-RLLM-F3-DICT-REWARD-DEFAULT",
            "framework": "rllm",
            "workload": "python-module-evaluator",
            "seed": 0,
            "natural_case": True,
            "severity": "high",
            "conclusion_flip": True,
            "expected_violations": ["ARCA-F3-INFORMATIVE"],
            "payload": {
                "reward_controls": [
                    rllm_first["missing_false_reward"],
                    rllm_first["missing_true_reward"],
                ]
            },
        },
        {
            "case_id": "P5-RLLM-CLEAN-EXPLICIT-DICT-REWARD",
            "framework": "rllm",
            "workload": "python-module-evaluator",
            "seed": 1,
            "natural_case": False,
            "severity": "control",
            "conclusion_flip": False,
            "expected_violations": [],
            "payload": {"reward_controls": [0.0, rllm_first["positive_reward"]]},
        },
        {
            "case_id": "P5-AGENTR1-CLEAN-STRUCTURED-ENV-ERROR",
            "framework": "Agent-R1",
            "workload": "webshop-agent-flow",
            "seed": 0,
            "natural_case": False,
            "severity": "control",
            "conclusion_flip": False,
            "expected_violations": [],
            "payload": {
                "execution_failures": [
                    {"numeric_reward": 0.0, "structured_status": "step_info.error"}
                ]
            },
        },
        {
            "case_id": "P5-RAGEN-CLEAN-STRUCTURED-TIMEOUT",
            "framework": "RAGEN",
            "workload": "lean-environment",
            "seed": 0,
            "natural_case": False,
            "severity": "control",
            "conclusion_flip": False,
            "expected_violations": [],
            "payload": {
                "execution_failures": [
                    {"numeric_reward": 0.0, "structured_status": "timeout"}
                ]
            },
        },
        {
            "case_id": "P5-AGENTLIGHTNING-CLEAN-MISSING-REWARD-PROVENANCE",
            "framework": "Agent-Lightning",
            "workload": "verl-daemon",
            "seed": 0,
            "natural_case": False,
            "severity": "control",
            "conclusion_flip": False,
            "expected_violations": [],
            "payload": {
                "execution_failures": [
                    {
                        "numeric_reward": 0.0,
                        "structured_status": "warning;has_reward=false",
                    }
                ]
            },
        },
    ]
    return {"schema_version": "arca.p5.cases.v1", "split": "external", "cases": cases}


def _inspection_records() -> list[dict[str, Any]]:
    return [
        {
            "framework": "slime",
            "inspection_status": "QUALIFYING_NATURAL_CASE",
            "natural_case_count": 1,
            "boundary": "official forge evaluation replay",
            "finding": "missing reward is exported as numeric 0.0 without a reward-present status",
        },
        {
            "framework": "Agent-R1",
            "inspection_status": "NO_QUALIFYING_CASE_IN_INSPECTED_BOUNDARY",
            "natural_case_count": 0,
            "boundary": "WebShop agent-flow environment step",
            "finding": "numeric fallback retains step_info.error; loud failures and structured fallbacks were excluded",
        },
        {
            "framework": "RAGEN",
            "inspection_status": "NO_QUALIFYING_CASE_IN_INSPECTED_BOUNDARY",
            "natural_case_count": 0,
            "boundary": "Lean environment timeout and server-error handling",
            "finding": "timeout/server_error status and diagnostics remain structured",
        },
        {
            "framework": "rllm",
            "inspection_status": "QUALIFYING_NATURAL_CASE",
            "natural_case_count": 1,
            "boundary": "documented lightweight Python evaluator return coercion",
            "finding": "dict without reward is accepted as 0.0 even when is_correct is true, with empty metadata",
        },
        {
            "framework": "Agent-Lightning",
            "inspection_status": "NO_QUALIFYING_CASE_IN_INSPECTED_BOUNDARY",
            "natural_case_count": 0,
            "boundary": "veRL daemon reward fill and validation metrics",
            "finding": "missing reward is warned and preserved as has_reward=false; explicit fallback is excluded",
        },
    ]


def build_artifacts(
    manifest_path: Path,
    upstreams_root: Path,
    output_dir: Path,
    *,
    require_git: bool = True,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["protocol_id"] != PROTOCOL_ID:
        raise ValueError("unexpected P5 protocol")
    paths, source_records = _verify_sources(manifest, upstreams_root, require_git)
    slime_probe = _probe_slime(paths["slime"][0])
    rllm_probe = _probe_rllm(paths["rllm"])
    negative_probes = _probe_negative_controls(paths)
    payload = _cases(slime_probe, rllm_probe)
    metrics, case_rows = evaluate(payload)
    inspections = _inspection_records()
    natural_cases = sum(row["natural_case_count"] for row in inspections)
    natural_frameworks = sum(row["natural_case_count"] > 0 for row in inspections)
    exact_matches = all(row["exact_match"] for row in case_rows)
    gate_pass = (
        natural_cases >= 2
        and natural_frameworks >= 1
        and metrics["clean_false_positive_rate"] <= 0.05
        and metrics["high_natural_recall"] == 1.0
        and exact_matches
    )
    summary = {
        "schema_version": "arca.p5.summary.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_framework_count": len(inspections),
        "new_natural_case_count": natural_cases,
        "new_natural_framework_count": natural_frameworks,
        "frozen_rule_changes": 0,
        "all_case_predictions_exact": exact_matches,
        "clean_false_positive_rate": metrics["clean_false_positive_rate"],
        "clean_false_positive_wilson_95": metrics["clean_false_positive_wilson_95"],
        "high_natural_recall": metrics["high_natural_recall"],
        "natural_conclusion_flips_detected": metrics["natural_conclusion_flips_detected"],
        "decision": (
            "PASS_P5_EXTERNAL_NATURAL_TO_PAPER"
            if gate_pass
            else "NARROW_P5_INSUFFICIENT_EXTERNAL_BREADTH"
        ),
        "gpu_executed": False,
        "external_disclosure_performed": False,
    }
    evidence = {
        "schema_version": "arca.p5.evidence.v1",
        "summary": summary,
        "sources": source_records,
        "probes": {
            "slime": slime_probe,
            "rllm": rllm_probe,
            "bounded_negative_controls": negative_probes,
        },
        "inspections": inspections,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_outputs = {
        "external_cases.json": payload,
        "external_metrics.json": metrics,
        "external_evidence.json": evidence,
    }
    for name, value in json_outputs.items():
        (output_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    with (output_dir / "external_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(case_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(case_rows)
    with (output_dir / "framework_inspections.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(inspections[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(inspections)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=root / "upstream_manifest.json")
    parser.add_argument("--upstreams-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts")
    args = parser.parse_args()
    evidence = build_artifacts(args.manifest, args.upstreams_root, args.output_dir)
    print(json.dumps(evidence["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
