from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = REPO_ROOT / "research" / "structured_action_supervision_v2"
sys.path.insert(0, str(V2_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


forensics = _load("sas_v2_forensics", V2_ROOT / "analyze_v1_1_failure.py")
protocol = _load("sas_v2_protocol", V2_ROOT / "prepare_protocol.py")
hook = _load("sas_v2_hook", V2_ROOT / "stage_completion_hook.py")
b2 = _load("sas_v2_b2_inference", V2_ROOT / "b2_inference.py")
b2_analysis = _load("sas_v2_b2_analysis", V2_ROOT / "analyze_b2.py")
b2_controller = _load("sas_v2_b2_controller", V2_ROOT / "run_b2_controller.py")


class SasV2ToolReadinessTests(unittest.TestCase):
    def test_v2_1_manifest_changes_capacity_only(self):
        successor = json.loads(
            (V2_ROOT / "stages" / "B2_1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        parent = json.loads(
            (V2_ROOT / "stages" / "B1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(successor["protocol_id"], "SAS-TR-v2.1")
        self.assertEqual(successor["runtime"]["max_running_prompts"], 1)
        self.assertEqual(successor["runtime"]["client_workers"], 1)
        self.assertEqual(successor["scientific_contract"]["cells"], parent["cells"])
        self.assertEqual(
            successor["scientific_contract"]["sampling_seeds"],
            parent["sampling_seeds"],
        )
        self.assertEqual(
            successor["scientific_contract"]["selection_order"],
            parent["selection_order"],
        )
        self.assertEqual(
            successor["scientific_contract"]["calibration_gates"],
            parent["calibration_gates"],
        )
        self.assertEqual(
            successor["scientific_contract"]["request_seed_salt"],
            "SAS-TR-v2.0",
        )
        self.assertFalse(successor["scientific_contract"]["reuse_v2_0_trajectories"])

    def test_v2_1_fail_fast_rejects_transport_loss_only(self):
        valid_trajectory = {
            "terminal_reason": "MISSING_TOOL_CALL",
            "turns": [{"raw_response": {"choices": []}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.json"
            payload = {
                "summary": {"raw_evidence_complete": True},
                "trajectories": [dict(valid_trajectory) for _ in range(16)],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")

            b2_controller._require_complete_client_evidence(path)

            payload["trajectories"][0]["terminal_reason"] = "REQUEST_ERROR"
            payload["summary"]["raw_evidence_complete"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fail-fast evidence gate"):
                b2_controller._require_complete_client_evidence(path)

    def test_v2_1_preflight_uses_no_frozen_row_and_preserves_response(self):
        response_body = json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "length",
                    }
                ]
            }
        ).encode()

        class FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preflight.json"
            with mock.patch.object(
                b2_controller.urllib.request,
                "urlopen",
                return_value=FakeResponse(response_body),
            ):
                b2_controller._preflight(
                    base_url="http://127.0.0.1:8765",
                    mode="disabled",
                    output=output,
                    deadline=time.time() + 60,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "PASS")
        self.assertFalse(payload["uses_frozen_task_row"])
        self.assertEqual(payload["raw_response"]["choices"][0]["finish_reason"], "length")

    def test_v2_1_serve_command_freezes_single_running_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(b2_controller.subprocess, "Popen") as popen:
                process, log_file = b2_controller._serve(
                    areno_bin=Path("/opt/areno"),
                    model_path=Path("/models/qwen"),
                    base_host="127.0.0.1",
                    port=8765,
                    disabled=True,
                    log_path=Path(tmp) / "serve.log",
                    max_running_prompts=1,
                )
                command = popen.call_args.args[0]
                log_file.close()

        self.assertIs(process, popen.return_value)
        index = command.index("--max-running-prompts")
        self.assertEqual(command[index + 1], "1")
        self.assertIn("--disable-thinking", command)

    def test_v2_1_signal_handler_is_fail_closed(self):
        with self.assertRaises(SystemExit):
            b2_controller._raise_controller_signal(15, None)

    def test_b2_request_seed_is_stable_and_factor_sensitive(self):
        first = b2.request_seed(3101, 0, 0)

        self.assertEqual(first, b2.request_seed(3101, 0, 0))
        self.assertNotEqual(first, b2.request_seed(3202, 0, 0))
        self.assertNotEqual(first, b2.request_seed(3101, 1, 0))
        self.assertNotEqual(first, b2.request_seed(3101, 0, 1))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_b2_validator_never_repairs_missing_or_malformed_calls(self):
        valid = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_catalog",
                                    "arguments": '{"categories":["jacket","bottle"]}',
                                },
                            }
                        ],
                    },
                }
            ]
        }

        accepted = b2.validate_response(valid, "search_catalog")
        missing = b2.validate_response(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "use search"},
                    }
                ]
            },
            "search_catalog",
        )

        self.assertTrue(accepted["valid"])
        self.assertEqual(accepted["arguments"]["categories"], ["jacket", "bottle"])
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["reason"], "MISSING_TOOL_CALL")
        self.assertIsNone(missing["arguments"])

    def test_b2_summary_requires_actual_raw_response_objects(self):
        base = {
            "first_turn_executable": True,
            "four_turn_complete": True,
            "strict_reward": 0.4,
            "fabricated_call_count": 0,
        }
        complete = [{**base, "turns": [{"raw_response": {"choices": []}}]}]
        missing = [{**base, "turns": [{"raw_response": None}]}]

        self.assertTrue(b2.summarize(complete)["raw_evidence_complete"])
        self.assertFalse(b2.summarize(missing)["raw_evidence_complete"])

    def test_b2_aggregate_enforces_frozen_cross_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for cell in ("D128", "D512", "N128", "N512"):
                for seed in (3101, 3202):
                    trajectories = []
                    for index in range(16):
                        positive = cell.startswith("N") and index == 0
                        trajectories.append(
                            {
                                "sampling_seed": seed,
                                "first_turn_executable": True,
                                "four_turn_complete": True,
                                "strict_reward": 0.4 if positive else 0.0,
                                "fabricated_call_count": 0,
                                "turns": [{"raw_response": {"choices": []}}],
                            }
                        )
                    payload = {
                        "protocol_id": "SAS-TR-v2.0",
                        "split": "calibration",
                        "cell_id": cell,
                        "trajectories": trajectories,
                    }
                    path = Path(tmp) / f"{cell}-{seed}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(path)

            result = b2_analysis.aggregate(paths, "calibration")

            self.assertEqual(result["selected_cell"], "N128")
            self.assertEqual(result["decision"], "PASS_CALIBRATION_SELECT_N128")
            self.assertTrue(result["cell_results"]["N128"]["passes_frozen_gates"])

    def test_consumed_v1_1_evidence_supports_registered_candidate(self):
        result = forensics.audit_evidence(forensics.DEFAULT_EVIDENCE_ROOT)

        self.assertEqual(
            result["decision"],
            "PASS_ROOT_CAUSE_CANDIDATE_THINK_BUDGET_EXHAUSTION",
        )
        self.assertEqual(result["counts"]["raw_samples"], 128)
        self.assertEqual(result["counts"]["budget_saturated"], 128)
        self.assertEqual(result["counts"]["closed_think"], 0)
        self.assertEqual(result["counts"]["parsed_tool_calls"], 0)

    def test_protocol_split_is_deterministic_and_disjoint(self):
        rows = [
            {
                "id": f"row-{index:02d}",
                "constraint_signature": f"signature-{index:02d}",
            }
            for index in range(48)
        ]

        first = protocol.split_rows(rows)
        second = protocol.split_rows(list(reversed(rows)))

        self.assertEqual(first, second)
        self.assertEqual({name: len(split) for name, split in first.items()}, {
            "calibration": 16,
            "validation": 16,
            "reserve": 16,
        })
        id_sets = [{row["id"] for row in split} for split in first.values()]
        self.assertTrue(id_sets[0].isdisjoint(id_sets[1]))
        self.assertTrue(id_sets[0].isdisjoint(id_sets[2]))
        self.assertTrue(id_sets[1].isdisjoint(id_sets[2]))

    def test_selection_prefers_smallest_passing_nonthinking_budget(self):
        passing = {
            "first_turn_executable_rate": 0.97,
            "four_turn_completion_rate": 0.80,
            "positive_reward_rate": 0.20,
            "fabricated_call_count": 0,
        }
        results = {
            "D128": dict(passing),
            "D512": dict(passing),
            "N128": dict(passing),
            "N512": dict(passing),
        }
        self.assertEqual(protocol.select_eligible_cell(results), "N128")

        results["N128"]["positive_reward_rate"] = 0.0
        self.assertEqual(protocol.select_eligible_cell(results), "N512")

        results["N512"]["four_turn_completion_rate"] = 0.50
        self.assertIsNone(protocol.select_eligible_cell(results))

    def test_prepare_writes_hashed_split_manifest_without_heldout(self):
        rows = [
            {
                "id": f"row-{index:02d}",
                "constraint_signature": f"signature-{index:02d}",
                "split": "train",
            }
            for index in range(48)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "train.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = Path(tmp) / "out"
            manifest = protocol.prepare(source, output)

            self.assertFalse(manifest["heldout_consumed"])
            self.assertEqual(manifest["selection_order"], ["N128", "N512"])
            for name in ("calibration", "validation", "reserve"):
                path = output / manifest["files"][name]["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(
                    protocol._sha256_bytes(path.read_bytes()),
                    manifest["files"][name]["sha256"],
                )

    def test_bridge_hook_cannot_upgrade_a_passed_cpu_stage(self):
        payload = json.loads(
            (V2_ROOT / "stages" / "B1" / "stage_result.json").read_text(
                encoding="utf-8"
            )
        )

        assessment = hook.assess(payload)

        self.assertEqual(assessment["decision"], "STAY_DIAGNOSTIC")
        self.assertIn(
            "tool_readiness_causally_validated",
            assessment["unmet_criteria"],
        )

    def test_v2_1_passed_interface_stays_diagnostic(self):
        payload = json.loads(
            (V2_ROOT / "stages" / "B2_1" / "stage_result.json").read_text(
                encoding="utf-8"
            )
        )

        assessment = hook.assess(payload)

        self.assertEqual(assessment["protocol_id"], "SAS-TR-v2.1")
        self.assertEqual(assessment["decision"], "STAY_DIAGNOSTIC")
        self.assertNotIn(
            "tool_readiness_causally_validated",
            assessment["unmet_criteria"],
        )
        self.assertIn("af_lf_contrast_instantiated", assessment["unmet_criteria"])


if __name__ == "__main__":
    unittest.main()
