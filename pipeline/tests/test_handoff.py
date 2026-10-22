"""Research handoff: real archived predictions and explicitly synthetic CPU smoke."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def command(script, *args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return subprocess.run([sys.executable, str(ROOT / "tools" / script), *map(str, args)],
                          cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)


class HandoffTests(unittest.TestCase):
    def test_replay_recomputes_all_archived_cells_and_original_stop_decision(self):
        with tempfile.TemporaryDirectory(prefix="HANDOFF_REPLAY_") as tmp:
            output = Path(tmp) / "replay"
            run = command("reproduce_screw.py", "--output", output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((output / "VERIFICATION.json").read_text())
            self.assertEqual(result["status"], "VERIFIED_ARCHIVED_SIMULATION_REPLAY")
            self.assertEqual(result["prediction_files"], 90)
            self.assertEqual(result["cells"], 90)
            self.assertEqual(result["decision"], "STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT")
            self.assertEqual(result["mismatches"], [])
            self.assertFalse(result["new_training"])
            self.assertFalse(result["real_world_video_evidence"])

    def test_replay_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="HANDOFF_OUTPUT_") as tmp:
            output = Path(tmp) / "replay"
            output.mkdir()
            (output / "keep.txt").write_text("preserve")
            run = command("reproduce_screw.py", "--output", output)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("output already exists", run.stderr)
            self.assertEqual((output / "keep.txt").read_text(), "preserve")

    def test_replay_rejects_corrupted_prediction_before_writing_results(self):
        with tempfile.TemporaryDirectory(prefix="HANDOFF_CORRUPT_") as tmp:
            experiment = Path(tmp) / "experiment"
            shutil.copytree(ROOT / "experiments/screw_repair", experiment)
            prediction = next((experiment / "artifacts/predictions").rglob("*.npz"))
            prediction.write_bytes(prediction.read_bytes() + b"corrupted")
            output = Path(tmp) / "replay"
            run = command("reproduce_screw.py", "--experiment", experiment, "--output", output)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("checksum mismatch", run.stderr)
            self.assertFalse(output.exists())

    def test_smoke_saves_real_cpu_tensors_and_never_claims_video_results(self):
        with tempfile.TemporaryDirectory(prefix="HANDOFF_SMOKE_") as tmp:
            output = Path(tmp) / "smoke"
            run = command("smoke_cpu.py", "--output", output)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((output / "SMOKE.json").read_text())
            self.assertEqual(result["status"], "SYNTHETIC_CPU_INTERFACE_CHECK")
            self.assertFalse(result["method_efficacy_result"])
            self.assertFalse(result["video_generated"])
            self.assertFalse(result["cuda_initialized"])
            self.assertEqual(result["memory_shape"], [3, 64, 1536])
            self.assertEqual(result["arms"]["paired"]["initial_sha256"],
                             result["arms"]["paired_worst_witness"]["initial_sha256"])
            for arm in ("paired", "paired_worst_witness"):
                self.assertGreater(result["arms"][arm]["parameter_delta_l2"], 0)
                self.assertTrue((output / (arm + "_memory.npz")).is_file())

    def test_smoke_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="HANDOFF_SMOKE_EXISTS_") as tmp:
            run = command("smoke_cpu.py", "--output", tmp)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("output already exists", run.stderr)

    def test_unsupported_native_platform_refuses_before_device_or_budget_access(self):
        try:
            guard = importlib.import_module("pipeline.guard")
        except ImportError as exc:
            self.fail("CPU inspection must not require Linux-only lock imports: " + str(exc))
        with tempfile.TemporaryDirectory(prefix="HANDOFF_PLATFORM_") as tmp:
            with patch.object(guard, "validate_inputs", return_value={"allowed": True}), \
                 patch.object(guard.sys, "platform", "win32"), \
                 patch.object(guard.subprocess, "check_output", side_effect=AssertionError("no device query")):
                result = guard.run_phase("diagnostics", Path(tmp))
            self.assertEqual(result["reason"], "NATIVE_LAUNCH_REQUIRES_LINUX")
            self.assertEqual(result["gpu_device_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
