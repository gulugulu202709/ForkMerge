"""Recompute the archived screw simulation from actual stored predictions; no training."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/screw_repair"


def check_payload(experiment):
    manifest = json.loads((experiment / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (experiment / name).resolve()
        if not path.is_relative_to(experiment) or ".." in Path(name).parts:
            raise ValueError("manifest path escapes experiment")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("checksum mismatch: " + name)
    predictions = set((experiment / "artifacts/predictions").rglob("*.npz"))
    recorded = {experiment / p for p in manifest["files"] if p.startswith("artifacts/predictions/")}
    if predictions != recorded or len(predictions) != 90:
        raise ValueError("expected exactly 90 checksum-verified prediction files")
    return manifest


def compare_values(actual, expected, name="summary"):
    """Compare every recorded cell/contrast, not just the headline decision."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [name + ": dictionary keys differ"]
        return [error for key in expected for error in compare_values(actual[key], expected[key], name + "." + key)]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [name + ": list length differs"]
        return [error for i, (a, b) in enumerate(zip(actual, expected))
                for error in compare_values(a, b, name + "[" + str(i) + "]")]
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return [] if isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10) else [name + ": numeric value differs"]
    return [] if type(actual) is type(expected) and actual == expected else [name + ": value differs"]


def reproduce(experiment, output):
    experiment, output = experiment.resolve(), output.resolve()
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    if output.is_relative_to(experiment) or experiment.is_relative_to(output):
        raise ValueError("output must be outside the immutable experiment archive")
    manifest = check_payload(experiment)
    output.mkdir(parents=True)
    for folder in ("configs", "data", "evidence", "predictions"):
        shutil.copytree(experiment / "artifacts" / folder, output / folder)
    os.environ["FORKMERGE_SCREW_RUN"] = str(output)
    sys.path.insert(0, str(experiment / "code"))
    importlib.import_module("analyze").analyze()
    summary = json.loads((output / "analysis/SUMMARY.json").read_text(encoding="utf-8"))
    expected = json.loads((experiment / "expected/SUMMARY.json").read_text(encoding="utf-8"))
    # Anonymizing protocol/checkpoint paths changes those two file hashes only.
    keys = set(expected) - {"checkpoint_freeze_sha256", "protocol_sha256"}
    mismatches = compare_values({k: summary[k] for k in keys}, {k: expected[k] for k in keys})
    result = {
        "status": "VERIFIED_ARCHIVED_SIMULATION_REPLAY" if not mismatches else "REPLAY_MISMATCH",
        "source_experiment": manifest["source_experiment"],
        "prediction_files": 90, "cells": len(summary["cells"]), "decision": summary["decision"],
        "mismatches": mismatches, "numeric_tolerance": {"relative": 1e-9, "absolute": 1e-10},
        "new_training": False, "new_model_inference": False, "gpu_device_seconds": 0,
        "real_world_video_evidence": False,
        "scope": "Archived instrumented simulation; private simulator states are evaluation-only, never visual model inputs.",
    }
    (output / "VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise ValueError("archived replay differs from recorded results: " + "; ".join(mismatches[:5]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/screw-replay")
    args = parser.parse_args()
    try:
        print(json.dumps(reproduce(args.experiment, args.output), indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, "Replay refused: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
