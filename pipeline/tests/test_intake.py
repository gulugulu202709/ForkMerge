"""CPU intake checks. All media here are temporary SELFTEST synthetic fixtures."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    intake = importlib.import_module("pipeline.intake")
except ModuleNotFoundError:
    intake = None

try:
    import cv2
    import numpy as np
    HAS_DECODER = hasattr(cv2, "CAP_PROP_N_THREADS")
except ImportError:
    cv2 = None
    np = None
    HAS_DECODER = False


def make_manifest(incoming, splits=("calibration",), create_files=True):
    """Files are placeholders for schema tests; decode tests replace with SELFTEST MP4."""
    clips, units = [], []
    resets = {"cup": "replace_cup", "teapot": "wash_refill_pot", "candy": "empty_refill_box"}
    for split in splits:
        for scenario in ("cup", "teapot", "candy"):
            unit_id = f"SELFTEST_{split}_{scenario}_unit"
            ids = []
            for index in range(3):
                clip_id = f"SELFTEST_{split}_{scenario}_{index}"
                ids.append(clip_id)
                clip = {
                    "id": clip_id, "file": clip_id + ".mp4",
                    "recording_id": clip_id + "_recording", "unit_id": unit_id,
                    "split": split, "scenario": scenario, "history_end": 1.0,
                    "stage_times": [0.25, 0.5, 0.75, 1.0],
                    "actions": ["observe", "interact", resets[scenario], "occlude"],
                    "future_end": 3.0,
                    "windows": {"left": [1.5, 2.0], "right": [2.5, 3.0]},
                    "rois": {"left": [0, 0, .5, 1], "right": [.5, 0, 1, 1]},
                    "readouts": {"left": {"kind": "count", "truth": 3},
                                 "right": {"kind": "manual", "truth": "SELFTEST_observed"}},
                    "verified": {"uncut": True, "history_actions_visible": True,
                                 "outcomes_visible": True, "actual_not_planned": True,
                                 "camera_stable": True, "layout_matched": True},
                    "source_note": "SELFTEST synthetic fixture; never real recordings or method results",
                }
                clips.append(clip)
                if create_files:
                    (incoming / clip["file"]).write_bytes(b"SELFTEST schema placeholder only")
            units.append({"id": unit_id, "split": split, "scenario": scenario, "clip_ids": ids})
    return {"version": 1, "dataset_id": "SELFTEST_dataset", "clips": clips, "units": units}


class IntakeAvailability(unittest.TestCase):
    def test_cpu_intake_api_exists(self):
        self.assertIsNotNone(intake, "pipeline.intake must implement the declared CPU intake API")


@unittest.skipIf(intake is None, "intake implementation not present yet")
class ManifestChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="SELFTEST_intake_schema_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.incoming = self.root / "incoming"
        self.incoming.mkdir()
        self.manifest = make_manifest(self.incoming)

    def errors(self, manifest=None):
        return intake.validate_manifest(self.manifest if manifest is None else manifest, self.incoming)

    def test_valid_manifest_is_not_mutated(self):
        before = copy.deepcopy(self.manifest)
        self.assertEqual(self.errors(), [])
        self.assertEqual(self.manifest, before)

    def test_calibration_units_are_optional_without_becoming_formal_units(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["units"] = []
        self.assertEqual(self.errors(manifest), [])
        for clip in manifest["clips"]:
            clip["unit_id"] = clip["id"] + "_ungrouped"
        self.assertEqual(self.errors(manifest), [])

    def test_temporal_future_leakage_order_and_bounds_are_rejected(self):
        cases = [
            ("stage_times", [0.25, .5, 1.01, 1.0]),
            ("stage_times", [0.5, .25, .75, 1.0]),
            ("stage_times", [.25, .5, .75, .99]),
            ("history_end", float("nan")),
            ("future_end", 1.0),
            ("windows", {"left": [.5, 2], "right": [2, 3]}),
            ("windows", {"left": [1, 2], "right": [2, 4]}),
            ("rois", {"left": [0, 0, 0, 1], "right": [.5, 0, 1, 1]}),
            ("actions", ["observe", "interact", "red", "occlude"]),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest["clips"][0][field] = value
                self.assertTrue(self.errors(manifest))

    def test_duplicate_recordings_paths_ids_and_split_leakage_are_rejected(self):
        for field in ("recording_id", "file", "id"):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["clips"][1][field] = manifest["clips"][0][field]
                self.assertTrue(self.errors(manifest))
        formal = make_manifest(self.incoming, ("train", "dev", "test"))
        formal["clips"][9]["unit_id"] = formal["clips"][0]["unit_id"]
        self.assertTrue(self.errors(formal))

    def test_unit_exact_membership_and_declarations_are_checked(self):
        for mutation in ("duplicate_unit", "two_members", "duplicate_member", "missing_unit", "scenario"):
            with self.subTest(mutation=mutation):
                manifest = make_manifest(self.incoming, ("train",))
                if mutation == "duplicate_unit":
                    manifest["units"].append(copy.deepcopy(manifest["units"][0]))
                elif mutation == "two_members":
                    manifest["units"][0]["clip_ids"].pop()
                elif mutation == "duplicate_member":
                    manifest["units"][0]["clip_ids"][1] = manifest["units"][0]["clip_ids"][0]
                elif mutation == "missing_unit":
                    manifest["units"].pop()
                else:
                    manifest["units"][0]["scenario"] = "teapot"
                self.assertTrue(self.errors(manifest))

    def test_planned_unknown_and_unverified_truth_are_rejected(self):
        for mutation in ("verification", "missing_truth", "planned_truth", "prospective", "wrong_count"):
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(self.manifest)
                clip = manifest["clips"][0]
                if mutation == "verification":
                    clip["verified"]["actual_not_planned"] = False
                elif mutation == "missing_truth":
                    del clip["readouts"]["left"]["truth"]
                elif mutation == "planned_truth":
                    clip["readouts"]["right"]["truth"] = "planned"
                elif mutation == "prospective":
                    manifest["status"] = "PROSPECTIVE_RECORDING_TEMPLATE_NOT_DATA_NOT_RESULTS"
                else:
                    clip["readouts"]["left"]["truth"] = True
                self.assertTrue(self.errors(manifest))

    def test_camera_and_common_layout_require_actual_human_verification(self):
        for field in ("camera_stable", "layout_matched"):
            for value in (None, False, "true", 1):
                with self.subTest(field=field, value=value):
                    manifest = copy.deepcopy(self.manifest)
                    if value is None:
                        del manifest["clips"][0]["verified"][field]
                    else:
                        manifest["clips"][0]["verified"][field] = value
                    self.assertTrue(self.errors(manifest))

    def test_sources_must_stay_inside_incoming(self):
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"SELFTEST outside fixture")
        for path in ("../outside.mp4", "..\\outside.mp4", str(outside), "C:\\outside.mp4"):
            with self.subTest(path=path):
                manifest = copy.deepcopy(self.manifest)
                manifest["clips"][0]["file"] = path
                self.assertTrue(self.errors(manifest))
        manifest = copy.deepcopy(self.manifest)
        manifest["clips"][1]["file"] = "./" + manifest["clips"][0]["file"]
        self.assertTrue(self.errors(manifest))

    def test_source_symlinks_cannot_escape_or_alias(self):
        source = self.incoming / self.manifest["clips"][0]["file"]
        alias = self.incoming / "SELFTEST_alias.mp4"
        try:
            alias.symlink_to(source)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable on this host")
        self.manifest["clips"][1]["file"] = alias.name
        self.assertTrue(self.errors())
        alias.unlink()
        outside = self.root / "SELFTEST_outside.mp4"
        outside.write_bytes(b"SELFTEST outside")
        alias.symlink_to(outside)
        self.assertTrue(self.errors())

    def test_malformed_input_returns_errors_instead_of_crashing(self):
        for value in (None, [], {}, {"version": 1, "dataset_id": "x", "clips": [None], "units": [None]}):
            with self.subTest(value=value):
                self.assertTrue(self.errors(value)) if value is not None else self.assertTrue(
                    intake.validate_manifest(None, self.incoming))
        for field in ("id", "scenario", "split", "unit_id", "recording_id", "stage_times", "actions"):
            manifest = copy.deepcopy(self.manifest)
            manifest["clips"][0][field] = {"SELFTEST": []}
            with self.subTest(field=field):
                self.assertTrue(self.errors(manifest))

    def test_missing_manifest_and_source_leave_no_ready_marker(self):
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "WAITING_RECORDINGS")
        self.assertFalse((self.root / "pipeline_data" / "SOURCE_READY.json").exists())
        (self.incoming / self.manifest["clips"][0]["file"]).unlink()
        (self.incoming / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.assertTrue(self.errors())
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "WAITING_RECORDINGS")
        self.assertFalse((self.root / "pipeline_data").exists())
        self.assertTrue((self.root / "pipeline_state" / "intake.json").exists())
        self.assertFalse((self.root / "pipeline_results").exists())

    def test_missing_calibration_groups_wait_without_decoding(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["clips"] = manifest["clips"][:3]
        manifest["units"] = manifest["units"][:1]
        (self.incoming / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "WAITING_RECORDINGS")
        self.assertFalse(result["calibration_ready"])
        self.assertFalse(result["formal_ready"])
        self.assertFalse((self.root / "pipeline_data").exists())

    def test_invalid_manifest_cannot_publish_ready_marker(self):
        self.manifest["clips"][0]["verified"]["uncut"] = False
        path = self.incoming / "manifest.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        result = intake.prepare(self.root, path)
        self.assertEqual(result["status"], "SOURCE_REJECTED")
        self.assertFalse((self.root / "pipeline_data").exists())


@unittest.skipUnless(intake is not None and HAS_DECODER, "SELFTEST MP4 checks require OpenCV with FFmpeg thread option and NumPy")
class VideoPreparationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="SELFTEST_intake_video_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.incoming = self.root / "incoming"
        self.incoming.mkdir()
        self.manifest = make_manifest(self.incoming, create_files=False)
        cv2.setNumThreads(2)
        self.write_videos(self.manifest["clips"])
        self.path = self.incoming / "manifest.json"
        self.write_manifest()

    def write_videos(self, clips, offset=0):
        for video_index, clip in enumerate(clips, start=offset):
            writer = cv2.VideoWriter(str(self.incoming / clip["file"]), cv2.VideoWriter_fourcc(*"mp4v"), 8, (96, 64))
            self.assertTrue(writer.isOpened(), "SELFTEST fixture MP4 writer unavailable")
            for index in range(25):
                # Red grows with time: post-boundary leakage is visibly testable.
                frame = np.full((64, 96, 3), (20, 30 + video_index * 3, index * 8), dtype=np.uint8)
                cv2.putText(frame, "SELFTEST", (2, 15), cv2.FONT_HERSHEY_PLAIN, .6, (255, 255, 255), 1)
                cv2.putText(frame, str(video_index), (2, 55), cv2.FONT_HERSHEY_PLAIN, .6, (255, 255, 255), 1)
                writer.write(frame)
            writer.release()

    def write_manifest(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_frozen_shapes_history_causality_phase_windows_and_read_only_reuse(self):
        raw_before = {p.name: p.read_bytes() for p in self.incoming.iterdir()}
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "SOURCE_READY", result)
        self.assertTrue(result["calibration_ready"])
        self.assertFalse(result["formal_ready"], "nine calibration clips never become a formal dataset")
        frozen = json.loads((self.root / "pipeline_data" / "manifest.json").read_text())
        clip = frozen["clips"][0]
        self.assertEqual(clip["window_fractions"], {"left": [.25, .5], "right": [.75, 1.0]})
        artifact = self.root / "pipeline_data" / "clips" / (clip["id"] + ".npz")
        with np.load(artifact, allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {"history_rgb", "current_rgb", "future_rgb"})
            self.assertEqual(arrays["history_rgb"].shape, (4, 5, 256, 448, 3))
            self.assertEqual(arrays["current_rgb"].shape, (256, 448, 3))
            self.assertEqual(arrays["future_rgb"].shape, (21, 256, 448, 3))
            self.assertEqual(arrays["history_rgb"].dtype, np.uint8)
            np.testing.assert_array_equal(arrays["current_rgb"], arrays["history_rgb"][-1, -1])
            np.testing.assert_array_equal(arrays["future_rgb"][0], arrays["current_rgb"])
            self.assertLess(float(arrays["history_rgb"][..., 0].mean()), 75)
            self.assertGreater(float(arrays["future_rgb"][-1, ..., 0].mean()), 175)
        for times, stage in zip(clip["history_sample_times"], clip["stage_times"]):
            self.assertTrue(all(0 <= t <= stage for t in times))
        prepared_before = {str(p.relative_to(self.root)): (p.stat().st_mtime_ns, p.read_bytes())
                           for p in (self.root / "pipeline_data").rglob("*") if p.is_file()}
        self.assertTrue(intake.prepare(self.root)["reused"])
        self.assertEqual(prepared_before, {str(p.relative_to(self.root)): (p.stat().st_mtime_ns, p.read_bytes())
                                          for p in (self.root / "pipeline_data").rglob("*") if p.is_file()})
        self.assertEqual(raw_before, {p.name: p.read_bytes() for p in self.incoming.iterdir()})
        self.assertFalse((self.root / "pipeline_results").exists())

    def test_changed_manifest_or_raw_file_cannot_replace_frozen_data(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        frozen = (self.root / "pipeline_data" / "manifest.json").read_bytes()
        self.manifest["clips"][0]["readouts"]["left"]["truth"] = 4
        self.write_manifest()
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertEqual((self.root / "pipeline_data" / "manifest.json").read_bytes(), frozen)
        self.manifest["clips"][0]["readouts"]["left"]["truth"] = 3
        self.write_manifest()
        source = self.incoming / self.manifest["clips"][0]["file"]
        with source.open("ab") as stream:
            stream.write(b"SELFTEST changed source")
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertEqual((self.root / "pipeline_data" / "manifest.json").read_bytes(), frozen)

    def test_corrupt_video_or_out_of_range_time_does_not_freeze_dataset(self):
        source = self.incoming / self.manifest["clips"][0]["file"]
        good_bytes = source.read_bytes()
        source.write_bytes(b"SELFTEST deliberately corrupt MP4")
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertFalse((self.root / "pipeline_data").exists())
        source.write_bytes(good_bytes)
        self.manifest["clips"][0]["future_end"] = 30
        self.write_manifest()
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertFalse((self.root / "pipeline_data").exists())

    def test_modified_or_incomplete_frozen_artifacts_are_rejected(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        artifact = self.root / "pipeline_data" / "clips" / (self.manifest["clips"][0]["id"] + ".npz")
        artifact.write_bytes(b"SELFTEST deliberately corrupt prepared RGB artifact")
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")

    def test_frozen_unit_annotations_cannot_silently_change(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        path = self.root / "pipeline_data" / "manifest.json"
        prepared = json.loads(path.read_text())
        prepared["units"][0]["split"] = "test"
        path.write_text(json.dumps(prepared), encoding="utf-8")
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")

    def extend_with_formal(self):
        formal = make_manifest(self.incoming, ("train", "dev", "test"), create_files=False)
        self.write_videos(formal["clips"], offset=9)
        self.manifest["clips"].extend(formal["clips"])
        self.manifest["units"].extend(formal["units"])
        self.write_manifest()

    def test_calibration_first_one_way_formal_extension_preserves_original_corpus(self):
        original = copy.deepcopy(self.manifest)
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        calibration_manifest = self.root / "pipeline_data" / "calibration_manifest.json"
        self.assertEqual(json.loads(calibration_manifest.read_text()), original)
        original_files = {p.name: (p.stat().st_mtime_ns, p.read_bytes())
                          for p in (self.root / "pipeline_data" / "clips").iterdir()}
        self.extend_with_formal()
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "SOURCE_READY", result)
        self.assertTrue(result["formal_ready"])
        self.assertTrue(result["calibration_ready"])
        self.assertTrue(result["extended"])
        self.assertEqual(json.loads(calibration_manifest.read_text()), original)
        for name, original_value in original_files.items():
            path = self.root / "pipeline_data" / "clips" / name
            self.assertEqual((path.stat().st_mtime_ns, path.read_bytes()), original_value)
        self.assertTrue(intake.prepare(self.root)["reused"])
        self.manifest["clips"][-1]["readouts"]["left"]["truth"] = 4
        self.write_manifest()
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")

    def test_formal_extension_after_fit_is_rejected(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        self.extend_with_formal()
        (self.root / "pipeline_state" / "fit_started.json").write_text('{"SELFTEST":true}', encoding="utf-8")
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "SOURCE_REJECTED")
        self.assertEqual(len(list((self.root / "pipeline_data" / "clips").glob("*.npz"))), 9)

    def test_formal_extension_after_formal_freeze_sentinel_is_rejected(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        self.extend_with_formal()
        (self.root / "pipeline_state" / "formal_frozen.json").write_text('{"SELFTEST":true}', encoding="utf-8")
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertEqual(len(list((self.root / "pipeline_data" / "clips").glob("*.npz"))), 9)

    def test_formal_extension_cannot_alter_existing_calibration_records(self):
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_READY")
        self.extend_with_formal()
        self.manifest["clips"][0]["readouts"]["left"]["truth"] = 4
        self.write_manifest()
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertEqual(len(list((self.root / "pipeline_data" / "clips").glob("*.npz"))), 9)

    def test_renamed_identical_raw_recording_is_not_independent_data(self):
        first = self.incoming / self.manifest["clips"][0]["file"]
        second = self.incoming / self.manifest["clips"][1]["file"]
        second.write_bytes(first.read_bytes())
        self.assertEqual(intake.prepare(self.root)["status"], "SOURCE_REJECTED")
        self.assertFalse((self.root / "pipeline_data").exists())

    def test_subframe_history_boundary_never_rounds_into_the_future(self):
        clip = self.manifest["clips"][0]
        clip["history_end"] = clip["stage_times"][-1] = 1.12
        self.write_manifest()
        result = intake.prepare(self.root)
        self.assertEqual(result["status"], "SOURCE_READY", result)
        prepared = json.loads((self.root / "pipeline_data" / "manifest.json").read_text())["clips"][0]
        self.assertLessEqual(max(max(t) for t in prepared["history_sample_times"]), 1.12)
        self.assertAlmostEqual(prepared["history_sample_times"][-1][-1], 1.0)
        with np.load(self.root / "pipeline_data" / "clips" / (clip["id"] + ".npz"), allow_pickle=False) as arrays:
            self.assertLess(float(arrays["current_rgb"][..., 0].mean()), 72)
            np.testing.assert_array_equal(arrays["future_rgb"][0], arrays["current_rgb"])


if __name__ == "__main__":
    unittest.main()
