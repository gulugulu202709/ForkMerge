"""CPU SELFTEST fixtures only; no experiment or method-result files are read."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.measurement import apply_review, build_review, calibrate, measure


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def colour_frames(colour=(225, 25, 20), count=21):
    frames = np.full((count, 60, 80, 3), 120, dtype=np.uint8)
    frames[:, 10:50, 8:72] = colour
    return frames


def count_frames(number=3):
    frames = np.full((21, 60, 80, 3), 120, dtype=np.uint8)
    for x in (10, 32, 54)[:number]:
        frames[:, 20:28, x:x + 8] = (225, 25, 20)
    return frames


class TruthMustNotBeRead(dict):
    def get(self, key, default=None):
        if key == "truth":
            raise AssertionError("prediction read the answer")
        return super().get(key, default)

    def __getitem__(self, key):
        if key == "truth":
            raise AssertionError("prediction read the answer")
        return super().__getitem__(key)


class PixelReaderSelftest(unittest.TestCase):
    def test_colour_prediction_ignores_wrong_truth(self):
        result = measure(colour_frames(), [0, 0, 1, 1],
                         TruthMustNotBeRead(kind="colour", truth="blue"))
        self.assertTrue(result["judgeable"])
        self.assertEqual(result["value"], "red")

    def test_count_prediction_is_measured_from_separated_components(self):
        result = measure(count_frames(3), [0, 0, 1, 1],
                         TruthMustNotBeRead(kind="count", truth=9,
                                            foreground_rgb=[225, 25, 20]))
        self.assertTrue(result["judgeable"])
        self.assertEqual(result["value"], 3)

    def test_colour_abstains_on_neutral_or_competing_colours(self):
        grey = np.full((21, 60, 80, 3), 120, dtype=np.uint8)
        competing = colour_frames()
        competing[:, 10:50, 40:72] = (20, 30, 225)
        for frames in (grey, competing):
            with self.subTest(neutral=frames is grey):
                result = measure(frames, [0, 0, 1, 1], {"kind": "colour"})
                self.assertFalse(result["judgeable"])
                self.assertIsNone(result["value"])

    def test_temporally_unstable_colour_abstains(self):
        frames = colour_frames()
        frames[::2] = colour_frames((20, 30, 225), 11)
        self.assertFalse(measure(frames, [0, 0, 1, 1], {"kind": "colour"})["judgeable"])

    def test_count_abstains_on_zero_boundary_and_unstable_components(self):
        empty = count_frames(0)
        boundary = count_frames(1)
        boundary[:, 20:28, :18] = (225, 25, 20)
        unstable = count_frames(3)
        unstable[::2] = count_frames(2)[::2]
        for frames in (empty, boundary, unstable):
            with self.subTest(case=frames is empty):
                result = measure(frames, [0, 0, 1, 1], {"kind": "count"})
                self.assertFalse(result["judgeable"])
                self.assertIsNone(result["value"])

    def test_manual_and_uncertain_identity_never_predict(self):
        for spec in ({"kind": "manual"}, {"kind": "colour", "manual_required": True},
                     {"kind": "count", "identity_uncertain": True},
                     {"kind": "colour", "transparent": True}):
            with self.subTest(spec=spec):
                self.assertFalse(measure(colour_frames(), [0, 0, 1, 1], spec)["judgeable"])

    def test_roi_excludes_a_competing_colour_outside_the_readout(self):
        frames = colour_frames()
        frames[:, :, 40:] = (20, 30, 225)
        result = measure(frames, [0, 0, .5, 1], {"kind": "colour"})
        self.assertTrue(result["judgeable"])
        self.assertEqual(result["value"], "red")

    def test_unreliable_foreground_colour_cannot_produce_count(self):
        result = measure(count_frames(3), [0, 0, 1, 1],
                         {"kind": "count", "foreground_rgb": [120, 120, 120]})
        self.assertFalse(result["judgeable"])
        self.assertIsNone(result["value"])

    def test_bad_pixels_or_roi_return_abstention_without_value(self):
        for frames, roi in ((colour_frames().astype(float), [0, 0, 1, 1]),
                            (colour_frames(), [-.1, 0, 1, 1]),
                            (colour_frames(), [1, 0, 0, 1])):
            with self.subTest(roi=roi):
                result = measure(frames, roi, {"kind": "colour"})
                self.assertFalse(result["judgeable"])
                self.assertIsNone(result["value"])


class PreparedSelftest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="SELFTEST_measurement_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        clips = []
        for scenario in ("cup", "teapot", "candy"):
            for index in range(3):
                clip_id = "SELFTEST_%s_%s" % (scenario, index)
                clip = {"id": clip_id, "file": clip_id + ".mp4",
                        "recording_id": clip_id, "unit_id": clip_id,
                        "split": "calibration", "scenario": scenario,
                        "history_end": 1., "future_end": 6.,
                        "stage_times": [0., .3, .6, 1.],
                        "actions": ["observe", "interact", "replace_cup", "occlude"],
                        "windows": {"left": [3., 6.], "right": [3., 6.]},
                        "window_fractions": {"left": [.4, 1.], "right": [.4, 1.]},
                        "rois": {"left": [0, 0, .5, 1], "right": [.5, 0, 1, 1]},
                        "readouts": {"left": {"kind": "colour", "truth": "red"},
                                     "right": {"kind": "colour", "truth": "red"}},
                        "verified": {"uncut": True, "history_actions_visible": True,
                                     "outcomes_visible": True, "actual_not_planned": True},
                        "source_note": "SELFTEST actual synthetic pixels, not experimental data"}
                if scenario == "cup":
                    clip["readouts"] = {"left": {"kind": "manual", "truth": "empty"},
                                        "right": {"kind": "manual", "truth": "clean"}}
                elif scenario == "candy":
                    clip["readouts"] = {"left": {"kind": "count", "truth": 3},
                                        "right": {"kind": "count", "truth": 3}}
                clips.append(clip)
                frames = colour_frames()
                if scenario == "candy":
                    frames[:] = 120
                    for x in (4, 16, 28, 44, 56, 68):
                        frames[:, 20:28, x:x + 6] = (225, 25, 20)
                path = self.root / "pipeline_data" / "clips" / (clip_id + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, future_rgb=frames, current_rgb=frames[0],
                                    history_rgb=np.broadcast_to(frames[0], (4, 5) + frames.shape[1:]))
        colours = ["clear", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "other"]
        self.manifest = {"version": 1, "dataset_id": "SELFTEST_only", "clips": clips, "units": [],
                         "observation_vocabularies": {"colour": colours,
                                                      "manual": ["empty", "nonempty", "clean", "marked"] + colours}}
        self.save_manifest()
        write_json(self.root / "pipeline_data" / "SOURCE_READY.json", {"calibration_ready": True})

    def save_manifest(self):
        write_json(self.root / "pipeline_data" / "manifest.json", self.manifest)

    def state(self):
        return json.loads((self.root / "pipeline_state" / "measurement.json").read_text(encoding="utf-8"))

    def ready(self):
        return self.root / "pipeline_state" / "MEASUREMENT_READY.json"

    def make_samples(self, scenario="cup"):
        samples = []
        for index, arm in enumerate(("paired", "paired_worst_witness")):
            video = self.root / "SELFTEST_media" / (arm + "_20261021.mp4")
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"SELFTEST non-video placeholder for review packaging " + bytes([index]))
            samples.append({"sample_id": "SELFTEST_%s_%s" % (arm, index), "video": str(video),
                            "scenario": scenario, "clip_id": "SELFTEST_" + scenario + "_0", "arm": arm,
                            "seed": 20261021, "memory": "SELFTEST actual fixture only"})
        return samples

    def write_answers(self, phase="SELFTEST_review", scenario="cup"):
        directory = self.root / "pipeline_state" / "reviews" / phase
        template = json.loads((directory / "template.json").read_text(encoding="utf-8"))
        answers = copy.deepcopy(template)
        left, right = {"cup": ("empty", "clean"), "teapot": ("red", "red"), "candy": (3, 3)}[scenario]
        for item in answers["samples"]:
            item.update(camera_stable=True, action_executed=True, left_exposed=True,
                        right_exposed=True, observed_left=left, observed_right=right,
                        reviewer="SELFTEST reviewer")
        write_json(directory / "answers.json", answers)
        return directory, answers

    def test_calibration_accepts_only_finite_observed_nine_clip_agreement(self):
        result = calibrate(self.root)
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertTrue(self.ready().exists())
        self.assertEqual(result["clip_count"], 9)
        self.assertEqual(result["auto_judgeable_readouts"], 12)
        self.assertEqual(result["auto_correct_readouts"], 12)
        self.assertEqual(result["manual_verified_readouts"], 6)
        self.assertEqual(result["claim_scope"], "finite_calibration_clips_only")

    def test_planned_or_unverified_source_cannot_calibrate(self):
        self.manifest["clips"][0]["verified"]["actual_not_planned"] = False
        self.save_manifest()
        result = calibrate(self.root)
        self.assertEqual(result["status"], "SOURCE_REJECTED")
        self.assertFalse(self.ready().exists())

    def test_missing_recordings_leave_no_ready_marker(self):
        (self.root / "pipeline_data" / "SOURCE_READY.json").unlink()
        self.assertEqual(calibrate(self.root)["status"], "WAITING_RECORDINGS")
        self.assertFalse(self.ready().exists())

    def test_wrong_automatic_readout_rejects_and_revokes_old_gate(self):
        calibrate(self.root)
        self.manifest["clips"][3]["readouts"]["left"]["truth"] = "blue"
        self.save_manifest()
        result = calibrate(self.root)
        self.assertEqual(result["status"], "CALIBRATION_REJECTED")
        self.assertFalse(self.ready().exists())
        self.assertFalse(result["rows"][6]["correct"])

    def test_automatic_abstention_waits_instead_of_becoming_success_or_error(self):
        frames = np.full((21, 60, 80, 3), 120, dtype=np.uint8)
        np.savez_compressed(self.root / "pipeline_data/clips/SELFTEST_teapot_0.npz", future_rgb=frames)
        result = calibrate(self.root)
        self.assertEqual(result["status"], "WAITING_MEASUREMENT")
        self.assertFalse(self.ready().exists())
        self.assertIsNone(result["rows"][6]["correct"])
        self.assertTrue(result["rows"][6]["manual_required"])

    def test_explicit_manual_readout_uses_verified_source_without_accuracy_credit(self):
        readout = self.manifest["clips"][0]["readouts"]["left"]
        readout.update(kind="manual", truth="empty")
        self.save_manifest()
        result = calibrate(self.root)
        self.assertEqual(result["status"], "CALIBRATED")
        self.assertEqual(result["manual_verified_readouts"], 6)
        self.assertEqual(result["auto_correct_readouts"], 12)
        self.assertEqual(result["rows"][0]["mode"], "source_human_verified")
        self.assertIsNone(result["rows"][0]["correct"])

    def test_fewer_or_duplicate_calibration_clips_cannot_pass(self):
        for clips in (self.manifest["clips"][:-1],
                      self.manifest["clips"][:-1] + [self.manifest["clips"][0]]):
            self.manifest["clips"] = clips
            self.save_manifest()
            self.assertEqual(calibrate(self.root)["status"], "SOURCE_REJECTED")
            self.assertFalse(self.ready().exists())

    def test_formal_append_preserves_calibration_binding(self):
        first = calibrate(self.root)
        formal = copy.deepcopy(self.manifest["clips"][0])
        formal.update(id="SELFTEST_formal_append", split="train")
        self.manifest["clips"].append(formal)
        self.save_manifest()
        second = calibrate(self.root)
        self.assertEqual(second["status"], "CALIBRATED")
        self.assertEqual(first["calibration_fingerprint"], second["calibration_fingerprint"])

    def test_reader_uses_only_annotated_window_not_earlier_unrevealed_frames(self):
        frames = colour_frames()
        frames[:8] = colour_frames((20, 30, 225), 8)
        np.savez_compressed(self.root / "pipeline_data/clips/SELFTEST_teapot_0.npz", future_rgb=frames)
        self.assertEqual(calibrate(self.root)["status"], "CALIBRATED")

    def test_corrupted_npz_refuses_calibration_without_ready_marker(self):
        for content in (b"PK\x03\x04SELFTEST corrupted zip", b""):
            (self.root / "pipeline_data/clips/SELFTEST_cup_0.npz").write_bytes(content)
            with self.subTest(empty=not content):
                self.assertEqual(calibrate(self.root)["status"], "SOURCE_REJECTED")
                self.assertFalse(self.ready().exists())

    def test_review_template_has_no_answers_truths_or_arm_names(self):
        samples = self.make_samples()
        result = build_review(self.root, "SELFTEST_review", samples)
        directory = Path(result["directory"])
        visible = (directory / "template.json").read_text(encoding="utf-8")
        self.assertNotIn("paired", visible)
        self.assertNotIn("truth", visible)
        self.assertNotIn("SELFTEST_cup_0", visible)
        template = json.loads(visible)
        self.assertEqual(len(template["samples"]), 2)
        for item in template["samples"]:
            self.assertIsNone(item["observed_left"])
            self.assertIsNone(item["camera_stable"])
            self.assertTrue(item["requested_action"])
            self.assertEqual(item["readouts"]["left"]["kind"], "manual")
            self.assertEqual(item["readouts"]["left"]["roi"], [0, 0, .5, 1])
            self.assertEqual(item["readouts"]["right"]["roi"], [.5, 0, 1, 1])
            self.assertEqual(item["readouts"]["left"]["window_fractions"], [.4, 1.])
            self.assertEqual(item["readouts"]["left"]["allowed_observations"],
                             ["empty", "nonempty"])
            self.assertTrue((directory / item["video"]).is_file())
        mapping = json.loads((directory / "private_map.json").read_text(encoding="utf-8"))
        self.assertEqual({r["sample_id"] for r in mapping["samples"]}, {s["sample_id"] for s in samples})
        self.assertTrue(all(row["readouts"]["left"]["truth"] == "empty" for row in mapping["samples"]))

    def test_unanswered_review_waits_and_counts_every_scheduled_sample(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "WAITING_REVIEW")
        self.assertFalse(score["complete"])
        self.assertFalse(score["judgeable"])
        self.assertFalse(score["all_correct"])
        self.assertEqual(len(score["rows"]), 2)

    def test_blinded_ids_not_answer_order_join_to_exact_original_samples(self):
        samples = self.make_samples()
        build_review(self.root, "SELFTEST_review", samples)
        directory, answers = self.write_answers()
        answers["samples"].reverse()
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertTrue(score["complete"])
        self.assertTrue(score["judgeable"])
        self.assertTrue(score["all_correct"])
        self.assertEqual([r["sample_id"] for r in score["rows"]], [s["sample_id"] for s in samples])
        self.assertEqual(len(score["rows"]), 2)
        self.assertTrue((directory / "score.json").is_file())

    def test_judgeable_wrong_readout_is_error_only_after_valid_exposure_review(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        answers["samples"][0]["observed_left"] = "nonempty"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertTrue(score["complete"])
        self.assertTrue(score["judgeable"])
        self.assertFalse(score["all_correct"])
        self.assertEqual(sum(row["state_correct"] is False for row in score["rows"]), 1)

    def test_unknown_or_failed_camera_action_exposure_is_inconclusive(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        for field, value in (("observed_left", "unknown"), ("observed_right", "illegible"),
                             ("camera_stable", False), ("action_executed", False),
                             ("left_exposed", False), ("right_exposed", False)):
            directory, answers = self.write_answers()
            answers["samples"][0][field] = value
            write_json(directory / "answers.json", answers)
            score = apply_review(self.root, "SELFTEST_review")
            with self.subTest(field=field):
                self.assertTrue(score["complete"])
                self.assertFalse(score["judgeable"])
                self.assertFalse(score["all_correct"])
                self.assertEqual(sum(row["state_correct"] is None for row in score["rows"]), 1)

    def test_missing_fields_and_missing_ids_wait_without_silent_omission(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        for remove in ("reviewer", "observed_left", "camera_stable", "sample"):
            directory, answers = self.write_answers()
            if remove == "sample":
                answers["samples"].pop()
            else:
                del answers["samples"][0][remove]
            write_json(directory / "answers.json", answers)
            score = apply_review(self.root, "SELFTEST_review")
            with self.subTest(remove=remove):
                self.assertEqual(score["status"], "WAITING_REVIEW")
                self.assertFalse(score["complete"])
                self.assertEqual(len(score["rows"]), 2)

    def test_duplicate_and_extra_answer_ids_are_rejected(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        for extra in (False, True):
            directory, answers = self.write_answers()
            if extra:
                answers["samples"].append(dict(answers["samples"][0], sample_id="SELFTEST_unscheduled"))
            else:
                answers["samples"][1]["sample_id"] = answers["samples"][0]["sample_id"]
            write_json(directory / "answers.json", answers)
            score = apply_review(self.root, "SELFTEST_review")
            with self.subTest(extra=extra):
                self.assertEqual(score["status"], "INVALID_REVIEW")
                self.assertFalse(score["complete"])
                self.assertFalse(score["all_correct"])
                self.assertEqual(len(score["rows"]), 2)

    def test_original_ids_and_planned_fields_cannot_substitute_actual_blinded_answers(self):
        samples = self.make_samples()
        build_review(self.root, "SELFTEST_review", samples)
        directory, answers = self.write_answers()
        answers["samples"][0]["sample_id"] = samples[0]["sample_id"]
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INVALID_REVIEW")
        directory, answers = self.write_answers()
        for item in answers["samples"]:
            item["planned_left"] = item.pop("observed_left")
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "WAITING_REVIEW")

    def test_changed_truth_or_changed_review_video_invalidates_review(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        self.write_answers()
        self.manifest["clips"][0]["readouts"]["left"]["truth"] = "blue"
        self.save_manifest()
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INVALID_REVIEW")
        self.assertEqual(len(score["rows"]), 2)
        self.assertFalse(score["all_correct"])

    def test_changed_blinded_video_cannot_reuse_previously_filled_answers(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        (directory / answers["samples"][0]["video"]).write_bytes(b"SELFTEST replaced after packaging")
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INVALID_REVIEW")
        self.assertFalse(score["all_correct"])
        self.assertEqual(len(score["rows"]), 2)

    def test_explicit_manual_outcome_still_requires_generated_video_readout(self):
        self.manifest["clips"][0]["readouts"]["left"] = {"kind": "manual", "truth": "empty"}
        self.save_manifest()
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        answers["samples"][0]["observed_left"] = "unknown"
        answers["samples"][1]["observed_left"] = "empty"
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INCONCLUSIVE")
        answers["samples"][0]["observed_left"] = "empty"
        write_json(directory / "answers.json", answers)
        self.assertTrue(apply_review(self.root, "SELFTEST_review")["all_correct"])

    def test_count_review_requires_observed_integer_and_reviewer_identity(self):
        build_review(self.root, "SELFTEST_review", self.make_samples("candy"))
        directory, answers = self.write_answers(scenario="candy")
        for item in answers["samples"]:
            item["observed_left"] = 3
        write_json(directory / "answers.json", answers)
        self.assertTrue(apply_review(self.root, "SELFTEST_review")["all_correct"])
        answers["samples"][0]["observed_left"] = "3"
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INCONCLUSIVE")
        answers["samples"][0]["observed_left"] = 3
        answers["samples"][0]["reviewer"] = "unknown"
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "WAITING_REVIEW")

    def test_visible_readout_context_and_vocabulary_do_not_select_actual_answer(self):
        self.manifest["clips"][0]["readouts"]["left"]["truth"] = "nonempty"
        self.save_manifest()
        samples = self.make_samples()
        samples[1]["clip_id"] = "SELFTEST_cup_1"
        result = build_review(self.root, "SELFTEST_review", samples)
        visible = json.loads((Path(result["directory"]) / "template.json").read_text(encoding="utf-8"))
        self.assertEqual(visible["samples"][0]["readouts"], visible["samples"][1]["readouts"])
        for item in visible["samples"]:
            for side in ("left", "right"):
                self.assertEqual(set(item["readouts"][side]),
                                 {"kind", "roi", "window_fractions", "allowed_observations",
                                  "unknown_observations", "observation_format", "property", "question",
                                  "answer_definitions", "missing_observation_labels", "task_compatible"})

    def test_manual_missing_shared_vocabulary_is_inconclusive_even_on_exact_answer(self):
        self.manifest["clips"][0]["readouts"]["left"] = {
            "kind": "manual", "truth": "SELFTEST_secret_observed_label",
            "allowed_observations": ["SELFTEST_secret_observed_label", "other"]}
        del self.manifest["observation_vocabularies"]["manual"]
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "WAITING_MEASUREMENT")
        result = build_review(self.root, "SELFTEST_review", self.make_samples())
        visible_text = (Path(result["directory"]) / "template.json").read_text(encoding="utf-8")
        self.assertNotIn("SELFTEST_secret_observed_label", visible_text)
        template = json.loads(visible_text)
        self.assertIsNone(template["samples"][0]["readouts"]["left"]["allowed_observations"])
        directory, answers = self.write_answers()
        for item in answers["samples"]:
            item["observed_left"] = "SELFTEST_secret_observed_label"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INCONCLUSIVE")
        self.assertTrue(score["complete"])
        self.assertTrue(all(row["state_correct"] is None for row in score["rows"]))

    def test_manual_vocabulary_is_shared_and_does_not_infer_free_text_synonyms(self):
        self.manifest["observation_vocabularies"]["manual"].append("empty_clear")
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "CALIBRATED")
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        for item in answers["samples"]:
            item["observed_left"] = "empty clear"
            self.assertEqual(item["readouts"]["left"]["allowed_observations"], ["empty", "nonempty"])
        write_json(directory / "answers.json", answers)
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INCONCLUSIVE")
        for item in answers["samples"]:
            item["observed_left"] = "  EMPTY  "
        write_json(directory / "answers.json", answers)
        self.assertTrue(apply_review(self.root, "SELFTEST_review")["all_correct"])

    def test_truth_outside_shared_vocabulary_does_not_become_wrong_state(self):
        self.manifest["clips"][0]["readouts"]["left"] = {"kind": "manual", "truth": "empty_clear"}
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "WAITING_MEASUREMENT")
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        for item in answers["samples"]:
            item["observed_left"] = "empty"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INCONCLUSIVE")
        self.assertTrue(all(row["state_correct"] is None for row in score["rows"]))

    def test_invalid_or_answer_revealing_singleton_vocabulary_is_rejected(self):
        for vocabulary in (["red"], [" red ", "RED"], ["red", "unknown"], "red,blue", ["red", 7]):
            self.manifest["observation_vocabularies"] = {"colour": vocabulary}
            self.save_manifest()
            with self.subTest(vocabulary=vocabulary):
                self.assertEqual(calibrate(self.root)["status"], "SOURCE_REJECTED")
                with self.assertRaises(ValueError):
                    build_review(self.root, "SELFTEST_review", self.make_samples())

    def test_vocabulary_is_bound_to_calibration_and_review_before_formal_append(self):
        first = calibrate(self.root)
        build_review(self.root, "SELFTEST_review", self.make_samples())
        self.write_answers()
        self.manifest["observation_vocabularies"]["manual"].append("SELFTEST_additional_label")
        self.save_manifest()
        second = calibrate(self.root)
        self.assertEqual(second["status"], "CALIBRATED")
        self.assertNotEqual(first["calibration_fingerprint"], second["calibration_fingerprint"])
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INVALID_REVIEW")

    def test_cup_fill_question_excludes_clear_even_if_shared_vocabulary_contains_it(self):
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        instructions = answers["samples"][0]["readouts"]["left"]
        self.assertEqual(instructions["property"], "liquid_presence")
        self.assertEqual(instructions["allowed_observations"], ["empty", "nonempty"])
        self.assertNotIn("clear", instructions["answer_definitions"])
        self.assertTrue(instructions["question"])
        answers["samples"][0]["observed_left"] = "clear"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INCONCLUSIVE")
        self.assertEqual(sum(row["state_correct"] is None for row in score["rows"]), 1)

    def test_cup_mark_question_does_not_infer_wetness_or_colour_from_a_mark(self):
        self.manifest["observation_vocabularies"]["manual"].append("wet")
        self.save_manifest()
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        instructions = answers["samples"][0]["readouts"]["right"]
        self.assertEqual(instructions["property"], "visible_spill_mark")
        self.assertEqual(instructions["allowed_observations"], ["clean", "marked"])
        self.assertNotIn("wet", instructions["answer_definitions"])
        for wrong_property in ("wet", "blue"):
            answers["samples"][0]["observed_right"] = wrong_property
            write_json(directory / "answers.json", answers)
            self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INCONCLUSIVE")
        answers["samples"][0]["observed_right"] = "marked"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "SCORED")
        self.assertFalse(score["all_correct"])

    def test_wrong_source_property_or_kind_cannot_calibrate_or_score_as_state_error(self):
        for kind, truth in (("manual", "clear"), ("colour", "red"), ("count", 0)):
            self.manifest["clips"][0]["readouts"]["left"] = {"kind": kind, "truth": truth}
            self.save_manifest()
            self.assertEqual(calibrate(self.root)["status"], "WAITING_MEASUREMENT")
            phase = "SELFTEST_wrong_property_" + kind
            build_review(self.root, phase, self.make_samples())
            directory, answers = self.write_answers(phase)
            for item in answers["samples"]:
                item["observed_left"] = truth
            write_json(directory / "answers.json", answers)
            score = apply_review(self.root, phase)
            self.assertEqual(score["status"], "INCONCLUSIVE")
            self.assertTrue(all(row["state_correct"] is None for row in score["rows"]))

    def test_teapot_question_always_has_colour_answers_and_never_empty(self):
        self.manifest["clips"][3]["readouts"]["left"] = {"kind": "manual", "truth": "clear"}
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "CALIBRATED")
        samples = self.make_samples("teapot")
        samples[1]["clip_id"] = "SELFTEST_teapot_1"
        build_review(self.root, "SELFTEST_review", samples)
        directory, answers = self.write_answers(scenario="teapot")
        expected = ["clear", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "other"]
        for item in answers["samples"]:
            for side in ("left", "right"):
                self.assertEqual(item["readouts"][side]["property"], "liquid_colour")
                self.assertEqual(item["readouts"][side]["allowed_observations"], expected)
                self.assertNotIn("empty", item["readouts"][side]["answer_definitions"])
            item["observed_left"] = "empty"
        write_json(directory / "answers.json", answers)
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INCONCLUSIVE")
        self.assertTrue(all(row["state_correct"] is None for row in score["rows"]))

    def test_partial_task_vocabulary_is_inconclusive_even_for_present_answer(self):
        self.manifest["observation_vocabularies"]["manual"].remove("nonempty")
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "WAITING_MEASUREMENT")
        build_review(self.root, "SELFTEST_review", self.make_samples())
        directory, answers = self.write_answers()
        self.assertEqual(answers["samples"][0]["readouts"]["left"]["missing_observation_labels"], ["nonempty"])
        score = apply_review(self.root, "SELFTEST_review")
        self.assertEqual(score["status"], "INCONCLUSIVE")
        self.assertTrue(all(row["state_correct"] is None for row in score["rows"]))

    def test_default_eight_hues_cannot_claim_complete_teapot_task_vocabulary(self):
        del self.manifest["observation_vocabularies"]["colour"]
        self.save_manifest()
        self.assertEqual(calibrate(self.root)["status"], "WAITING_MEASUREMENT")
        build_review(self.root, "SELFTEST_review", self.make_samples("teapot"))
        directory, answers = self.write_answers(scenario="teapot")
        self.assertEqual(answers["samples"][0]["readouts"]["left"]["missing_observation_labels"], ["clear", "other"])
        self.assertEqual(apply_review(self.root, "SELFTEST_review")["status"], "INCONCLUSIVE")

    def test_existing_phase_cannot_silently_replace_roster_or_answers(self):
        samples = self.make_samples()
        first = build_review(self.root, "SELFTEST_review", samples)
        directory, answers = self.write_answers()
        before = (directory / "answers.json").read_bytes()
        second = build_review(self.root, "SELFTEST_review", samples)
        self.assertEqual(first["directory"], second["directory"])
        self.assertEqual(before, (directory / "answers.json").read_bytes())
        with self.assertRaises(ValueError):
            build_review(self.root, "SELFTEST_review", samples[:1])
        with self.assertRaises(ValueError):
            build_review(self.root, "SELFTEST_new", samples + [samples[0]])


if __name__ == "__main__":
    unittest.main()
