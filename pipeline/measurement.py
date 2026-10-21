"""Fixed CPU diagnostics and mandatory blinded review for observed video outcomes.

The pixel readers are deliberately conservative diagnostics, not learned identity
or empty-container detectors. Calibration supports agreement on nine source clips
only. Human review of generated action, exposure, camera and readouts is mandatory.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import shutil
from zipfile import BadZipFile

import numpy as np

from .config import PROMPTS


SCENARIOS = ("cup", "teapot", "candy")
SIDES = ("left", "right")
COLOUR_VOCABULARY = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
SETTINGS = {
    "version": "fixed_hsv_components_v1",
    "minimum_frames": 3,
    "temporal_agreement": .9,
    "minimum_saturation": .35,
    "minimum_value": .18,
    "minimum_colour_fraction": .02,
    "colour_dominance": .85,
    "foreground_hue_tolerance_degrees": 18.,
    "component_area_fraction": [.001, .25],
    "maximum_component_aspect": 3.5,
    "minimum_component_fill": .30,
    "maximum_component_area_ratio": 2.5,
    "maximum_noise_fraction": .10,
    "maximum_count": 20,
}
UNKNOWN = {"unknown", "illegible", "unclear", "uncertain", "unreadable", "inconclusive",
           "cannot tell", "not visible", "n/a", "na", "?"}
REQUIRED = ("camera_stable", "action_executed", "left_exposed", "right_exposed",
            "observed_left", "observed_right", "reviewer")
BOOL_FIELDS = REQUIRED[:4]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value, what):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("invalid " + what)
    return value


def _manual_reason(spec):
    if spec.get("kind") == "manual" or spec.get("manual_required") is True:
        return "explicit_manual_readout"
    if spec.get("transparent") is True:
        return "transparent_contents_require_human_observation"
    if spec.get("identity_uncertain") is True or spec.get("identity_confident") is False:
        return "uncertain_identity_requires_human_observation"
    return None


def _abstain(reason, diagnostics=None, support=0.):
    return {"judgeable": False, "value": None, "support": float(support),
            "reason": reason, "diagnostics": diagnostics or {}}


def _hsv(rgb):
    rgb = rgb.astype(np.float32) / 255.
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(delta)
    active = delta > 0
    for channel, offset in ((0, 0), (1, 2), (2, 4)):
        selected = active & (np.argmax(rgb, axis=-1) == channel)
        difference = rgb[..., (channel + 1) % 3] - rgb[..., (channel + 2) % 3]
        hue[selected] = (difference[selected] / delta[selected] + offset) % 6
    return hue * 60., saturation, maximum


def _colour_frame(rgb):
    hue, saturation, value = _hsv(rgb)
    visible = (saturation >= SETTINGS["minimum_saturation"]) & (value >= SETTINGS["minimum_value"])
    if visible.mean() < SETTINGS["minimum_colour_fraction"]:
        return None, "no_reliable_chromatic_signal"
    selected = hue[visible]
    bins = (("red", (selected < 20) | (selected >= 345)),
            ("orange", (selected >= 20) & (selected < 45)),
            ("yellow", (selected >= 45) & (selected < 75)),
            ("green", (selected >= 75) & (selected < 165)),
            ("cyan", (selected >= 165) & (selected < 195)),
            ("blue", (selected >= 195) & (selected < 265)),
            ("purple", (selected >= 265) & (selected < 300)),
            ("magenta", (selected >= 300) & (selected < 345)))
    label, mask = max(bins, key=lambda item: int(item[1].sum()))
    if float(mask.mean()) < SETTINGS["colour_dominance"]:
        return None, "competing_colours"
    return label, "dominant_chromatic_region"


def _components(mask):
    """8-connected component areas/bounds using row runs, without a GPU or cv2."""
    parents, segments, previous = [], [], []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y, row in enumerate(mask):
        transitions = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        current = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            index = len(parents)
            parents.append(index)
            segments.append((y, start, end))
            for old_start, old_end, old_index in previous:
                if old_start > end:
                    break
                if old_end >= start:
                    parents[find(old_index)] = find(index)
            current.append((start, end, index))
        previous = current
    groups = {}
    for index, (y, start, end) in enumerate(segments):
        root = find(index)
        if root not in groups:
            groups[root] = [0, start, y, end, y + 1]
        item = groups[root]
        item[0] += end - start
        item[1], item[2] = min(item[1], start), min(item[2], y)
        item[3], item[4] = max(item[3], end), max(item[4], y + 1)
    return list(groups.values())


def _count_frame(rgb, foreground_hue):
    hue, saturation, value = _hsv(rgb)
    mask = (saturation >= SETTINGS["minimum_saturation"]) & (value >= SETTINGS["minimum_value"])
    if foreground_hue is not None:
        distance = np.abs(hue - foreground_hue)
        mask &= np.minimum(distance, 360 - distance) <= SETTINGS["foreground_hue_tolerance_degrees"]
    total = int(mask.sum())
    if total == 0:
        return None, "absence_does_not_establish_empty"
    minimum, maximum = SETTINGS["component_area_fraction"]
    area = mask.size
    components = _components(mask)
    objects = [item for item in components if item[0] >= max(4, math.ceil(area * minimum))]
    noise = total - sum(item[0] for item in objects)
    if noise / total > SETTINGS["maximum_noise_fraction"]:
        return None, "excess_segmentation_noise"
    if not objects or len(objects) > SETTINGS["maximum_count"]:
        return None, "no_reliable_separated_objects"
    height, width = mask.shape
    for size, x0, y0, x1, y1 in objects:
        w, h = x1 - x0, y1 - y0
        if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
            return None, "object_crosses_roi_boundary"
        if (size > area * maximum or max(w / h, h / w) > SETTINGS["maximum_component_aspect"]
                or size / (w * h) < SETTINGS["minimum_component_fill"]):
            return None, "ambiguous_or_merged_components"
    sizes = [item[0] for item in objects]
    if max(sizes) / min(sizes) > SETTINGS["maximum_component_area_ratio"]:
        return None, "inconsistent_object_sizes"
    return len(objects), "separated_chromatic_components"


def measure(frames, roi, spec):
    """Read pixels with fixed settings. This function never accesses spec['truth']."""
    if not isinstance(spec, dict):
        return _abstain("invalid_readout_spec")
    manual = _manual_reason(spec)
    if manual:
        return _abstain(manual)
    kind = spec.get("kind")
    if kind not in ("colour", "count"):
        return _abstain("unsupported_readout_kind")
    try:
        array = np.asarray(frames)
        if array.ndim == 3:
            array = array[None]
        if (array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8
                or min(array.shape[:3]) <= 0):
            return _abstain("invalid_rgb_frames")
        bounds = np.asarray(roi, dtype=float)
        if (bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds < 0)
                or np.any(bounds > 1) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
            return _abstain("invalid_normalized_roi")
        height, width = array.shape[1:3]
        x0, y0 = int(math.floor(bounds[0] * width)), int(math.floor(bounds[1] * height))
        x1, y1 = int(math.ceil(bounds[2] * width)), int(math.ceil(bounds[3] * height))
        cropped = array[:, y0:y1, x0:x1]
        if min(cropped.shape[1:3]) < 3:
            return _abstain("roi_too_small")
        if len(cropped) < SETTINGS["minimum_frames"]:
            return _abstain("insufficient_window_frames")
        foreground_hue = None
        foreground = spec.get("foreground_rgb")
        if kind == "count" and foreground is not None:
            reference = np.asarray(foreground, dtype=float)
            if (reference.shape != (3,) or not np.isfinite(reference).all()
                    or np.any(reference < 0) or np.any(reference > 255)):
                return _abstain("invalid_foreground_colour")
            h, s, v = _hsv(reference.astype(np.uint8)[None, None])
            if s.item() < SETTINGS["minimum_saturation"] or v.item() < SETTINGS["minimum_value"]:
                return _abstain("foreground_not_chromatically_identifiable")
            foreground_hue = float(h.item())
        readings = [_colour_frame(frame) if kind == "colour" else _count_frame(frame, foreground_hue)
                    for frame in cropped]
    except (TypeError, ValueError, OverflowError) as error:
        return _abstain("invalid_measurement_input", {"error_type": type(error).__name__})
    values = [value for value, _ in readings if value is not None]
    diagnostics = {"reader": SETTINGS["version"], "window_frames": len(cropped),
                   "frame_values": [value for value, _ in readings],
                   "frame_reasons": dict(Counter(reason for _, reason in readings))}
    if not values:
        return _abstain("no_judgeable_frames", diagnostics)
    value, votes = Counter(values).most_common(1)[0]
    support = votes / len(cropped)
    if support < SETTINGS["temporal_agreement"]:
        return _abstain("insufficient_temporal_agreement", diagnostics, support)
    return {"judgeable": True, "value": value, "support": support,
            "reason": "fixed_reader_temporal_agreement", "diagnostics": diagnostics}


def _normalize_value(value, kind):
    if kind == "count":
        return value if type(value) is int and value >= 0 else None
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().split())
        return normalized if normalized and normalized not in UNKNOWN else None
    if kind == "manual" and type(value) is int and value >= 0:
        return value
    return None


def _observation_vocabularies(manifest):
    """Shared annotation metadata only; never derive a vocabulary from clip truths."""
    declared = manifest.get("observation_vocabularies", {})
    if not isinstance(declared, dict) or set(declared) - {"colour", "manual"}:
        raise ValueError("observation_vocabularies must declare shared colour/manual label lists")
    vocabularies = {"colour": list(COLOUR_VOCABULARY), "manual": None}
    for kind, labels in declared.items():
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError("observation vocabulary must be a list of labels")
        normalized = [_normalize_value(label, kind) for label in labels]
        if any(label is None for label in normalized):
            raise ValueError("unknown or blank values cannot be allowed observation labels")
        distinct = list(dict.fromkeys(normalized))
        if len(distinct) < 2:
            raise ValueError("observation vocabulary requires at least two distinct labels")
        vocabularies[kind] = distinct
    return vocabularies


def _readout_task(scenario, side):
    """One fixed observable property per scenario/side, independent of every truth."""
    if scenario == "cup" and side == "left":
        return {"property": "liquid_presence", "kinds": ("manual",),
                "question": "Is liquid visibly present in the cup within the readout region?",
                "labels": ["empty", "nonempty"], "answer_definitions": {
                    "empty": "No liquid is visibly present in the cup; transparency alone does not establish this.",
                    "nonempty": "Liquid is visibly present in the cup, regardless of its colour."}}
    if scenario == "cup" and side == "right":
        return {"property": "visible_spill_mark", "kinds": ("manual",),
                "question": "Is a spill mark visible on the surface within the readout region?",
                "labels": ["clean", "marked"], "answer_definitions": {
                    "clean": "No spill mark is visible on the surface.",
                    "marked": "A spill mark is visible; this does not assert moisture or chemical state."}}
    if scenario == "teapot":
        glass = "display glass" if side == "left" else "receiver glass"
        definitions = {"clear": "Visible liquid is colourless; this does not mean the container is empty."}
        definitions.update({hue: "Visible liquid has the named " + hue + " hue." for hue in COLOUR_VOCABULARY})
        definitions["other"] = "Visible liquid has a discernible colour outside the eight named hues."
        return {"property": "liquid_colour", "kinds": ("colour", "manual"),
                "question": "What is the visible colour of liquid in the " + glass + " within the readout region?",
                "labels": ["clear"] + list(COLOUR_VOCABULARY) + ["other"], "answer_definitions": definitions}
    if scenario == "candy":
        return {"property": "candy_count", "kinds": ("count",),
                "question": "How many individual candies are visibly present within the readout region?",
                "labels": None, "answer_definitions": {
                    "nonnegative_json_integer": "Count each distinguishable candy once; use unknown if hidden, merged or uncertain."}}
    raise ValueError("no fixed readout question for scenario/side")


def _task_options(scenario, side, spec, vocabularies):
    task = _readout_task(scenario, side)
    compatible = spec["kind"] in task["kinds"]
    if task["labels"] is None:
        return task, None, [], compatible
    shared = vocabularies.get(spec["kind"])
    allowed = None if shared is None else [label for label in task["labels"] if label in shared]
    missing = [label for label in task["labels"] if shared is None or label not in shared]
    return task, allowed, missing, compatible


def _observation_value(value, scenario, side, spec, vocabularies):
    task, allowed, missing, compatible = _task_options(scenario, side, spec, vocabularies)
    if not compatible or missing:
        return None
    if task["labels"] is None:
        return _normalize_value(value, "count")
    normalized = _normalize_value(value, "manual")
    return normalized if allowed is not None and normalized in allowed else None


def _readout_instructions(clip, vocabularies):
    instructions = {}
    for side in SIDES:
        spec = clip["readouts"][side]
        roi = np.asarray(clip.get("rois", {}).get(side), dtype=float)
        window = np.asarray(clip.get("window_fractions", {}).get(side), dtype=float)
        if (roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi < 0) or np.any(roi > 1)
                or roi[0] >= roi[2] or roi[1] >= roi[3]):
            raise ValueError("review requires a valid normalized readout ROI")
        if (window.shape != (2,) or not np.isfinite(window).all() or window[0] < 0
                or window[1] > 1 or window[0] > window[1]):
            raise ValueError("review requires valid normalized reveal-window fractions")
        kind = spec["kind"]
        task, allowed, missing, compatible = _task_options(clip["scenario"], side, spec, vocabularies)
        instructions[side] = {"kind": kind, "roi": roi.tolist(), "window_fractions": window.tolist(),
                              "property": task["property"], "question": task["question"],
                              "answer_definitions": task["answer_definitions"],
                              "allowed_observations": allowed, "missing_observation_labels": missing,
                              "task_compatible": compatible,
                              "unknown_observations": ["unknown", "illegible"],
                              "observation_format": "task_incompatible_no_semantic_score" if not compatible else
                              "task_vocabulary_incomplete_no_semantic_score" if missing else
                              "nonnegative_json_integer" if task["labels"] is None else "normalized_exact_label"}
    return instructions


def _validate_clip(clip):
    _safe_name(clip.get("id"), "clip id")
    if clip.get("scenario") not in SCENARIOS:
        raise ValueError("unknown source scenario")
    verified = clip.get("verified", {})
    if not all(verified.get(key) is True for key in
               ("uncut", "history_actions_visible", "outcomes_visible", "actual_not_planned")):
        raise ValueError("source must contain verified actual observations")
    for side in SIDES:
        spec = clip.get("readouts", {}).get(side, {})
        if spec.get("kind") not in ("colour", "count", "manual"):
            raise ValueError("missing actual readout kind")
        if _normalize_value(spec.get("truth"), spec["kind"]) is None:
            raise ValueError("missing or inconclusive actual readout")


def _manifest(root):
    data = _read_json(Path(root) / "pipeline_data/manifest.json")
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("clips"), list):
        raise ValueError("invalid prepared manifest")
    clips = data["clips"]
    if not all(isinstance(clip, dict) and isinstance(clip.get("id"), str) for clip in clips):
        raise ValueError("invalid prepared clip records")
    if len({clip["id"] for clip in clips}) != len(clips):
        raise ValueError("duplicate prepared clip IDs")
    return data


def _window_frames(frames, fractions):
    bounds = np.asarray(fractions, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] < 0
            or bounds[1] > 1 or bounds[0] > bounds[1]):
        raise ValueError("invalid prepared window fractions")
    if frames.ndim != 4 or frames.shape[0] != 21 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("invalid prepared future_rgb")
    positions = np.linspace(0., 1., len(frames))
    return frames[(positions >= bounds[0] - 1e-9) & (positions <= bounds[1] + 1e-9)]


def calibrate(root):
    """Check fixed diagnostics on exactly nine verified calibration source clips."""
    root = Path(root)
    state_path = root / "pipeline_state/measurement.json"
    ready_path = root / "pipeline_state/MEASUREMENT_READY.json"
    result = {"status": "WAITING_RECORDINGS", "ready": False,
              "claim_scope": "finite_calibration_clips_only", "clip_count": 0,
              "auto_judgeable_readouts": 0, "auto_correct_readouts": 0,
              "manual_verified_readouts": 0, "rows": [], "errors": [],
              "fixed_reader_settings": SETTINGS}

    def finish(status):
        result["status"] = status
        result["ready"] = status == "CALIBRATED"
        _write_json(state_path, result)
        if result["ready"]:
            _write_json(ready_path, result)
        elif ready_path.exists():
            ready_path.unlink()
        return result

    source_ready = root / "pipeline_data/SOURCE_READY.json"
    if not source_ready.is_file() or not (root / "pipeline_data/manifest.json").is_file():
        return finish("WAITING_RECORDINGS")
    try:
        if _read_json(source_ready).get("calibration_ready") is not True:
            return finish("WAITING_RECORDINGS")
        manifest = _manifest(root)
        vocabularies = _observation_vocabularies(manifest)
        result["observation_vocabularies"] = vocabularies
        clips = [clip for clip in manifest["clips"] if clip.get("split") == "calibration"]
        result["clip_count"] = len(clips)
        if len(clips) != 9 or Counter(clip.get("scenario") for clip in clips) != Counter({s: 3 for s in SCENARIOS}):
            raise ValueError("calibration requires exactly three actual clips per scenario (nine total)")
        frozen = []
        for clip in sorted(clips, key=lambda item: item["id"]):
            _validate_clip(clip)
            path = root / "pipeline_data/clips" / (clip["id"] + ".npz")
            frozen.append({"clip": clip, "npz_sha256": _file_digest(path)})
        result["calibration_fingerprint"] = _digest({"clips": frozen, "observation_vocabularies": vocabularies,
                                                    "task_semantics": "fixed_scenario_side_property_v1"})
        waiting = False
        mismatch = False
        for clip in clips:
            with (root / "pipeline_data/clips" / (clip["id"] + ".npz")).open("rb") as source:
                with np.load(source, allow_pickle=False) as arrays:
                    frames = arrays["future_rgb"]
            for side in SIDES:
                spec = clip["readouts"][side]
                selected = _window_frames(frames, clip.get("window_fractions", {}).get(side))
                reading = measure(selected, clip.get("rois", {}).get(side), spec)
                explicit_manual = spec["kind"] == "manual" or spec.get("manual_required") is True
                row = {"clip_id": clip["id"], "side": side, "scenario": clip["scenario"],
                       "truth": spec["truth"], "reading": reading, "correct": None,
                       "manual_required": not reading["judgeable"], "mode": "fixed_diagnostic"}
                truth_value = _observation_value(spec["truth"], clip["scenario"], side, spec, vocabularies)
                if truth_value is None:
                    waiting = True
                    row["mode"] = "readout_task_inconclusive"
                elif explicit_manual:
                    row["mode"] = "source_human_verified"
                    result["manual_verified_readouts"] += 1
                elif reading["judgeable"]:
                    measured_value = _observation_value(reading["value"], clip["scenario"], side, spec, vocabularies)
                    if measured_value is None:
                        waiting = True
                        row["mode"] = "observation_vocabulary_required"
                    else:
                        result["auto_judgeable_readouts"] += 1
                        row["correct"] = measured_value == truth_value
                        result["auto_correct_readouts"] += int(row["correct"])
                        mismatch |= not row["correct"]
                else:
                    waiting = True
                    row["mode"] = "manual_declaration_required"
                result["rows"].append(row)
        return finish("CALIBRATION_REJECTED" if mismatch else "WAITING_MEASUREMENT" if waiting else "CALIBRATED")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, BadZipFile, EOFError) as error:
        result["errors"].append(str(error))
        return finish("SOURCE_REJECTED")


def _review_directory(root, phase):
    return Path(root) / "pipeline_state/reviews" / _safe_name(phase, "review phase")


def _review_paths(directory, sample_count):
    return {"status": "WAITING_REVIEW", "directory": str(directory), "sample_count": sample_count,
            "template": str(directory / "template.json"), "answers": str(directory / "answers.json"),
            "private_mapping": str(directory / "private_map.json")}


def build_review(root, phase, samples):
    """Package an exact roster into opaque, randomized videos and blank questions."""
    root = Path(root).resolve()
    directory = _review_directory(root, phase)
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("review requires a nonempty scheduled roster")
    manifest = _manifest(root)
    vocabularies = _observation_vocabularies(manifest)
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    roster, used_ids, relevant = [], set(), {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str) or not sample["sample_id"].strip():
            raise ValueError("missing scheduled sample id")
        if sample["sample_id"] in used_ids:
            raise ValueError("duplicate scheduled sample id")
        used_ids.add(sample["sample_id"])
        clip = clips.get(sample.get("clip_id"))
        if clip is None or sample.get("scenario") != clip.get("scenario"):
            raise ValueError("scheduled sample has no matching actual source clip")
        _validate_clip(clip)
        relevant[clip["id"]] = clip
        video = Path(sample["video"])
        video = (root / video).resolve() if not video.is_absolute() else video.resolve()
        if not video.is_file():
            raise ValueError("scheduled video is missing: " + sample["sample_id"])
        row = {key: sample[key] for key in ("sample_id", "clip_id", "scenario", "arm", "seed", "memory") if key in sample}
        row.update(schedule_index=index, video=str(video), video_sha256=_file_digest(video),
                   readouts=clip["readouts"], readout_instructions=_readout_instructions(clip, vocabularies))
        roster.append(row)
    roster_digest = _digest(roster)
    clip_fingerprint = _digest([relevant[key] for key in sorted(relevant)])
    if directory.exists():
        mapping_path = directory / "private_map.json"
        if not mapping_path.is_file():
            raise ValueError("existing review phase is incomplete; refusing to replace it")
        existing = _read_json(mapping_path)
        if (existing.get("roster_fingerprint") != roster_digest or existing.get("clip_fingerprint") != clip_fingerprint
                or existing.get("observation_vocabularies") != vocabularies):
            raise ValueError("existing review phase cannot replace its frozen scheduled roster")
        return _review_paths(directory, len(roster))
    directory.mkdir(parents=True)
    (directory / "media").mkdir()
    secrets.SystemRandom().shuffle(roster)
    visible = []
    for row in roster:
        review_id = "R_" + secrets.token_hex(10)
        suffix = Path(row["video"]).suffix.lower()
        suffix = suffix if suffix in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif"} else ".video"
        relative = "media/" + review_id + suffix
        shutil.copyfile(row["video"], directory / relative)
        row.update(review_id=review_id, review_video=relative)
        item = {"sample_id": review_id, "video": relative, "requested_action": PROMPTS[row["scenario"]],
                "readouts": row["readout_instructions"]}
        item.update({field: None for field in REQUIRED})
        visible.append(item)
    template = {"version": 1, "instructions":
                "Watch each video. Record actual observations only. Use unknown or illegible when a readout cannot be seen. "
                "For each side use its readout kind, ROI and reveal window: ROI is [x0,y0,x1,y1] with "
                "coordinates normalized to the full image; window_fractions is relative to the full video duration. "
                "Answer only each side's fixed question using its answer definitions. "
                "Use a nonnegative JSON integer for a count and one allowed label for a colour or manual readout. "
                "A label for another property is incompatible: clear is not a cup-presence answer, "
                "empty is not a liquid-colour answer, and a visible mark does not establish wetness. "
                "Only letter case and whitespace are normalized; synonyms are not inferred. "
                "When the task vocabulary is incomplete or a readout kind is incompatible, "
                "record what is seen; semantic scoring remains inconclusive. "
                "Mark camera, the requested action and each region's exposure independently. "
                "Fill every question and identify the reviewer.",
                "samples": visible}
    _write_json(directory / "private_map.json", {"version": 1, "phase": phase, "samples": roster,
                "roster_fingerprint": roster_digest, "clip_fingerprint": clip_fingerprint,
                "observation_vocabularies": vocabularies})
    _write_json(directory / "template.json", template)
    _write_json(directory / "answers.json", template)
    return _review_paths(directory, len(roster))


def apply_review(root, phase):
    """Join exact blinded answer IDs, check actual visibility, then score outcomes."""
    root = Path(root).resolve()
    directory = _review_directory(root, phase)
    score = {"status": "WAITING_REVIEW", "complete": False, "judgeable": False,
             "all_correct": False, "rows": [], "errors": []}
    try:
        mapping = _read_json(directory / "private_map.json")
        scheduled = sorted(mapping["samples"], key=lambda row: row["schedule_index"])
        for sample in scheduled:
            row = {key: sample[key] for key in ("sample_id", "clip_id", "scenario", "arm", "seed", "memory") if key in sample}
            row.update(review_id=sample["review_id"], complete=False, judgeable=False,
                       state_correct=None, left_correct=None, right_correct=None,
                       observed_left=None, observed_right=None, reason="missing_actual_review")
            score["rows"].append(row)
        if not scheduled or len({row["review_id"] for row in scheduled}) != len(scheduled):
            raise ValueError("invalid private scheduled roster")
        manifest = _manifest(root)
        vocabularies = _observation_vocabularies(manifest)
        if vocabularies != mapping.get("observation_vocabularies"):
            raise ValueError("shared observation vocabulary changed after review packaging")
        clip_by_id = {clip["id"]: clip for clip in manifest["clips"]}
        relevant_ids = {row["clip_id"] for row in scheduled}
        relevant = [clip_by_id[clip_id] for clip_id in sorted(relevant_ids)]
        if _digest(relevant) != mapping["clip_fingerprint"]:
            raise ValueError("actual source records changed after review packaging")
        for sample in scheduled:
            if (_file_digest(directory / sample["review_video"]) != sample["video_sha256"]
                    or _file_digest(sample["video"]) != sample["video_sha256"]):
                raise ValueError("scheduled or blinded video changed after review packaging")
        answers_path = directory / "answers.json"
        if not answers_path.is_file():
            _write_json(directory / "score.json", score)
            return score
        document = _read_json(answers_path)
        answers = document.get("samples") if isinstance(document, dict) else document
        if not isinstance(answers, list) or not all(isinstance(item, dict) for item in answers):
            raise ValueError("answers must contain a list of actual sample observations")
        answer_ids = [item.get("sample_id") for item in answers]
        expected = {row["review_id"] for row in scheduled}
        if (not all(isinstance(item, str) for item in answer_ids) or len(set(answer_ids)) != len(answer_ids)
                or set(answer_ids) - expected):
            raise ValueError("duplicate, invalid or unscheduled answer IDs")
        by_id = {item["sample_id"]: item for item in answers}
        for sample, row in zip(scheduled, score["rows"]):
            answer = by_id.get(sample["review_id"], {})
            missing = [key for key in REQUIRED if key not in answer or answer[key] is None
                       or (isinstance(answer[key], str) and not answer[key].strip())]
            if missing:
                row["missing_fields"] = missing
                continue
            if not isinstance(answer["reviewer"], str) or _normalize_value(answer["reviewer"], "manual") is None:
                row["missing_fields"] = ["reviewer"]
                continue
            if not all(type(answer[key]) is bool for key in BOOL_FIELDS):
                row["missing_fields"] = [key for key in BOOL_FIELDS if type(answer[key]) is not bool]
                continue
            row.update(complete=True, reviewer=answer["reviewer"].strip(),
                       observed_left=answer["observed_left"], observed_right=answer["observed_right"])
            row.update({key: answer[key] for key in BOOL_FIELDS})
            if not all(answer[key] for key in BOOL_FIELDS):
                row["reason"] = "camera_action_or_exposure_inconclusive"
                continue
            observed = {side: _observation_value(answer["observed_" + side], sample["scenario"], side,
                                                 sample["readouts"][side], vocabularies)
                        for side in SIDES}
            truths = {side: _observation_value(sample["readouts"][side]["truth"],
                                               sample["scenario"], side, sample["readouts"][side], vocabularies)
                      for side in SIDES}
            if any(value is None for value in truths.values()):
                row["reason"] = "readout_vocabulary_inconclusive"
                continue
            if any(value is None for value in observed.values()):
                row["reason"] = "observed_readout_inconclusive"
                continue
            row["judgeable"] = True
            for side in SIDES:
                row[side + "_correct"] = observed[side] == truths[side]
            row["state_correct"] = row["left_correct"] and row["right_correct"]
            row["reason"] = "actual_readouts_match" if row["state_correct"] else "actual_readout_mismatch"
        score["complete"] = all(row["complete"] for row in score["rows"])
        score["judgeable"] = score["complete"] and all(row["judgeable"] for row in score["rows"])
        score["all_correct"] = score["judgeable"] and all(row["state_correct"] for row in score["rows"])
        score["status"] = "SCORED" if score["judgeable"] else "INCONCLUSIVE" if score["complete"] else "WAITING_REVIEW"
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        score["status"] = "INVALID_REVIEW" if (directory / "private_map.json").exists() else "WAITING_REVIEW"
        score["errors"].append(str(error))
    _write_json(directory / "score.json", score)
    return score
