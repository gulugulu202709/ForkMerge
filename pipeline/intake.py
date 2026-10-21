"""CPU-only admission and immutable RGB preparation of verified source videos.

Nothing is decoded on import. Input annotations attest observed events; intake
checks their consistency, not whether the physical events actually happened.
"""
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import tempfile

from .config import ACTIONS, FPS, FRAMES, SCENARIOS, SIZE

_SPLITS = ("calibration", "train", "dev", "test")
_VERIFIED = ("uncut", "history_actions_visible", "outcomes_visible", "actual_not_planned",
             "camera_stable", "layout_matched")
_RESET = {"cup": {"replace_cup"}, "teapot": {"wash_refill_pot", "replace_lid"},
          "candy": {"empty_refill_box"}}
_PREPARED_FIELDS = {"window_fractions", "history_sample_times", "future_sample_times",
                    "source_fps", "source_sha256"}


def _number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _raw_path(incoming, value):
    if not _text(value):
        raise ValueError("file must be a nonempty relative path")
    portable = value.replace("\\", "/")
    if Path(portable).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("source file must be relative to incoming")
    if ".." in portable.split("/"):
        raise ValueError("source file cannot traverse outside incoming")
    root = Path(incoming).resolve()
    source = (root / portable).resolve()
    if not source.is_relative_to(root) or source == root:
        raise ValueError("resolved source file must stay inside incoming")
    return source


def _planned(value):
    """Reject explicit prospective provenance; never infer truth from scripts."""
    if not isinstance(value, dict):
        return False
    if value.get("planned_labels_are_not_truth") is True or value.get("truth_is_planned") is True:
        return True
    if any(value.get(key) is False for key in ("truth_verified", "outcome_verified", "actual_not_planned")):
        return True
    if value.get("planned") is True:
        return True
    for key in ("status", "truth_source", "provenance"):
        text = value.get(key)
        if isinstance(text, str) and any(word in text.lower() for word in ("prospective", "template", "planned", "unverified")):
            return True
    return False


def validate_manifest(manifest, incoming):
    """Return all structural/source-admission errors without changing inputs.

    Minimum corpus sizes are readiness checks in prepare, so a valid partial
    manifest can be distinguished from invalid or unverified source material.
    """
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        errors.append("manifest version must be 1")
    if not _text(manifest.get("dataset_id")):
        errors.append("dataset_id must be nonempty")
    if _planned(manifest):
        errors.append("prospective/planned manifest is not observed data")
    clips, units = manifest.get("clips"), manifest.get("units")
    if not isinstance(clips, list):
        errors.append("clips must be a list")
        clips = []
    if not isinstance(units, list):
        errors.append("units must be a list")
        units = []
    seen_ids, seen_recordings, seen_paths, clip_by_id, unit_members = set(), set(), set(), {}, {}
    for index, clip in enumerate(clips):
        prefix = f"clips[{index}]"
        if not isinstance(clip, dict):
            errors.append(prefix + " must be an object")
            continue
        clip_id = clip.get("id")
        if not _text(clip_id) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", clip_id):
            errors.append(prefix + ": id must be a safe single filename component")
        elif clip_id in seen_ids:
            errors.append(prefix + ": duplicate clip id")
        else:
            seen_ids.add(clip_id)
            clip_by_id[clip_id] = clip
        for field, seen in (("recording_id", seen_recordings),):
            value = clip.get(field)
            if not _text(value):
                errors.append(prefix + f": {field} must be nonempty")
            elif value in seen:
                errors.append(prefix + f": duplicate {field}; recordings cannot overlap splits")
            else:
                seen.add(value)
        unit_id = clip.get("unit_id")
        if not _text(unit_id):
            errors.append(prefix + ": unit_id must be nonempty")
        elif _text(clip_id):
            unit_members.setdefault(unit_id, []).append(clip)
        split, scenario = clip.get("split"), clip.get("scenario")
        if split not in _SPLITS:
            errors.append(prefix + ": split must be calibration/train/dev/test")
        if scenario not in SCENARIOS:
            errors.append(prefix + ": invalid scenario")
        try:
            source = _raw_path(incoming, clip.get("file"))
            normalized = os.path.normcase(str(source))
            if normalized in seen_paths:
                errors.append(prefix + ": duplicate resolved raw source path")
            seen_paths.add(normalized)
            if not source.is_file():
                errors.append(prefix + ": missing source file: " + str(clip.get("file")))
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(prefix + ": " + str(exc))
        history_end, future_end = clip.get("history_end"), clip.get("future_end")
        valid_span = _number(history_end) and history_end >= 0 and _number(future_end) and future_end > history_end
        if not valid_span:
            errors.append(prefix + ": require finite 0<=history_end<future_end")
        stages = clip.get("stage_times")
        if not isinstance(stages, list) or len(stages) != 4 or not all(_number(t) for t in stages):
            errors.append(prefix + ": stage_times must contain four finite times")
        elif not _number(history_end) or any(t < 0 or t > history_end for t in stages) or any(
                stages[i] > stages[i + 1] for i in range(3)) or not math.isclose(stages[-1], history_end, abs_tol=1e-8):
            errors.append(prefix + ": stages must be ordered, <=history_end, and end at history_end")
        actions = clip.get("actions")
        if (not isinstance(actions, list) or len(actions) != 4 or
                actions[0] != "observe" or actions[1] != "interact" or actions[3] != "occlude" or
                not isinstance(actions[2], str) or not isinstance(scenario, str) or
                actions[2] not in _RESET.get(scenario, set()) or
                any(not isinstance(a, str) or a not in ACTIONS for a in actions)):
            errors.append(prefix + ": actions must be observe/interact/scenario reset/occlude")
        windows, rois, readouts = clip.get("windows"), clip.get("rois"), clip.get("readouts")
        for side in ("left", "right"):
            window = windows.get(side) if isinstance(windows, dict) else None
            if (not isinstance(window, list) or len(window) != 2 or not all(_number(t) for t in window) or
                    not valid_span or not history_end <= window[0] < window[1] <= future_end):
                errors.append(prefix + f": {side} window must lie within the future span")
            roi = rois.get(side) if isinstance(rois, dict) else None
            if (not isinstance(roi, list) or len(roi) != 4 or not all(_number(t) for t in roi) or
                    not all(0 <= t <= 1 for t in roi) or not roi[0] < roi[2] or not roi[1] < roi[3]):
                errors.append(prefix + f": {side} ROI must be normalized with positive area")
            spec = readouts.get(side) if isinstance(readouts, dict) else None
            if not isinstance(spec, dict):
                errors.append(prefix + f": missing {side} observed readout")
                continue
            kind, truth = spec.get("kind"), spec.get("truth")
            if kind not in ("colour", "count", "manual"):
                errors.append(prefix + f": invalid {side} readout kind")
            valid_truth = ((type(truth) is int and truth >= 0) if kind == "count" else
                           (_text(truth) or (type(truth) is int and truth >= 0)))
            if not valid_truth or (isinstance(truth, str) and truth.strip().lower() in
                                   {"planned", "unknown", "tbd", "pending", "unverified", "not_observed", "template"}):
                errors.append(prefix + f": {side} truth must be an actual observed label/integer")
            if _planned(spec):
                errors.append(prefix + f": {side} readout is explicitly planned/unverified")
            colour = spec.get("foreground_rgb")
            if colour is not None and (not isinstance(colour, list) or len(colour) != 3 or
                                       not all(_number(v) and 0 <= v <= 255 for v in colour)):
                errors.append(prefix + f": invalid {side} foreground_rgb")
        verified = clip.get("verified")
        if not isinstance(verified, dict) or any(verified.get(key) is not True for key in _VERIFIED):
            errors.append(prefix + ": all observed source/camera/layout verifications must be true")
        if _planned(clip):
            errors.append(prefix + ": planned/unverified clip is not data")
        if not _text(clip.get("source_note")):
            errors.append(prefix + ": source_note must record provenance")
        if _PREPARED_FIELDS.intersection(clip):
            errors.append(prefix + ": input contains reserved prepared provenance fields")
    declared = set()
    for index, unit in enumerate(units):
        prefix = f"units[{index}]"
        if not isinstance(unit, dict):
            errors.append(prefix + " must be an object")
            continue
        unit_id, ids = unit.get("id"), unit.get("clip_ids")
        if not _text(unit_id):
            errors.append(prefix + ": id must be nonempty")
            continue
        if unit_id in declared:
            errors.append(prefix + ": duplicate unit declaration")
        declared.add(unit_id)
        if unit.get("split") not in _SPLITS or unit.get("scenario") not in SCENARIOS:
            errors.append(prefix + ": invalid unit split/scenario")
        if not isinstance(ids, list) or len(ids) != 3 or not all(_text(cid) for cid in ids) or len(set(ids)) != 3:
            errors.append(prefix + ": unit must contain exactly three distinct clip IDs")
            continue
        expected = [c.get("id") for c in unit_members.get(unit_id, [])]
        if len(expected) != 3 or set(expected) != set(ids):
            errors.append(prefix + ": unit membership must exactly match clip unit_id values")
        for clip_id in ids:
            member = clip_by_id.get(clip_id)
            if member is None or any(member.get(key) != unit.get(unit_key) for key, unit_key in
                                     (("unit_id", "id"), ("split", "split"), ("scenario", "scenario"))):
                errors.append(prefix + ": member missing or has mismatched unit/split/scenario")
    for unit_id, members in unit_members.items():
        if (any(not isinstance(c.get(key), str) for c in members for key in ("split", "scenario")) or
                len({(c.get("split"), c.get("scenario")) for c in members}) != 1):
            errors.append(f"unit {unit_id}: cannot cross splits/scenarios")
        if any(c.get("split") != "calibration" for c in members) and unit_id not in declared:
            errors.append(f"unit {unit_id}: formal clips require an explicit exact-three unit")
        if len(members) > 3:
            errors.append(f"unit {unit_id}: unit ID repeated outside its three members")
    calibration_counts = Counter(c.get("scenario") for c in clips if isinstance(c, dict) and
                                 c.get("split") == "calibration" and isinstance(c.get("scenario"), str))
    if any(count > 3 for count in calibration_counts.values()):
        errors.append("calibration must contain exactly three clips per scenario, not additional formal data")
    return errors


def _readiness(manifest):
    counts = Counter((clip["split"], clip["scenario"]) for clip in manifest["clips"])
    calibration = all(counts[("calibration", scenario)] == 3 for scenario in SCENARIOS)
    units = Counter((unit["split"], unit["scenario"]) for unit in manifest["units"])
    formal = all(units[(split, scenario)] >= 1 for split in ("train", "dev", "test") for scenario in SCENARIOS)
    has_formal = any(clip["split"] != "calibration" for clip in manifest["clips"])
    return calibration, formal, has_formal


def _json_read(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number: " + value)))


def _json_write(path, value, exclusive=False):
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".intake_json_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _decode(clip, path):
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    if not hasattr(cv2, "CAP_PROP_N_THREADS"):
        raise ValueError("OpenCV must support explicit FFmpeg decoder threads; no uncapped fallback")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        if not capture.isOpened():
            raise ValueError("FFmpeg could not open video")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("video has invalid FPS")
        history_times = [[max(0., float(stage) - (4 - j) / FPS) for j in range(5)]
                         for stage in clip["stage_times"]]
        future_times = np.linspace(clip["history_end"], clip["future_end"], FRAMES).tolist()
        # Select the latest actual decoded timestamp <= each requested time.
        # Phone footage may be variable-rate, so index/FPS is not a safe clock.
        requested = sorted(set(future_times + [t for stage in history_times for t in stage]))
        samples, selected_times = {}, {}
        pending = 0
        previous_time, previous_frame = None, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            if not math.isfinite(timestamp) or timestamp < 0 or (previous_time is not None and timestamp <= previous_time):
                raise ValueError("source timestamps unavailable/nonmonotone; no FPS-based leakage fallback")
            while pending < len(requested) and requested[pending] < timestamp:
                if previous_frame is None or previous_time > requested[pending]:
                    raise ValueError("requested sample precedes the first decoded source timestamp")
                target = requested[pending]
                samples[target] = previous_frame
                selected_times[target] = previous_time
                pending += 1
            if pending == len(requested):
                break
            previous_frame = cv2.cvtColor(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            previous_time = timestamp
        if previous_time is None:
            raise ValueError("source video contains no decoded frames")
        while pending < len(requested) and requested[pending] <= previous_time:
            target = requested[pending]
            samples[target] = previous_frame
            selected_times[target] = previous_time
            pending += 1
        if pending < len(requested):
            raise ValueError("future_end exceeds the last decodable source-frame timestamp")
        history = np.stack([np.stack([samples[t] for t in stage]) for stage in history_times])
        current = history[-1, -1].copy()
        future = np.stack([samples[t] for t in future_times])
        future[0] = current
        return ({"history_rgb": history, "current_rgb": current, "future_rgb": future},
                {"history_sample_times": [[selected_times[t] for t in stage] for stage in history_times],
                 "future_sample_times": [selected_times[t] for t in future_times], "source_fps": source_fps})
    finally:
        capture.release()


def _validate_artifact(path):
    import numpy as np
    expected = {"history_rgb": (4, 5, SIZE[1], SIZE[0], 3),
                "current_rgb": (SIZE[1], SIZE[0], 3), "future_rgb": (FRAMES, SIZE[1], SIZE[0], 3)}
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(expected):
            raise ValueError("prepared RGB array names changed")
        loaded = {key: arrays[key] for key in expected}
        if any(loaded[key].dtype != np.uint8 or loaded[key].shape != shape for key, shape in expected.items()):
            raise ValueError("prepared RGB shape/dtype changed")
        if (not np.array_equal(loaded["history_rgb"][-1, -1], loaded["current_rgb"]) or
                not np.array_equal(loaded["future_rgb"][0], loaded["current_rgb"])):
            raise ValueError("prepared current/future boundary changed")


def _same_calibration_extension(old, new):
    if old["dataset_id"] != new["dataset_id"] or any(c["split"] != "calibration" for c in old["clips"]):
        return False
    if any(new.get(key) != value for key, value in old.items() if key not in ("clips", "units")):
        return False
    old_clips = {c["id"]: c for c in old["clips"]}
    new_clips = {c["id"]: c for c in new["clips"]}
    if any(new_clips.get(key) != value for key, value in old_clips.items()):
        return False
    if any(c["split"] == "calibration" and c["id"] not in old_clips for c in new["clips"]):
        return False
    old_units = {u["id"]: u for u in old["units"]}
    new_units = {u["id"]: u for u in new["units"]}
    return all(new_units.get(key) == value for key, value in old_units.items())


def _fit_started(root):
    state = root / "pipeline_state"
    return any((state / name).exists() for name in ("formal_frozen.json", "fit_started.json"))


def prepare(root, manifest_path=None):
    """Validate/freeze source RGB, or report a CPU-only source-admission stop.

    An existing calibration-only freeze may be extended once with the complete
    formal corpus before a fit. Existing clip arrays and the first calibration
    manifest are never overwritten. All errors leave training unpermitted.
    """
    root = Path(root).resolve()
    incoming, data, state = root / "incoming", root / "pipeline_data", root / "pipeline_state"
    state.mkdir(parents=True, exist_ok=True)
    marker = data / "SOURCE_READY.json"
    result = {"status": "WAITING_RECORDINGS", "calibration_ready": False, "formal_ready": False,
              "reused": False, "extended": False, "errors": [], "gpu_started": False,
              "training_permitted": False}

    def finish(status, errors=()):
        result["status"] = status
        result["errors"] = list(errors)
        if status != "SOURCE_READY":
            result["calibration_ready"] = result["formal_ready"] = False
            # A marker is a revocable gate; raw recordings and frozen RGB stay
            # untouched when a changed/missing input invalidates admission.
            if marker.exists():
                marker.unlink()
        _json_write(state / "intake.json", result)
        return result

    lock = state / ".intake.lock"
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        result["status"] = "SOURCE_REJECTED"
        result["errors"] = ["another intake operation owns .intake.lock; no writes performed"]
        return result
    staging = None
    try:
        path = Path(manifest_path) if manifest_path is not None else incoming / "manifest.json"
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            return finish("WAITING_RECORDINGS", ["missing source manifest: " + str(path)])
        try:
            manifest = _json_read(path)
        except (OSError, ValueError, TypeError) as exc:
            return finish("SOURCE_REJECTED", ["source manifest cannot be read: " + str(exc)])
        errors = validate_manifest(manifest, incoming)
        if errors:
            status = "WAITING_RECORDINGS" if all(": missing source file:" in error for error in errors) else "SOURCE_REJECTED"
            return finish(status, errors)
        calibration, formal, has_formal = _readiness(manifest)
        if not calibration or (has_formal and not formal):
            return finish("WAITING_RECORDINGS", ["need exactly three calibration clips/scenario and complete train/dev/test units for any formal corpus"])
        old_source, old_prepared, old_by_id = None, None, {}
        if data.exists():
            try:
                old_source = _json_read(data / "source_manifest.json")
                old_prepared = _json_read(data / "manifest.json")
                if validate_manifest(old_source, incoming):
                    return finish("SOURCE_REJECTED", ["frozen source manifest/source files no longer valid"])
                source_annotations = copy.deepcopy(old_prepared)
                for clip in source_annotations["clips"]:
                    for field in _PREPARED_FIELDS:
                        clip.pop(field, None)
                if source_annotations != old_source:
                    raise ValueError("frozen prepared/source annotations disagree")
                old_by_id = {clip["id"]: clip for clip in old_prepared["clips"]}
                if set(old_by_id) != {clip["id"] for clip in old_source["clips"]}:
                    raise ValueError("frozen clip roster differs from frozen source manifest")
                for old_clip in old_source["clips"]:
                    prepared = old_by_id[old_clip["id"]]
                    if _digest(_raw_path(incoming, old_clip["file"])) != prepared["source_sha256"]:
                        raise ValueError("frozen raw source content changed")
                    _validate_artifact(data / "clips" / (old_clip["id"] + ".npz"))
            except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                return finish("SOURCE_REJECTED", ["existing frozen dataset is incomplete/changed: " + str(exc)])
            if manifest == old_source:
                result.update(calibration_ready=calibration, formal_ready=formal, reused=True,
                              dataset_id=manifest["dataset_id"], clip_count=len(manifest["clips"]))
                expected_marker = {"version": 1, "dataset_id": manifest["dataset_id"],
                                   "calibration_ready": calibration, "formal_ready": formal,
                                   "clip_count": len(manifest["clips"])}
                if marker.exists() and _json_read(marker) != expected_marker:
                    return finish("SOURCE_REJECTED", ["source ready marker differs from frozen corpus"])
                if not marker.exists():
                    _json_write(marker, expected_marker, exclusive=True)
                return finish("SOURCE_READY")
            if (not formal or _readiness(old_source)[1] or _fit_started(root) or
                    not _same_calibration_extension(old_source, manifest)):
                return finish("SOURCE_REJECTED", ["changed frozen input; only unchanged calibration-to-complete-formal extension before any fit is allowed"])
        import numpy as np
        staging = Path(tempfile.mkdtemp(prefix=".SELFTEST_intake_" if manifest["dataset_id"].startswith("SELFTEST") else ".intake_", dir=root))
        (staging / "clips").mkdir()
        prepared_manifest = copy.deepcopy(manifest)
        source_digests = {}
        for clip in prepared_manifest["clips"]:
            source = _raw_path(incoming, clip["file"])
            digest = _digest(source)
            if digest in source_digests:
                raise ValueError(f"duplicate raw recording content: {clip['id']} and {source_digests[digest]}")
            source_digests[digest] = clip["id"]
            if clip["id"] in old_by_id:
                clip.update({key: copy.deepcopy(old_by_id[clip["id"]][key]) for key in _PREPARED_FIELDS})
                continue
            arrays, provenance = _decode(clip, source)
            if _digest(source) != digest:
                raise ValueError("raw source changed during decoding: " + clip["id"])
            clip.update(provenance)
            clip["source_sha256"] = digest
            span = clip["future_end"] - clip["history_end"]
            clip["window_fractions"] = {side: [(t - clip["history_end"]) / span for t in clip["windows"][side]]
                                        for side in ("left", "right")}
            np.savez_compressed(staging / "clips" / (clip["id"] + ".npz"), **arrays)
        # Detect concurrent operator edits before publishing a gate.
        if _json_read(path) != manifest:
            raise ValueError("source manifest changed during intake")
        if any(_digest(_raw_path(incoming, clip["file"])) != clip["source_sha256"] for clip in prepared_manifest["clips"]):
            raise ValueError("raw source changed during intake")
        ready = {"version": 1, "dataset_id": manifest["dataset_id"], "calibration_ready": calibration,
                 "formal_ready": formal, "clip_count": len(manifest["clips"])}
        _json_write(staging / "source_manifest.json", manifest, exclusive=True)
        _json_write(staging / "manifest.json", prepared_manifest, exclusive=True)
        if old_source is None:
            if not formal:
                _json_write(staging / "calibration_manifest.json", manifest, exclusive=True)
            _json_write(staging / "SOURCE_READY.json", ready, exclusive=True)
            # Rename a complete directory; refuse a preexisting destination.
            if data.exists():
                raise ValueError("pipeline_data appeared during preparation; refusing overwrite")
            os.rename(staging, data)
            staging = None
        else:
            if not (data / "calibration_manifest.json").exists() or _json_read(data / "calibration_manifest.json") != old_source:
                raise ValueError("original calibration manifest is missing/changed")
            if marker.exists():
                marker.unlink()
            for artifact in (staging / "clips").iterdir():
                destination = data / "clips" / artifact.name
                with artifact.open("rb") as source_stream, destination.open("xb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)
            _json_write(data / "source_manifest.json", manifest)
            _json_write(data / "manifest.json", prepared_manifest)
            _json_write(marker, ready, exclusive=True)
            result["extended"] = True
        result.update(calibration_ready=calibration, formal_ready=formal,
                      dataset_id=manifest["dataset_id"], clip_count=len(manifest["clips"]))
        return finish("SOURCE_READY")
    except Exception as exc:
        # Includes codec/container exceptions from lazy CPU dependencies.
        return finish("SOURCE_REJECTED", [type(exc).__name__ + ": " + str(exc)])
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)
