# ForkMerge — research handoff

[中文入门](README.zh-CN.md) · [接手指南](docs/HANDOFF_ZH.md) · [Code map](docs/CODE_MAP.md) · [Evidence status](docs/EVIDENCE_STATUS.md)

This repository contains a reproducible **archived screw simulation** and a **video-memory prototype awaiting real-video validation**. It is not a completed paper implementation or evidence of method superiority.

## Replay actual archived outputs: CPU, no server account

Use Python 3.12 in a virtual environment. Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m venv .venv
# Activate the environment before the following commands.
python -m pip install -r requirements-replay.txt
python tools/reproduce_screw.py
```

The release includes 90 actual stored prediction files, six simulator audit arrays, scientific source, frozen selection metadata and original results (about 8 MB). The script checks SHA-256, reruns physical decision evaluation and statistics, and compares all 90 cells and eight contrasts to the recorded summary. Only NumPy is needed. See [the experiment README](experiments/screw_repair/README.md).

Expected output: `runtime/screw-replay/VERIFICATION.json` with status `VERIFIED_ARCHIVED_SIMULATION_REPLAY`. The original decision remains **`STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT`**. These are actual simulation outputs, not real-world screw videos or generated-video results. Full RGB sequences, frozen visual features and checkpoints are absent, so this is evaluation replay rather than full retraining reproduction.

Existing output directories are never overwritten; use `--output runtime/another-replay` for another run.

## Latest prototype: synthetic CPU interface check

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu-tested.txt
python tools/smoke_cpu.py
python -m pipeline.entry status
python -m unittest discover -s pipeline/tests -v
```

`runtime/cpu-smoke/` contains actual saved memory tensors from two identically initialized ordinary updaters, each taking one CPU optimizer step on seeded random inputs. An artificial tensor slice exercises the loss API. **No video generator runs, no videos are produced, and tensor changes do not establish semantic erasure/protection or performance.** Memory size remains 64 × 1536. Most software tests also use synthetic fixtures.

The prototype arms are `paired` and `paired_worst_witness`: the same ordinary updater and paired flow supervision, with an additional worst-readout-region relation term in the latter. This may be ordinary loss reweighting; incremental value remains untested. TinyHistory is not integrated.

## Continue research

- Start with [the handoff guide](docs/HANDOFF_ZH.md), [code map](docs/CODE_MAP.md) and [missing evidence/resources](docs/EVIDENCE_STATUS.md).
- Use [the source contract](pipeline/CONTRACT.md) and [annotation template](recording/pipeline_annotations_template.json) for real recordings. The template is prospective and cannot admit training. Nine calibration clips are separate from at least 27 disjoint formal train/dev/test clips; minimum counts are not statistical sufficiency.
- CPU intake/calibration/review use `python -m pipeline.entry`. Without recordings, status reports `WAITING_RECORDINGS`; its phase gates report `WAITING_VERIFIED_RECORDINGS`.
- Native video stages require Linux, separately installed [VACE](https://github.com/ali-vilab/VACE) at commit `48eb44f1c4be87cc65a98bff985a26976841e9f3`, Wan2.1-VACE-1.3B weights, verified recordings, readout/visibility review and authentic settled budget records. Configure `FORKMERGE_VACE_REPO`, `FORKMERGE_MODEL_DIR`, `FORKMERGE_BUDGET_PARENT`. Native generation/backward on the proposed cases remains unvalidated.
- The native guard is the original campaign's once-only budget continuation, **not a generic fresh-run trainer**. A new independent campaign needs a new explicit protocol and budget entrypoint. Do not fabricate old receipts or bypass data/review checks. CPU replay/setup needs none of these private records.

The earlier canonical-state plus protected-witness model is in `experiments/screw_repair/code/models.py`; the latest ordinary updater is in `pipeline/model.py`. Their tasks/results are distinct. [Handoff verification](validation/HANDOFF_CHECKS.json) records the tested environment and checks; it is not new model-efficacy evidence.

Retain upstream terms and attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project-wide distribution license has been selected. No acceptance or method-performance claim is made.
