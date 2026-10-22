# Code and artifact map

## Latest video prototype

|File|Interface / purpose|
|---|---|
|`pipeline/config.py`|Frozen capacity, action vocabulary, seeds, original budget constants; external paths use environment variables|
|`pipeline/model.py`|`OrdinaryUpdater.forward(features, actions, trace=False)`: `(B,4,64,1536)` features + `(B,4)` action IDs → `(B,64,1536)` memory; optionally stage tensors|
|`pipeline/model.py`|`paired_objective`: three flow predictions/targets `(3,C,T,H,W)` and readout masks `(3,T,H,W)`; both arms share the same graph|
|`pipeline/intake.py`|Observed-source admission, no future leakage or split overlap, immutable RGB preparation|
|`pipeline/measurement.py`|Fixed readout calibration, visibility review templates and observed-answer scoring|
|`pipeline/spine.py`|External VACE loader, frozen video model and `Reader` memory-injection hook|
|`pipeline/worker.py`|Shared features, native flow loss, training, video generation and actual tensor saving|
|`pipeline/policy.py`|CPU readiness gates and cost arithmetic|
|`pipeline/guard.py`|Linux-only native admission, authentic original budget, once-only phases and owned-process cleanup|
|`pipeline/entry.py`|`status`, `prepare`, `calibrate`, `review`, `run`, `report`|
|`tools/smoke_cpu.py`|Synthetic CPU forward/backward, identical initialization, actual tensor artifacts; no video model|
|`pipeline/tests/`|Synthetic regression checks and archived-result replay checks|

Ignored runtime directories: `incoming/` raw videos; `pipeline_data/` prepared RGB; `pipeline_cache/` shared features; `pipeline_state/` readiness/reviews; `pipeline_results/` native outputs; `runtime/` smoke/replay outputs. These are different evidence types.

Full schema: [pipeline/CONTRACT.md](../pipeline/CONTRACT.md). Upstream VACE interfaces and device/precision compatibility remain outside CPU verification.

## Archived screw simulation

All paths below are within `experiments/screw_repair/`, except the portable tool.

|Path|Purpose|
|---|---|
|`code/models.py`|Earlier `CanonicalWitness`, ordinary baseline construction, equivalent/surviving response supervision|
|`code/baseline_model.py`|Ordinary transformer/GRU heads|
|`code/train.py`, `evaluate.py`|Original scientific training/selection/prediction source; missing external feature caches/checkpoints disclosed|
|`code/physics.py`, `physics_prior.py`|Idealized interface dynamics; not hardware calibration|
|`code/family_source.py`, `render.py`, `feature_source.py`|Family generation, instrumented RGB and feature extraction source|
|`code/analyze.py`, `baseline_metrics.py`|Physical decision execution, metrics, paired family bootstrap and frozen screen|
|`artifacts/predictions/`|90 byte-preserved predictions/targets/family-ID files|
|`artifacts/data/`|Six byte-preserved simulator audit files; privileged truth for evaluation only|
|`artifacts/evidence/FIT_FREEZE.json`|Anonymized frozen selections; referenced checkpoints absent|
|`expected/`|Original summary, 90-cell table and report, including negative results|
|`MANIFEST.json`|Released/original SHA-256 and disclosure of path/encoding changes|
|`tools/reproduce_screw.py` (repository root)|Portable CPU replay and all-cell/all-contrast verification|

The two code generations solve different tasks. Archived simulation results cannot establish performance of the newer video prototype.
