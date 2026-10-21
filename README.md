# ForkMerge

Research prototype for selective history erasure and protection in video prediction.

**Current status: CPU preparation tests only. No real-video training, generated-video comparison, or method-efficacy result is available.** The 77 tests in this snapshot are software tests on synthetic fixtures, not 77 video experiments. This repository is not a completed benchmark or a final paper implementation.

## Implemented comparison

Both arms use the same ordinary recurrent updater, frozen Wan2.1-VACE-1.3B generator, 64 x 1536 memory tokens, data cache, initialization seeds, 32-update schedule and sampling settings:

- `paired`: proper flow loss plus whole-video paired residual supervision.
- `paired_worst_witness`: the same updater and losses, with an additional worst-region flow-distance contrast term over left, right and whole-video masks.

The masks describe readouts from one recorded future. They are not three counterfactual intervention rollouts. Changes in saved memory tensors alone do not establish semantic erasure or protection. The new term may amount to ordinary loss reweighting; incremental value remains untested.

The three prospective controlled cases are cup replacement with a remaining spill mark, teapot washing/refilling with a separate receiver glass, and candy-box emptying/refilling with removed objects remaining outside. Planned outcomes must never substitute for observed labels.

## CPU checks

Use Linux, Python 3.10 or newer, and the dependencies in `requirements.txt`. Native generation dependencies are separate.

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m unittest discover -s pipeline/tests -v
python -m pipeline.entry status
```

Without actual verified recordings, status must report `WAITING_VERIFIED_RECORDINGS`. CPU fixtures never produce method results. Linux is required for the deployment guard; no GPU or external API is needed for these tests.

## Data and admission

`recording/pipeline_annotations_template.json` is prospective: actual readings and times are null and all verification flags are false. It cannot admit training. See [the source schema](pipeline/CONTRACT.md), [protocol assumptions](docs/PROTOCOL.md), and [operator instructions](docs/OPERATOR_ZH.md).

The nine calibration clips are excluded from formal testing. Formal train/dev/test recordings must be disjoint. Calibration checks, a privileged renderer diagnostic, observed action/target visibility, and the existing cost guard must pass before a model comparison. Privileged diagnostics use the true endpoint and are not method predictions. The ordinary arm runs first; all-correct reviewed test outputs stop the candidate arm. Underfitting, unreadable outputs, missing data or exhausted budgets remain inconclusive.

## Native model integration

Download upstream [VACE](https://github.com/ali-vilab/VACE) and its Wan2.1-VACE-1.3B weights separately, respecting upstream terms. The integration was developed against VACE commit `48eb44f1c4be87cc65a98bff985a26976841e9f3`; native generation/training on the new cases has not been validated.

Configure external paths without editing source:

```sh
export FORKMERGE_VACE_REPO=/path/to/VACE
export FORKMERGE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B
export FORKMERGE_BUDGET_PARENT=/path/to/existing/budget/run
```

The native launch guard retains the original once-only budget continuation requirements. This review export is not a stand-alone fresh training campaign: absence of authentic, settled budget records blocks native launch. Do not fabricate successful receipts or bypass source/review checks. Model weights, real recordings, credentials and private operational logs are not distributed here.

## Layout

- `pipeline/intake.py`, `measurement.py`: verified source intake and fixed readout questions.
- `pipeline/model.py`, `worker.py`, `spine.py`: ordinary updater, supervision comparison and native generator integration.
- `pipeline/policy.py`, `guard.py`, `costs.py`: admission, stop conditions and accounting.
- `pipeline/tests/`: synthetic CPU software tests.

Third-party acknowledgements are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No paper acceptance or method-performance claim is made.
