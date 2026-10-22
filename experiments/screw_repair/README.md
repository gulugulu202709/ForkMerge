# Archived screw comparison — actual simulation outputs

This is the `direct_compare_20261015` pilot of the earlier canonical-state plus protected-witness response model. It uses an idealized instrumented damaged joint interface, not calibrated screw hardware or natural video.

From the repository root:

```bash
python -m pip install -r requirements-replay.txt
python tools/reproduce_screw.py --output runtime/screw-replay
```

The command verifies payload SHA-256, copies evaluation inputs into a new directory, executes original `code/analyze.py`, re-integrates selected maintenance actions, recomputes metrics and paired family bootstrap, then compares all90 cells/eight contrasts with `expected/SUMMARY.json`. It does not train or call a learned model. Existing output directories are never overwritten.

Original decision: **`STOP_NO_EXTRA_BENEFIT_IN_THIS_PILOT`**. Read `expected/RESULTS_REPORT.md` for local improvements, regressions and scope. Primary controls are `tf_pair` and `tf_scope`; `tf_plain` and differently represented public-RGB decoder `decoded_pair` are retained. Data/update/selection matching follows the protocol; parameter counts/FLOPs and decoded versus visual representations are not claimed equal.

## Included

- `artifacts/predictions/draw{0,1}/*.npz`: 90 actual files, with `predictions/targets` `(512,4,9)` and `family_ids` `(512,)`.
- `artifacts/data/draw{0,1}/*_audit.npz`: six actual files with simulator states, public-instrument decoded measurements, reference responses, parameters, roles, operations and family IDs. **Privileged state is evaluation truth, never a visual-model input.**
- `artifacts/configs/protocol.json`: original numerical settings/stop rule; private locations anonymized.
- `artifacts/evidence/FIT_FREEZE.json`: selections for30 draw/model/seed combinations; checkpoints not bundled.
- `expected/`: recorded summary, table and report.
- `code/`: original scientific/orchestration source with disclosed portability edits; old campaign scripts are not a quickstart.

Two draws contain512 new base families each. The three operation panels share matched families; technical model seeds are not independent physical specimens. Bootstrap averages technical seeds first and retains matching. The protocol states utility, selection, margins and planned contrasts.

## Provenance and retraining limits

`MANIFEST.json` records original/released hashes. All96 NPZ files are byte-identical to original evidence. Changes configure run/source/weight locations through `FORKMERGE_SCREW_RUN`, `FORKMERGE_SCREW_SOURCE`, `FORKMERGE_SIGLIP_WEIGHTS`, use current Python and UTF-8 text output. Numeric routines are unchanged. Newly computed checkpoint/protocol metadata hashes differ after path anonymization; all other summary fields are verified.

Full RGB, training/validation features, checkpoints, pretrained encoders and private host/budget records are absent. This is **evaluation replay with inspectable training code**, not independent end-to-end retraining. Do not use `campaign.py` as the entrypoint or manufacture missing freeze/weight/budget records. A new study needs its own data/features, upstream dependencies and explicit resource protocol.

This result does not evaluate the newer `pipeline/` video-memory prototype.
