# Module contract v1

Directory root is parent of pipeline/. New artifacts go under root/pipeline_data, root/pipeline_state, root/pipeline_results. Raw files root/incoming remain read-only. Existing report/evidence/source audit are not overwritten. CPU code must not import spine/VACE/CUDA at module import. Common config.py supplied by root.

## Source manifest

root/incoming/manifest.json: {version:1,dataset_id,observation_vocabularies(optional),clips:[...],units:[...]}. Shared vocabulary is independent of actual clip labels: colour defaults to the fixed eight reader hues; manual must supply a shared explicit vocabulary before calibration/review. Missing or out-of-vocabulary manual observations are inconclusive, not inferred synonyms. Counts are nonnegative JSON integers. Each clip:

P26 task questions/answer subsets further constrain both source calibration and review: cup-left liquid presence(empty/nonempty); cup-right visible spill mark(clean/marked), not moisture; teapot-left/right liquid colour(clear,eight standard hues,other; empty is not a colour); candy-left/right nonnegative integer count. These questions are shared per scenario/side and never generated from truth. Applicable shared colour/manual vocabulary must contain the full task subset; default eight hues alone are insufficient for teapot. Unsupported attributes/labels/readout kinds remain inconclusive instead of scoring a false mismatch.

```
id, file(relative incoming), recording_id, unit_id, split(calibration|train|dev|test), scenario(cup|teapot|candy),
history_end:float, stage_times:[t0,t1,t2,t3] (0<=nondecreasing<=history_end;last=history_end),
actions:[observe,interact,resetaction,occlude],
future_end:float,
windows:{left:[begin,end],right:[begin,end]} (>=history_end,<=future_end),
rois:{left:[x0,y0,x1,y1],right:[x0,y0,x1,y1]} normalized0..1,
readouts:{left:{kind:colour|count|manual,truth:observed_label_or_int,foreground_rgb(optional),...},right:{...}},
verified:{uncut:true,history_actions_visible:true,outcomes_visible:true,actual_not_planned:true,camera_stable:true,layout_matched:true},
source_note
```

Units: {id,split,scenario,clip_ids:[exact3ids]}. Unit IDs can repeat only among the3members of their own single unit, never across splits/scenarios. Recording IDs and resolved raw source paths unique per clip. Calibration units optional, exactly3clips/scenario(9); formal at least1unit/scenario/split(train/dev/test) each3clips. Source ID/labels not passed to model. Formal dataset = same frozen corpus both arms. Intake freezes copies of manifest and extracted arrays; no overwrite of prepared clips. Calibration-only corpus can be extended once by adding formal records before formalfreeze/anyfit; all existing calibration records/files remain unchanged. Preserve first calibration manifest. Once formal_ready, changed manifest is rejected.

## intake.py API (CPU)

`prepare(root, manifest_path=None)->dict`: missing/incomplete => status WAITING_RECORDINGS or SOURCE_REJECTED, no ready marker. Writes pipeline_state/intake.json. Success extracts each clip into pipeline_data/clips/<id>.npz with history_rgb uint8(4,5,256,448,3),current_rgb(256,448,3),future_rgb(21,256,448,3). First future frame equals current frame. Uses source times only<=history_end for history. Window bounds normalized to future span in prepared manifest clip `window_fractions`. Writes pipeline_data/manifest.json, SOURCE_READY.json with calibration_ready and formal_ready flags only after validated groups and successful decode. Future ROIs/truth used only for measurement/training targets, not formal generator inputs. Auto-reusing same frozen dataset allowed read-only; never silent overwrite. Calibration-first extension described above is allowed only with unmodified existing clip records and no prior fit; add tests for this exact lifecycle.

`validate_manifest(manifest,incoming)->list[str]` no mutation. CPU tests synthetic videos under temporary SELFTEST dirs; no outputs in pipeline_results. Exact API JSON and npz names above are shared.

## measurement.py API (CPU)

`calibrate(root)->dict`: loads actual prepared calibration clips and actual verified truth. Writes pipeline_state/measurement.json and MEASUREMENT_READY.json only if finite-sample calibration accepted; clearly marks manual-requiredreadouts. Fixed diagnostic reader parameters; no fitting/test-result tuning. Must not treat planned template as data.

`measure(frames, roi, spec)->dict`: returns judgeable,value,support,reason,diagnostics; spec contains kind and truth is not consulted for prediction. Colour/count readers may abstain; manual returns unjudgeable until observed by human.

`build_review(root,phase,samples)->dict`: each sample {sample_id,video,scenario,clip_id,arm(optional),seed(optional),memory(optional)}. Create a randomized/deidentified review JSON template (answers null; no row truth/method labels) plus private mapping under pipeline_state/reviews/<phase>/. Visible readout instructions include kind, normalizedROI, window fractions, the fixed attribute question/label definitions and applicable shared answer vocabulary. The per-arm stage/order can reveal method to the operator: independent double blinding is not established. No guessed answers.

`apply_review(root,phase)->dict`: reads answers.json with eachsample's camera_stable,action_executed,left_exposed,right_exposed bool, observed_left,observed_right, reviewer fields. Require all IDs exactly once, all required fields actual. Compare observed values to frozen clip readout truths only here. Missing => WAITING_REVIEW; illegible=>inconclusive. All scheduled samples retained. Writes score.json with complete,judgeable,all_correct,rows. No state correctness inferred from attention/metric missingness.

## worker.py / guard.py / entry.py (root)

`entry.py status` CPUonly returns prerequisite gates and observed output markers, not device/budget permission. `prepare` calls intake; `calibrate` calls measurement; no GPU. `run` revalidates both and reapplies current required review answers before launch admission. Native diagnostic needs calibration only; formal fit needs formal_ready + approved diagnostics + corpus unchanged. Guard charges parent envelope through append-only continuation ledger and locks; no use of earlier unused reservation; repeat phase forbidden. Review waits stop/release GPU. Actual tokens saved only after native fit. Root supplies model.py,worker.py,guard.py,entry.py.
