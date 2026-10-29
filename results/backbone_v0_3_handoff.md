# Backbone v0.3 — Action Inference and Supervision Handoff

## Status

**Connected:** the trained S4 partial-observation action classifier runs in the existing replay through `ReplayEngine(..., action_model=session)`. It is player-facing with respect to the replay's synthetic partial-observation contract, not raw video perception. Predictions are event-conditioned demonstrations with an explicitly supplied actor and query time; this is neither a continuous detector nor a best-action recommender.

The isolated output bundle is [outputs/backbone_v0_3_20261020](../outputs/backbone_v0_3_20261020/). It contains the integration manifest, seven event-conditioned student inputs, seven separate action targets, the mixed-action replay, and the acceptance record. The earlier 180-record v0.2 export was not modified.

## Checkpoint and provenance

- Checkpoint: `outputs/partial_observation_20261017/students/final/S4/fov120/seed42/final.pt`
- SHA256: `8a73b1980cebd650d57ded29a7f168bb2d9ddd0d7b8c82c1dcac495c1d53f4c2`
- Variant S4, seed 42, 96-wide, two layers/four heads, 17 fixed epochs; 221,975 parameters.
- Fitting matches recorded in the checkpoint config: `117092, 117093, 118576, 118578, 128057, 132831, 132877`.
- The experiment report identifies `118577` and `128058` as held-out validation matches. The selected checkpoint config has no separate dev-only match list; these held-out matches were not used for the v0.3 fitting export.
- The v0.3 export consumes only the current v0.2 `training_index.json` shards: `117092_1st` and `117093_1st`, both in the action checkpoint fitting set. All new labels and replays are demonstrations on fitting matches, not held-out evaluation.

The original experiment report gives an S4 120-degree held-out action macro-F1 mean of 0.4893 across seeds. That is historical experiment-level evaluation, not a v0.3 result; this sprint did not retrain or benchmark the model.

## Taxonomy and timing

The seven trained classes retain their original mapping:

| Model class | Existing recode source label(s) |
|---|---|
| `PASS` | `Pass` |
| `HIGH_PASS` | `High Pass` |
| `DRIVE` | `Drive` |
| `CROSS` | `Cross` |
| `SHOT` | `Shot` |
| `HEADER` | `Header` |
| `DEFENSIVE_INTERVENTION` | `Ball Player Block`, `Player Successful Tackle` |

The BAS source label is preserved verbatim in `optional_targets.jsonl` (for example, `HIGH PASS`); the table lists the corresponding recode-file labels. Labels are joined by stable event ID, including its annotation index into BAS `actions`; the join verifies period, BAS frame, source label, and recode. It is not a row-order join. The causal query anchor is `floor(position_ms / 40)`, as in the original training builder. BAS stable-ID frame and query anchor can differ by one; both are recorded. The 11 input samples are spaced five frames apart and end at that anchor, with no post-event sample.

The classifier was trained at annotated event times, and the underlying report marks physical-contact synchronization as unverified. Supplying an actor and query time supports event-conditioned demonstrations only. The v0.2 tracking replay can provide the compatible synthetic pitch-coordinate stream; pixel-only video remains unsupported until pitch projection, synchronization, identities, and team assignments are accepted.

Labels are written only to `optional_targets.jsonl`. Student inputs contain no action label. Arbitrary v0.2 windows remain unlabeled; missing annotation is not a negative label or `NO_ACTION`. The existing indexed shards' optional-target hashes are recorded and verified unchanged in `acceptance.json`.

## Reproduced input contract

- Inputs: `WorldInput.x` shaped `[11, 22, 10]` and entity mask `[11, 22]`; the selected actor is present and visible at every history sample. The actor need not possess the ball.
- Sampling: 25 Hz source, every five frames (0.2 s), from −2.0 s through the query time.
- Feature order: actor-relative `dx/52.5m`, `dy/34m`; `vx/12m/s`, `vy/12m/s` clipped to `[-1,1]`; visible; ever-seen; age; team relation `+1/-1`; goalkeeper; selected actor.
- Age: v0.2 encodes `min(age, 8s)/8`; the action adapter multiplies by `8/5` and clips to `[0,1]`, matching S4's `min(age, 5s)/5`. The 2-second model history keeps ages within the shared range.
- Visibility: synthetic 120-degree FOV and 45 m range, causal motion-heading fallback, with artificial dropout disabled as in the training cache. The deterministic event seed can produce a different random fallback draw than the original cache seed; the causal algorithm and distribution are the same. This is not gaze or video-derived visibility.
- Memory/identity: causal first-sight memory, visibility from feature 5's predecessor (`x[...,4]`), entity mask matching the ever-seen feature (`x[...,5]`). S4 has no slot embeddings, so v0.2 first-sight slots are compatible with its per-example anonymous slot permutation.
- To match the original builder's current-roster contract, event samples are excluded if a track seen earlier in the two-second window is absent at query time or if observed team/role metadata changes in that window. All seven selected examples passed these checks.
- The action adapter sees only the relative partial-observation tensor and masks. It does not receive BAS labels, global hidden coordinates, global player ordering, actor possession, restart, or outcome.

The adapter returns `football.action-output.0.1` with a predicted class, `confidence: null`, `support_status: partial`, and all target/location/direction/timing fields unset. It also retains the seven softmax scores as **uncalibrated classifier scores**, not tactical value or a calibrated uncertainty estimate. Ball position and possession remain unknown. Restart context and outcome are separate. Passing geometry is an optional independent module; it is not used to rank action classes. No compatible exo-reference action checkpoint is configured. All actions outside the trained taxonomy are explicitly unsupported.

## Demonstration outputs and errors

Selection took the first stable event per class with sufficient input coverage; prediction correctness was not a selection criterion. The bundle includes every raw score vector and keeps annotations in the evaluation-only section.

| Event | Annotation (evaluation-only) | Player-facing class | Match |
|---|---|---|---|
| `117092_p1_f135_2` | PASS | PASS | Yes |
| `117092_p1_f86_1` | HIGH_PASS | PASS | **No** |
| `117092_p1_f956_8` | DRIVE | DRIVE | Yes |
| `117092_p1_f1318_15` | CROSS | DRIVE | **No** |
| `117092_p1_f8014_149` | SHOT | DRIVE | **No** |
| `117092_p1_f32972_543` | HEADER | HEADER | Yes |
| `117092_p1_f982_9` | DEFENSIVE_INTERVENTION | DEFENSIVE_INTERVENTION | Yes |

These seven examples are fitting-match demonstrations and must not be interpreted as model accuracy. The visible errors include HIGH_PASS→PASS, CROSS→DRIVE, and SHOT→DRIVE. The replay stores prediction scores, annotations, unavailable exo reference, and optional passing geometry separately.

## Existing functions and reproduction

- Load once: `src.backbone.actions.ActionModelSession(checkpoint, device="cpu")`
- Predict: `ActionModelSession.predict(WorldInput)`
- Replay: `src.backbone.pipeline.ReplayEngine(..., action_model=session).step(observation)`
- Original forward for parity: `PartialObservationStudent(x, entity_mask, visible, memory_mode="memory")`, matching `train/run_partial_observation_experiments.py`.
- Numerical parity was checked on identical tensors against the original forward; the focused test compares all seven class scores.

From a PowerShell prompt at the repository root, with a fresh output destination:

```powershell
Set-Location 'C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2'
$env:PYTHONPATH = (Get-Location).Path
& 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m src.backbone.v0_3 --output outputs\backbone_v0_3_reproduced
& 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m pytest tests\test_backbone_v0_3.py tests\test_backbone.py tests\test_backbone_ingestion.py -q
```

The builder refuses to overwrite an existing output directory. No training, human annotation changes, commits, or pushes were performed.
