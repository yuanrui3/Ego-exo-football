# Backbone v0.5 handoff

**Status:** complete with documented limitations. **No commit or push was made.**

## What runs

The standard batch replay now loads the trained S4 partial-observation student through `ActionModelSession`, once per batch, and calls it only at explicitly supplied event-conditioned actor/query times. It is a **player-facing prediction over synthetic, causal player-view observations**, not an exo-only classifier. The seven trained outputs, in checkpoint order, are `PASS`, `HIGH_PASS`, `DRIVE`, `CROSS`, `SHOT`, `HEADER`, and `DEFENSIVE_INTERVENTION`. The action-output contract remains `football.action-output.0.1`.

The 11-step input ends at the causal query frame and spans 2 seconds at 0.2-second intervals. It contains `x`, entity-mask and visibility data, with shape `[11, 22, 10]`. Inputs are built sequentially using causal heading and first-sight memory; the visibility simulation uses a 120-degree field of view and 45-metre range, with the actor visible throughout. Track identities are source-local and anonymous. The adapter receives neither global hidden coordinates nor global player ordering nor the true label. Ball state and possession remain unknown unless independently evidenced. The selected actor is not assumed to have the ball.

Student NPZ age uses `min(age, 5 seconds) / 5`; the shared causal memory's `min(age, 8 seconds) / 8` convention is converted at the model boundary. The G1 world baseline remains a separate exo/scene reference (checkpoint SHA-256 `c2732e3b0d99a8fc70afda85ab1da99451d09a09b1731b343200421acc073805`); its hidden-coordinate estimates do not enter the S4 action adapter.

One deterministic 12-segment retrospective batch covers all seven available annotated classes. It includes non-PASS classes, uses label/input availability and source coverage rather than prediction correctness, and records predictions, annotations and optional passing geometry separately. For example, at query frame 77 in match 118577, the model predicted `PASS` with score 0.6943 while the evaluation-only BAS annotation was `DRIVE`. The output explicitly says the softmax score is uncalibrated; it is neither confidence nor tactical value. Receiver ranking and passing-lane assessment remain optional passing modules, not action-backbone requirements.

The checkpoint is `outputs/backbone_v0_5_20261020/training/model/final.pt` (SHA-256 `a6cbf7a0fb76f0d7a52cd34727070a80df3402a26bfa516bfea2109c4d71c69c`). It was fitted on matches `117092`, `117093`, `118576`, `118578`, `128057`, and `132831`; development selection used `132877`. The checkpoint was selected at epoch 4; the bounded run stopped after 7 epochs. Its initialization was S4/120-degree, width 96, from `outputs/partial_observation_20261017/students/development/S4/width96/best.pt` (SHA-256 `f308b679ad0104ebde27b648c6b25dcf887ab3c2697e723fa2e93a3f0b43d5c9`). Fitting included 12,038 examples.

## Timing, evaluation and data

`timing_coverage.json` covers 19,311 raw events: all 19,311 have supported source-time resolution; 17,992 are eligible labeled examples (12,038 fitting, 2,158 development, 3,796 retrospective evaluation). There are also 4,456 fixed-interval unlabeled windows with no action target. Missing annotations are not converted into negative or `NO_ACTION` labels.

The query frame is `floor(position_ms * native_fps / 1000)`. A stable BAS frame is an independent nearest-rounded consistency check, never the query-frame authority. This accepts the observed two-frame stable-ID offset relative to the legacy floor rule for match 128057 (942 first-half and 990 second-half events) without using those later IDs as observations. Physical contact timing is not certified.

The retrospective diagnostic over matches 118577 and 128058 achieved accuracy 0.6088 and macro-F1 0.5188 on 3,796 examples. Both matches appeared in earlier research; this is not a newly unseen test result. Reloaded-adapter parity against direct checkpoint inference passed with a maximum absolute probability difference of `4.77e-7`.

The current preflight passed. The local-video workload decoded/inferred 900 frames across three clips, producing 5,671 pixel detections and 855 unverified image-space tracklet candidates. Pitch projection, player identity/team/role verification, and action inference from this video remain unsupported.

## Resume-state defect fixed

A safe resume probe exposed that `torch.load(..., map_location=cuda)` moved saved CPU RNG byte tensors to CUDA before `torch.set_rng_state`. `_restore_rng` now copies CPU and CUDA RNG tensors back to CPU before restoration. The CUDA probe restored optimizer and RNG state from the existing `last.pt`, ran zero additional epochs (state was at epoch 7 with patience exhausted), and left the original resume checkpoint unchanged. `last.pt` SHA-256 is `dca5d1892c9367b8ac26d538e7dc99cada85530835cedb378ac4c23fb3a13abd`.

## Tests and outputs

- Focused backbone regression suite: **55 passed**.
- Clean source/config/test-only copy of the v0.5 focused tests (no local data or outputs): **6 passed**.
- Pylance diagnostics on changed Python files: **0**.
- Twelve replay jobs completed; a subsequent `--resume` skipped all twelve by hash match.
- Machine-readable status and gate details: [acceptance_record.json](../outputs/backbone_v0_5_20261020/acceptance_record.json).
- Timing coverage: [timing_coverage.json](../outputs/backbone_v0_5_20261020/timing_coverage.json).
- Training and diagnostic evaluation: [training_result.json](../outputs/backbone_v0_5_20261020/training/training_result.json), [retrospective_evaluation.json](../outputs/backbone_v0_5_20261020/training/retrospective_evaluation.json).
- Batch viewer and summary: [viewer.html](../outputs/backbone_v0_5_20261020/batch/viewer.html), [last_executed_summary.json](../outputs/backbone_v0_5_20261020/batch/last_executed_summary.json).
- Video workload: [manifest.json](../outputs/backbone_v0_5_20261020/video/manifest.json).

## PowerShell reproduction

Run from the repository root in PowerShell. These commands use the tested Python 3.14 interpreter. The checkpoint resume probe uses copies in a disposable temp directory: do not aim `--resume` at the canonical training output, because finalization rewrites model/report files even when the epoch loop is already exhausted.

```powershell
$Python = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
Set-Location 'C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2'

& $Python -m src.backbone.preflight_v05 `
  --config configs\backbone_v0_5.json `
  --output outputs\backbone_v0_5_20261020\preflight.json

& $Python -m src.backbone.timing_report_v05 `
  --config configs\backbone_v0_5.json `
  --index outputs\backbone_v0_5_20261020\corpus\training_index.json `
  --output outputs\backbone_v0_5_20261020\timing_coverage.json

$ResumeProbe = Join-Path $env:TEMP ('backbone-v05-resume-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $ResumeProbe | Out-Null
Copy-Item outputs\backbone_v0_5_20261020\training\last.pt (Join-Path $ResumeProbe 'last.pt')
Copy-Item outputs\backbone_v0_5_20261020\training\best.pt (Join-Path $ResumeProbe 'best.pt')
& $Python -m src.backbone.train_action `
  --index outputs\backbone_v0_5_20261020\corpus\training_index.json `
  --config configs\backbone_v0_5_training.json `
  --output $ResumeProbe `
  --resume

& $Python -m src.backbone.train_action `
  --index outputs\backbone_v0_5_20261020\corpus\training_index.json `
  --output outputs\backbone_v0_5_20261020\training `
  --evaluate `
  --checkpoint outputs\backbone_v0_5_20261020\training\model\final.pt

& $Python -m src.backbone.replay_config_v05 `
  --index outputs\backbone_v0_5_20261020\corpus\training_index.json `
  --action-checkpoint outputs\backbone_v0_5_20261020\training\model\final.pt `
  --output outputs\backbone_v0_5_20261020\retrospective_batch.json

& $Python -m src.backbone.batch `
  --config outputs\backbone_v0_5_20261020\retrospective_batch.json `
  --output outputs\backbone_v0_5_20261020\batch `
  --resume

& $Python -m pytest `
  tests\test_backbone.py `
  tests\test_backbone_ingestion.py `
  tests\test_player_centric_observation.py `
  tests\test_backbone_v0_3.py `
  tests\test_backbone_v05.py -q
```

The three-clip local-video workload can be rerun separately with:

```powershell
& $Python -m src.backbone.video_ingest `
  --config configs\backbone_v0_5.json `
  --output outputs\backbone_v0_5_20261020\video_reproduction
```

Do not treat the classifier as a continuous detector or best-action recommender. Untrained action branches, physical contact alignment, calibrated confidence/value, and pitch-projected video player states remain unsupported.
