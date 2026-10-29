# Cross-match video-to-player experiment handoff

**Status: experimental prototype; release bridge remains blocked.** This run connects real video detections and local tracks to experimental pitch positions and the unchanged tracking backbone. It does not establish globally reliable video-to-pitch observations, tactical accuracy, or unseen-match backbone generalization. The original frozen release and its historical acceptance status are unchanged.

## What ran

- **Sources:** 117092 first half (3840×1906, 25 fps) and 118576 second half (4096×1080, 25 fps). Their source-video, annotation, calibration, detector and release-config hashes are locked in `outputs/video_player_cross_match_20261028_130610/protocol.json`.
- **Coverage:** six preselected, non-overlapping 120-second segments at 300–420, 1200–1320 and 2400–2520 source seconds per match; 600 samples per segment at 5 Hz. The final set contains 3,600 source samples and 12 frozen prediction streams (two detector routes × six segments). The adapter/backbone replays cover all three 120-second 118576 segments plus the 120-second 117092 final0 segment.
- **Coordinates and time:** detector tile offsets are restored to native raw panorama footpoints. The source-specific K,D,Knew fisheye rectification precedes the fixed-ROI rectified-image-to-pitch projection; the output is top-left pitch metres, converted by the existing adapter to its centered convention. There is no global crop/resize of the source coordinate frame. Samples use native frame indices derived from the locked source timestamps at 5 Hz. Predictions outside the predetermined ROI/fitting hull remain missing. All projected positions remain `UNVALIDATED/EXPERIMENTAL`.
- **Frozen backbone:** release configuration SHA-256 `b142e777389d4601f8c09fcaf1b5073e66dedd8aeaec5fabeb82e781ab58e4a9`; G1 SHA-256 `c2732e3b0d99a8fc70afda85ab1da99451d09a09b1731b343200421acc073805`; original v0.5 S4 SHA-256 `a6cbf7a0fb76f0d7a52cd34727070a80df3402a26bfa516bfea2109c4d71c69c`. These checkpoints were hash-checked at load; no action-model training or release-default change occurred.
- **Perception models:** frozen source detector SHA-256 `792db3593ee1d1b26ef4789545527a02fa64c8568010f0116ab6cfe8610dec09`; two-epoch adapted detector SHA-256 `800d92288ec4cce0fc022a7df5796ab73f64aeb77cbc8d75274e25e3096e63da`; COCO-person appearance crop detector SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`; appearance classifier SHA-256 `54f5322a1411af4d1543731713351a516d902ffa67d09051119ca8bee2aa8003`.
- **Environment:** Windows venv, Python 3.14.8, PyTorch 2.11.0+cu128 and Ultralytics 8.4.182. CUDA was unavailable, so these runs used CPU; PyTorch emitted a CUDA initialization warning despite CPU selection.

## Calibration and position evidence

The global calibration gate required at least 50 withheld landmarks, median error ≤2 m and p90 ≤5 m. Both matches had 65 withheld points over three blocks, but failed the error limits:

| Match | Withheld points | Median | p90 | Result |
|---|---:|---:|---:|---|
| 117092 | 65 | 8.2705 m | 20.3042 m | FAIL |
| 118576 | 65 | 8.1435 m | 16.2095 m | FAIL |

Fixed-ROI setup projectors do not repair that global failure. Their independent validations were too small for acceptance:

| Match | Setup landmarks | Fit / validation | Local validation median / p90 | In fitting hull | Result |
|---|---:|---:|---:|---:|---|
| 117092 | 22 | 16 / 6 | 3.399 / 6.361 m | 1/6 | FAIL: fewer than 50 points and both error limits exceeded |
| 118576 | 21 | 15 / 6 | 1.663 / 3.642 m | 2/6 | FAIL: 6 rather than 50 validation points; 4/6 outside the fitting hull |

The 118576 in-hull subset has median 1.418 m and p90 1.830 m, but it contains only two validation points and is not a certification result. Earlier local ROI homography attempts also failed: 117092 used 13 fit/6 withheld correspondences; 118576 used 14/7.

Per-landmark prediction-minus-truth residual vectors and pitch-plane arrow overlays were exported for the 65 withheld points per source. Mean residual vectors by spatial block (metres) were:

| Match | Block 0 | Block 1 | Block 2 | Withheld points outside the fit hull |
|---|---:|---:|---:|---|
| 117092 | (+10.04, +10.02) | (+0.11, −5.46) | (+2.85, +4.56) | 100%, 77.3%, 100% |
| 118576 | (+6.92, +7.23) | (+0.75, −3.33) | (−7.58, +7.26) | 100%, 81.8%, 100% |

These direction changes do not support a single common translation; the opposing horizontal residuals on 118576 and large block-to-block changes indicate a spatially varying mapping error under substantial extrapolation. This sparse evidence cannot distinguish scale/rotation or coordinate-convention mismatch from spatial distortion, so no scalar offset or post-hoc correction was applied. The exact withheld vectors, per-block statistics and fit-hull flags are in the [117092 diagnostics](../outputs/video_player_cross_match_20261028_130610/calibration_diagnostics/landmark_validation_117092_diagnostics.json) and [118576 diagnostics](../outputs/video_player_cross_match_20261028_130610/calibration_diagnostics/landmark_validation_118576_diagnostics.json); see the [117092 overlay](../outputs/video_player_cross_match_20261028_130610/calibration_diagnostics/landmark_validation_117092_diagnostics_pitch_residual_overlay.png) and [118576 overlay](../outputs/video_player_cross_match_20261028_130610/calibration_diagnostics/landmark_validation_118576_diagnostics_pitch_residual_overlay.png). Arrow lengths are scaled at 8 pixels/metre; blue dots are withheld ground-truth landmarks.

Pixel-matched detections inside the fixed ROI and fitting hull provide only a conditional diagnostic: per-interval median/p90 errors were 1.39–1.54/2.72–3.12 m for adapted 117092, 1.14–1.72/2.56–3.30 m for frozen 118576, and 1.08–1.70/2.38–3.11 m for adapted 118576. These are not global validation or evidence that unsupported regions work.

## Detection and downstream execution

Pixel detection scores use bottom-center footpoint matching at the frozen 40-pixel tolerance, over 39,600 annotated player instances per match. Labels were opened only after all 12 prediction files passed the integrity gate.

| Match / detector | Detections | Matches | Precision | Recall |
|---|---:|---:|---:|---:|
| 117092 frozen | 8 | 4 | 50.00% | 0.01% |
| 117092 adapted | 4,079 | 4,030 | 98.80% | 10.18% |
| 118576 frozen | 37,130 | 34,040 | 91.68% | 85.96% |
| 118576 adapted | 30,185 | 29,415 | 97.45% | 74.28% |

Every prediction, score and replay in the original six-interval E evaluation used the first adapted detector, trained for two of 12 requested epochs; its D validation mAP50 was 0.15264 and mAP50–95 was 0.0505. It improved precision but reduced 118576 recall and remains an incomplete experimental candidate.

A separate two-epoch continuation from that candidate's best checkpoint finished without changing any frozen or previously scored output. Its provenance is `outputs/video_player_cross_match_20261028_130610/detector_adaptation_continued_20261028/training_provenance.json`; best-checkpoint SHA-256 is `e68b373f2bd9184fb826038189c22ff16d0dd5877d89759115411908b64116a4`. On the same D validation interval, mAP50 rose to 0.22288 and mAP50–95 to 0.07822. This is a D-validation improvement, not independent or E evidence. The continued checkpoint was used only for one additional 120-second D-development integration replay; E labels were not scored. Neither detector candidate replaces the frozen default.

The autonomous appearance classifier is also experimental: an 11-neighbor HSV/Lab appearance model, trained from 1,314 track-downsampled samples and validated on 655 temporally separate samples from the same 117093 match. Validation accuracy was 91.76% for team and 99.24% for role at 0.5 confidence. These same-match results do not establish cross-match attribute accuracy.

Tracking remains fragmented: across the three 118576 intervals, the frozen/adapted routes reported 2,868/1,896 identity switches and adapted median track lengths of 30–43 samples. On 117092, adapted median track length was 5–7 samples with 54 adjacent matched-identity switches. These are diagnostics, not identity-accuracy certification.

Across the three 118576 adapted segments, 30,185 detections yielded 9,813 experimental positions and 9,764 confidence-qualified team/role estimates. The remaining 20,372 unprojected detections and 49 projectable detections without confidence-qualified attributes retained unknown team/role. Attributes were available for 9,764/9,813 position-supported detections (99.5%). These are availability counts, not cross-match attribute-accuracy measurements.

| Route / coverage | G1 forwards | S4 scheduled / neural forwards | Notes |
|---|---:|---:|---|
| 118576 adapted, autonomous, final0–2 | 614 | 30 / 6 | Neural forwards by segment: 0/10, 3/10, 3/10 |
| 118576 adapted, attributes-assisted, final0–2 | 556 | 30 / 7 | 0/10, 4/10, 3/10; assistance supplies attributes only |
| 117092 adapted, autonomous, final0 | 4 | 10 / 0 | Sparse usable actor observations |
| 118576 frozen, autonomous, final0 | 154 | 10 / 0 | Same video-selected actor anchors as adapted comparison |
| 117092 frozen, autonomous, D development | 0 | 10 / 0 | Five detections and five experimental positions; too sparse to continue either video-selected actor |
| 117092 continued candidate, autonomous, D development | 51 | 10 / 2 | Two actors; both supported queries returned DRIVE; same D validation domain |

Most scheduled S4 queries were unsupported because a causal 11-sample actor history was unavailable. The five query times per actor were supplied at 20, 40, 60, 80 and 100 seconds; they were not detected automatically. G1 and S4 receive the observation-history input independently: G1 hidden estimates do **not** feed S4. Calls and scores are execution diagnostics, not action accuracy.

The paired D-development stream contains 600 continuous samples. The continued candidate produced 2,634 detections, 1,888 experimental pitch positions and 746 unprojected detections; team/role were estimated for 1,878 detections and remained unknown for 756. The frozen detector produced five detections and five experimental positions, with no usable actor track for G1/S4. The continued candidate produced two video-selected actors; G1 made 51 forwards and S4 made 2 neural forwards from 10 supplied queries, with the other 8 unsupported for lack of causal history. Both supported outputs were DRIVE with uncalibrated scores. This is the same D validation domain, not an independent comparison, and all positions remain unvalidated.

The separate annotation-assisted diagnostic substitutes evaluation-only team/role attributes without copying player positions or selecting actors. Its outputs are not autonomous estimates. Unknown team/role, unsupported targets and missing positions remain unknown. Both 117092 and 118576 occur in the canonical G1/S4 fitting-match lists, so this does not demonstrate whole-backbone unseen-match generalization.

## Playable evidence

The four original comparison MP4s and the matched D-development pair were fully decoded and hash-checked; each is 60 seconds, 300 frames, 5 fps, 1920×1080:

- Frozen detector, 118576 final0: `outputs/video_player_cross_match_20261028_130610/video_comparisons/frozen_118576_2nd_final0_60s.mp4`
- Adapted detector, 118576 final0: `outputs/video_player_cross_match_20261028_130610/video_comparisons/adapted_118576_2nd_final0_60s.mp4`
- Adapted detector, 118576 final1: `outputs/video_player_cross_match_20261028_130610/video_comparisons/adapted_118576_2nd_final1_60s.mp4`
- Adapted detector, 118576 final2: `outputs/video_player_cross_match_20261028_130610/video_comparisons/adapted_118576_2nd_final2_60s.mp4`
- Frozen detector, 117092 D-development: `outputs/candidate_20261029/video_comparisons/frozen_117092_1st_development_60s.mp4` (SHA-256 `c6b15f853db1f14e5ccb69bdf84e6ce8a3148981770689b92faf2a0489ccdbe8`)
- Continued detector candidate, 117092 D-development: `outputs/candidate_20261029/video_comparisons/adapted_117092_1st_development_60s.mp4` (SHA-256 `34f131acb6eaf8745f4ae0888a47b3f075f6d06c1b4813e89e04cf2b69a7b48d`; same detector-validation domain, not an independent test)

They show video detections/tracks and projected positions together, with explicit experimental/unvalidated labels.

## Reproduction on this machine

This reproduction uses the already frozen detector-prediction streams; it creates new adapter, replay and render outputs without rerunning detector inference or overwriting the scored experiment. Run from the repository root in PowerShell. Use a new `$run` directory for each reproduction.

```powershell
$py = '.venv\Scripts\python.exe'
$protocol = 'outputs\video_player_cross_match_20261028_130610\protocol.json'
$source = 'outputs\video_player_cross_match_20261028_130610'
$run = "outputs\video_player_cross_match_20261028_130610\replay_repro_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path -LiteralPath $run) { throw "Choose a new, empty output directory: $run" }
New-Item -ItemType Directory -Path $run | Out-Null
Copy-Item -LiteralPath "$source\predictions" -Destination $run -Recurse
Copy-Item -LiteralPath "$source\calibration" -Destination $run -Recurse
New-Item -ItemType Directory -Path "$run\detector_adaptation_5hz_run2" | Out-Null
Copy-Item -LiteralPath "$source\detector_adaptation_5hz_run2\training_provenance.json" -Destination "$run\detector_adaptation_5hz_run2"
& $py -m src.backbone.video_player_cross_match --protocol $protocol --output $run validate-final
if ($LASTEXITCODE -ne 0) { throw 'Final prediction integrity validation failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run build --match 118576 --interval final1 --detector-route adapted --device cpu
if ($LASTEXITCODE -ne 0) { throw 'External-observation build failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run replay --match 118576 --interval final1 --detector-route adapted
if ($LASTEXITCODE -ne 0) { throw 'Frozen-backbone replay failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run render --match 118576 --interval final1 --detector-route adapted --duration-seconds 60
if ($LASTEXITCODE -ne 0) { throw 'Video render failed' }
```

This requires the matching local source video, `yolo11n.pt` COCO appearance weights, the local appearance classifier at `outputs\overnight_video_player_20261028\attribute_model\appearance_knn.npz`, the adapted detector checkpoint named in the provenance, the frozen detector, camera setup assets, release G1/S4 checkpoints, Python environment and portable ffmpeg. These are not bundled or claimed to be available on a fresh machine. Annotation-assisted mode additionally requires the exact local annotation file; it is not needed for the autonomous command above.

The bounded trainer can also create a separate continued candidate from the preserved best checkpoint. It does not update the frozen detector or previously scored predictions; optimizer state is restarted:

```powershell
$trainingRun = "outputs\video_player_cross_match_20261028_130610\detector_adaptation_continued_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $py -m src.backbone.video_player_cross_match --protocol $protocol train `
  --dataset-yaml "$source\adaptation_dataset_5hz\dataset.yaml" `
  --training-output $trainingRun `
  --initial-weights "$source\detector_adaptation_5hz_run2\detector_candidate\weights\best.pt" `
  --epochs 2 --device cpu --budget-minutes 160
```

The completed continued candidate can be replayed separately on D development (117092, 900–1020 seconds). Use a short output root to stay below the Windows path-length limit. This command does not score annotations or alter release defaults:

```powershell
$py = '.venv\Scripts\python.exe'
$protocol = 'outputs\video_player_cross_match_20261028_130610\protocol.json'
$source = 'outputs\video_player_cross_match_20261028_130610'
$weights = "$source\detector_adaptation_continued_20261028\detector_candidate\weights\best.pt"
$run = "outputs\candidate_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path -LiteralPath $run) { throw "Choose a new, empty output directory: $run" }
New-Item -ItemType Directory -Path "$run\calibration" | Out-Null
Copy-Item -LiteralPath "$source\calibration\local_projector_117092.json" -Destination "$run\calibration"
& $py -m src.backbone.video_player_cross_match --protocol $protocol --output $run infer --match 117092 --interval development --route adapted --weights $weights --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Candidate inference failed' }
& $py -m src.backbone.video_player_cross_match --protocol $protocol --output $run infer --match 117092 --interval development --route frozen --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Frozen comparison inference failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run build --match 117092 --interval development --detector-route adapted --device cpu
if ($LASTEXITCODE -ne 0) { throw 'External-observation build failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run replay --match 117092 --interval development --detector-route adapted
if ($LASTEXITCODE -ne 0) { throw 'Frozen-backbone replay failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run render --match 117092 --interval development --detector-route adapted --duration-seconds 60
if ($LASTEXITCODE -ne 0) { throw 'Video render failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run build --match 117092 --interval development --detector-route frozen --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Frozen comparison observation build failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run replay --match 117092 --interval development --detector-route frozen
if ($LASTEXITCODE -ne 0) { throw 'Frozen comparison replay failed' }
& $py -m src.backbone.video_player_cross_match_replay --protocol $protocol --output $run render --match 117092 --interval development --detector-route frozen --duration-seconds 60
if ($LASTEXITCODE -ne 0) { throw 'Frozen comparison render failed' }
```

For this candidate, detector inference took 374 seconds on CPU. The output is a same-D-domain integration check only. No E annotations were read or scored.

Focused regression command:

```powershell
& '.venv\Scripts\python.exe' -m pytest `
  tests\test_video_player_cross_match.py `
  tests\test_video_player_cross_match_replay.py `
  tests\test_regional_pitch_mapping.py `
  tests\test_video_player_overnight.py `
  tests\test_video_player_replay.py `
  tests\test_video_pitch_experimental.py
```

## Limitations and next work

- The global video-to-pitch bridge is blocked; local projectors are sparse, setup-assisted experiments, not certified observations.
- Team and role are estimated from appearance and often unknown; the optional assisted diagnostic uses annotation truth for those attributes only.
- Visibility is simulated, not measured gaze. G1/S4 receive no actual gaze observation.
- S4 requires supplied actor/event-query times and a causal 11-sample supported history.
- The seven S4 classes do not cover every football action. Ball/possession and unsupported targets remain unknown.
- Model scores are not tactical value or best-action recommendations. No tactical accuracy was evaluated.
- The original two-epoch detector candidate is incomplete. Its continued candidate improved only on the same D validation data and remains source-dependent; keep the frozen detector and original v0.5 S4 as defaults.
- Streaming replay metrics and event-tensor metrics use different protocols and must not be compared as if they were one evaluation.
- Prioritize a better-supported local video-to-pitch observation region and repeatable attribute availability; do not broaden the claim from these limited validations.

## Evidence and preservation

The run log and exact stage evidence are in `outputs/video_player_cross_match_20261028_130610/stage_log.json`; protocol/source hashes are in the adjacent `protocol.json`. The 12 predictions, their sidecars, calibration residuals, score denominators, detector provenance, replay JSON and videos remain in the experiment output tree. No human annotation was edited, no GNN work was resumed, and no frozen checkpoint/configuration was changed.

Final focused verification: 22 tests passed; Pylance reported no errors in the affected Python modules; all six comparison videos decoded to 300 frames at 5 fps and matched their sidecar SHA-256 values. The frozen release config, G1/S4 checkpoints, source detector, selected annotations, and calibration assets matched their locked hashes.
