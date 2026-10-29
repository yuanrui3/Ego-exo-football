# Overnight Video-to-Player Continuation Handoff

## Status

The unfinished continuation authorized by “please finish the rest” is complete. The original ten-hour deadline and its overrun remain recorded in the stage log; this continuation does not change that history. The tracking release, its canonical configuration, G1 and original v0.5 S4 checkpoints, and prior acceptance records were not modified. No commit, push, annotation edit, or benchmark/training-corpus admission was made.

The output is an **experimental, locally assisted video-to-player prototype**, not a globally validated bridge or an accuracy claim about actions. The input is the local 117093 first-half panorama (4096×1080, 25 fps); output samples are continuous at 5 Hz. The detector is fine-tuned for compact ground-contact boxes. Attributes come from a separate appearance classifier on matched pretrained COCO person boxes. Pitch mapping is restricted to the fixed, setup-assisted optical-centre region. Held-out annotations were loaded only after video inference for evaluation and for a separate, clearly identified attribute-assisted diagnostic.

## Held-out evidence

| Interval | Video detections (precision / recall) | Within fixed projection support (precision / recall vs all players) | Position error on supported matches (median / p90) | Team estimate accuracy | Video-estimated G1 / S4 calls |
|---|---:|---:|---:|---:|---:|
| 1740–1800 s | 6,175 boxes; 5,920/6,600 matched (95.87% / 89.70%) | 3,877; 3,723 matched (96.03% / 56.41%) | 0.969 / 2.471 m (3,723 pairs) | 88.99% at inference threshold 0.5 (3,723 pairs) | 166 / 1 |
| 1800–1860 s | 5,697 boxes; 5,462/6,592 matched (95.88% / 82.86%) | 2,407; 2,309 matched (95.93% / 35.03%) | 0.795 / 1.558 m (2,309 pairs) | 94.89% at inference threshold 0.5 (2,309 pairs) | 140 / 1 |

Supported-pitch coverage is limited and differs substantially by segment; the second segment’s lower coverage is not hidden by its lower position error. Position comparisons use annotations only as post-inference evaluation. The maximum position errors were 14.78 m and 11.79 m, respectively.

The local geometric tracker reached median identity purity 98.17% / 97.09%, but its 10th-percentile purity was only 59.22% / 53.36%; tracks are not certified identities. Attribute validation is temporal but same-match: identities can recur across fitting and validation intervals. On the first held-out segment, goalkeeper-role accuracy was 99.52% over 3,723 pairs, but only 30 were annotated goalkeepers; at the inference threshold, goalkeeper precision was 62.5%. On the second, the supported evaluation set contained **zero** annotated goalkeepers; the model predicted nine goalkeeper labels and none matched. Its 99.61% overall role accuracy is therefore not evidence of goalkeeper recognition. Missing appearance matches or below-threshold estimates abstain; the assisted route is not autonomous.

The existing external-observation adapter was exercised end-to-end. Annotation-assisted G1 calls were 239 and 179, with one supplied-time S4 query each. S4 returned DRIVE and PASS, respectively, but neither query has an action label or outcome. These are forward-call demonstrations, not action-detection or tactical-quality evaluations. The output files retain synthetic visibility, not measured gaze; ball and possession remain unknown. Seven action classes are incomplete, and scores are uncalibrated classifier outputs—not tactical value or best-action recommendations. Distillation did not improve the prior experiment. Streaming and event-tensor metrics use different protocols.

## Videos and reports

- [1740–1800 s before/after video](../outputs/overnight_video_player_20261028/replays/before_after_1740.mp4)
- [1800–1860 s before/after video](../outputs/overnight_video_player_20261028/replays/before_after_1800.mp4)
- [1740–1800 s measurements](../outputs/overnight_video_player_20261028/replays/replay_result_1740.json)
- [1800–1860 s measurements](../outputs/overnight_video_player_20261028/replays/replay_result_1800.json)
- [Per-route adapter observations and model-call logs](../outputs/overnight_video_player_20261028/replays/)
- [Training and continuation evidence log](../outputs/overnight_video_player_20261028/stage_log.json)

Both MP4s decode as 300-frame, 5 fps, 1920×1080 H.264 videos; the first frame was read successfully for each.

## Reproduction

From the repository root, with the original local footage, calibration/annotation sources, pretrained `yolo11n.pt`, trained experimental detector and attribute model, frozen backbone checkpoints, Python environment, and bundled FFmpeg available:

```powershell
.\.venv\Scripts\python.exe -m src.backbone.video_player_replay --start-s 1740 --duration-s 60 --output outputs\overnight_video_player_20261028\replays_reproduced
.\.venv\Scripts\python.exe -m src.backbone.video_player_replay --start-s 1800 --duration-s 60 --output outputs\overnight_video_player_20261028\replays_reproduced
```

These commands use checked-in source plus **local-only data and weights**; they do not establish fresh-machine reproducibility. Use a new output directory to avoid replacing the preserved evidence. Do not copy evaluation annotations into the autonomous input path.

Focused regression command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_video_player_overnight.py tests\test_video_player_replay.py tests\test_video_pitch_experimental.py
```

## Training provenance and limits

The separate YOLO11n footpoint model started from official COCO `yolo11n.pt` (30 epochs, fitting data 1500–1620 s; 600 frames, 2,400 tiles, 13,200 labels) and used 1620–1680 s for temporal validation. Best-weight SHA-256: `792db3593ee1d1b26ef4789545527a02fa64c8568010f0116ab6cfe8610dec09`. Detector and attribute-model details are in the adjacent provenance JSON files. The appearance KNN used 1,314 fitting samples and 655 temporal validation samples; its role/team thresholds were selected on that same-match validation split.

The global video-to-pitch gate remains failed. This demonstration uses a fixed local region assisted by released camera setup calibration and fitting landmarks; it does not certify other regions or footage. Simulated visibility is not gaze. G1 predictions are not inputs to S4. S4 requires a supplied actor/time query and is not a continuous action detector. Unknown team/role inputs can block the learned models. No tactical accuracy or outcome improvement was measured.
