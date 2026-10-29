# Priority 1: Overnight Video-to-Pitch Observation Bridge

## Status: BLOCKED at metric projection

The local recording was decoded and processed into person detections and
conservative adjacent-frame image-space tracklet candidates. No calibration
passed the frozen validation policy, so no pixel coordinate was converted to
pitch metres, no video-derived observation entered the external adapter, and
neither G1 nor S4 ran on video input. This is a blocked bridge, not a successful
video-to-pitch demo.

The 12-hour window began at first inspection on 2026-10-27 21:28 EDT; its
deadline was 2026-10-28 09:28 EDT. Further processing stopped once the supported
calibration alternatives failed and no valid pitch observation could be
produced. The conditional second-segment step was not run because the first
bridge did not pass.

## Source audit and frozen validation protocol

The selected alternative source is the local `117093` first-half panorama:

| Property | Audited evidence |
|---|---|
| Video | `data/video_review_v1_2/videos/117093/117093_panorama_1st_half.mp4` |
| SHA-256 | `f50dfebc70743439a8c778abe06953885a50be8b47e3c9d3d1ce8d533fe315c8` |
| Encoded image | 4096 x 1080 pixels, 25 fps, 67,625 decoded frames |
| Calibration source | Match-specific released v1.2 field keypoints and `K`, `D`, `Knew`; 65 image-to-pitch correspondences |
| Calibration hashes | Keypoints `bcc874620cfa1bcbd49b33a6132abf51f35525f25cafb677084e8ce3e803a94f`; intrinsics `fe3f14a96495aa6706e6bf716490d599e0818ed0a48e7bf8e4387e7fe985c720` |
| Field keypoint bounds | x=111..3935, y=155..719, within native video bounds |
| Released lens-fit RMS | 8.637 px; release does not establish an independent fit from the withheld landmarks |
| Coordinate direction | Image pixels to pitch coordinates in metres (105 m x 68 m) |
| Distortion variants | Primary attempt used `cv2.fisheye.undistortPoints` with the released `K`, `D`, `Knew`; a separate diagnostic used raw-native panorama pixels |
| Spatial transforms | No source crop/resize transform is declared for calibration landmarks. Detector inference uses overlapping tiles; returned horizontal coordinates are restored with each tile's source-x offset. Resized overlays are display-only. |
| Time | 25 Hz frame-index sampling. Dataset release v1.2 aligns annotation frame k with video frame k. Timestamp-based seeking can see a 1 s stream offset for this half; frame-index access avoids that offset. |
| Assistance | Both field keypoints and camera intrinsics are released calibration inputs. This is annotation-assisted, recording-specific work, not autonomous video/field-evidence calibration. |

The gate was frozen before the new fits and was not changed: at least 50
withheld correspondences across three spatial pitch-length blocks, median
discrepancy <=2 m, p90 <=5 m; report the count, per-block errors, inliers,
support hull and image/pitch coverage. The complete policy is
[`validation_policy.json`](../outputs/video_pitch_bridge_20261027/validation_policy.json).
The raw-native sensitivity policy is
[`validation_policy_raw_native.json`](../outputs/video_pitch_bridge_20261027/validation_policy_raw_native.json).
The independent-source policy is
[`validation_policy_118578.json`](../outputs/video_pitch_bridge_20261027/validation_policy_118578.json).

## Evidence and gate outcomes

| Source / transform | Withheld denominator | Block results (median / p90 metres) | Aggregate median / p90 | Decision |
|---|---:|---|---:|---|
| 117093 1st / fisheye-undistorted | 65 (22/22/21) | 10.43/16.89; 2.70/5.11; 11.00/17.34 | 6.79/15.99 | BLOCKED |
| 117093 1st / raw-native | 65 (22/22/21) | 30.84/40.14; 6.26/15.78; 40.55/50.75 | 26.58/44.60 | BLOCKED |
| 118578 2nd / fisheye-undistorted | 65 (22/22/21) | 11.58/17.55; 2.11/3.15; 9.33/11.96 | 5.88/14.38 | BLOCKED |

For the fisheye attempts, the withheld blocks were mostly outside the fitted
image-space hull: 117093 fractions were 1.00, 0.864, 1.00; 118578 fractions
were 1.00, 0.818, 1.00. The local diagnostic regions did not rescue support:
117093 had 21 points (14 fit, 7 withheld) and failed to fit; 118578 had 20
(14 fit, 6 withheld), with median/p90 123.87/142.40 m and all withheld points
outside the fitting hull. They do not meet the minimum denominator and are not
deployable.

The earlier 117092 whole-panorama attempt also remains rejected (65 withheld,
8.27/20.30 m aggregate). Its separate raw thin-plate-spline diagnostic had only
2 withheld points inside the fit hull and 8.35/10.99 m errors. These historical
artifacts were not overwritten.

Full machine-readable results:

- [`mapping_117093.json`](../outputs/video_pitch_bridge_20261027/mapping_117093.json)
- [`mapping_117093_raw_native.json`](../outputs/video_pitch_bridge_20261027/mapping_117093_raw_native.json)
- [`mapping_118578_2nd.json`](../outputs/video_pitch_bridge_20261027/mapping_118578_2nd.json)
- [`decision_record.json`](../outputs/video_pitch_bridge_20261027/decision_record.json)

## What actually runs from video

The existing local corpus contains 42,000 sampled frames: 36,000 fitting and
6,000 development samples over 120 and 20 source minutes respectively. The
independent corpus validator passed all 280 chunks and rechecked all eight
source hashes. In the 20-minute 117093 first-half excerpt, the stored output
contains 6,000 sampled frames, 22,653 detections, 4.67% empty-detection frames,
3/7 p50/p90 detections per frame, and 2,980 tracklet candidates.

These are real video-derived pixel detections, but the tracklets are explicitly
`UNVERIFIED_IMAGE_SPACE_CANDIDATES`; neither identity accuracy nor metre
accuracy was measured. The corpus contains zero accepted pitch observations.
Thus validated continuous pitch coverage is **0 seconds**, not 20 minutes.

The existing `ExternalObservationSource` requires validated metre coordinates,
declared pitch axes/dimensions, provenance and 0.2-second timing. Since no
mapping passed, it was not given video positions. G1 and the original v0.5 S4
were not run on this video. No reference player coordinate, team, role, or
identity was supplied to inference. Team and goalkeeper role remain unknown.

The canonical baseline remains unchanged: config SHA-256
`b142e777389d4601f8c09fcaf1b5073e66dedd8aeaec5fabeb82e781ab58e4a9`, G1
checkpoint SHA-256
`c2732e3b0d99a8fc70afda85ab1da99451d09a09b1731b343200421acc073805`, and
original v0.5 S4 checkpoint SHA-256
`a6cbf7a0fb76f0d7a52cd34727070a80df3402a26bfa516bfea2109c4d71c69c`.
Historical acceptance remains PARTIAL and the video bridge remains BLOCKED.

## Verification

- Existing video-corpus validator: **PASS**, 42,000 unique sampled frames,
  280 completed chunks, 8 source hashes, no errors.
- Coordinate-transform and video-ingestion tests: **8 passed**.
- Frozen baseline release/external-input regressions: **4 passed**.
- Pylance diagnostics: no errors in the changed calibration utility or test.
- No new detector inference, model training, calibration downloads, annotation
  edits, or changes to the frozen checkpoints/configuration.

The new source utility change makes raw-native and fisheye-undistorted point
transforms explicit while preserving fisheye-undistorted as the default. It
does not export or deploy a mapping.

## Reproduction (Windows PowerShell, repository root)

Use the configured local Python and choose new output paths; existing reports
should be preserved:

```powershell
$Python = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'

# Fisheye-undistorted 117093 calibration validation
& $Python -m src.backbone.regional_pitch_mapping_v06 `
  --metadata outputs\backbone_v0_6_20261021\stage_b\calibration_assisted\video\revision_metadata\verified.json `
  --policy outputs\video_pitch_bridge_20261027\validation_policy.json `
  --video data\video_review_v1_2\videos\117093\117093_panorama_1st_half.mp4 `
  --output outputs\video_pitch_bridge_20261027\mapping_117093_repro.json

# Independent 118578 source condition
& $Python -m src.backbone.regional_pitch_mapping_v06 `
  --metadata outputs\backbone_v0_6_20261021\stage_b\calibration_assisted\video\revision_metadata\verified.json `
  --policy outputs\video_pitch_bridge_20261027\validation_policy_118578.json `
  --video data\video_review_v1_2\videos\118578\118578_panorama_2nd_half.mp4 `
  --output outputs\video_pitch_bridge_20261027\mapping_118578_repro.json

# Revalidate already-completed detection/tracklet corpus; this does not rerun inference
& $Python -m src.backbone.validate_video_ingest_v06 `
  --config outputs\backbone_v0_6_20261021\stage_b\video_config_v06.json `
  --output outputs\backbone_v0_6_20261021\stage_b\video_corpus `
  --root . `
  --report outputs\video_pitch_bridge_20261027\video_corpus_validation_repro.json

# Focused regressions
& $Python -m pytest -q tests\test_regional_pitch_mapping.py tests\test_backbone_v06_video.py tests\test_backbone_release.py
```

## Next prerequisites

Do not project or replay this footage until a camera/region-specific map passes
the same frozen holdout gate. The next useful input is an independently
measured, sufficiently dense set of region-local field correspondences (or
trusted withheld landmarks) paired to the exact encoded panorama and verified
camera transform, plus independently assessed detector-footpoint and
track-identity quality. Any assisted calibration must stay labeled as such.
Only after that gate passes should accepted observations be serialized through
the external adapter and the frozen replay evaluated. Keep all unobserved
team/role attributes unknown.
