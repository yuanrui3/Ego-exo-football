# Assisted local video-to-pitch prototype

**Status: position-only experimental prototype demonstrated; global calibration remains failed.**
This is not a validated player-position source and is not admitted to training or benchmark data.
The frozen tracking release, its defaults, and the historical acceptance records remain unchanged.

## What ran

The source is the existing 117093 first-half panorama (4096 x 1080, 25 fps) and its existing
5 Hz detector/tracklet output. Detection boxes are in native panorama pixels; the clip has no
source-frame crop/resize, and its stored observations are on native frame indices. The demo
checks the video, tracklet, and clip-manifest hashes and verifies the 0.2-second frame clock.
The detector's internal tiles were scaled and their offsets restored to native coordinates
before boxes were stored.
It uses the bbox bottom-middle as a footpoint and does not use player-position annotations.

The assisted region is fixed at +/-650 native pixels around the released camera's optical
principal point (x=2015.08 px), before looking at validation errors. The keypoint split has
16 fitting landmarks and 6 withheld validation landmarks, of which 4 lie inside the fitting
convex hull. The local path undistorts native footpoints and landmarks with the released
fisheye K, D, and Knew, fits pitch-to-rectified-image coordinates on fitting landmarks, and
inverts that homography for projection. Positions outside the optical ROI, fitting hull, or
pitch bounds are withheld. This is setup-assisted and remains explicitly experimental.

## Calibration diagnosis

The original global gates remain failed and were not edited:

| Match | Original withheld points | Original median / p90 |
|---|---:|---:|
| 117093 | 65 | 6.79 m / 15.99 m |
| 118578 | 65 | 5.88 m / 14.38 m |

`global_landmark_residuals.csv` contains the per-point x/y residual vectors, norms, support
membership, and fitting-only translation/affine correction estimates for all three spatial
folds. The pitch-space overlays draw each withheld residual vector and distinguish points
inside versus outside the fit hull. The errors are not explained by a deployable common
translation: the extrapolation-heavy folds retain large held-out errors after fitting-only
correction, and correction effects vary by fold. Across fitting-only affine fits, rotations range
from -1.33 to +1.12 degrees and principal scales from 0.970 to 1.036; the withheld extrapolation
folds still have median errors of 9.17–11.64 m and p90 errors of 11.79–17.64 m after correction.
Thus neither a shared 6 m offset nor one uniform scale/rotation correction explains the failure.
The low errors occur primarily where withheld points interpolate inside the fit hull, while
outside-hull residuals vary by location. This is consistent with insufficient spatial support
and coordinate/distortion effects, not proof of any single cause.

The dataset-author homography route fits pitch-to-image and applies its inverse to image
points. On the fixed local split, that route on native pixels gives 4.49 m median / 9.27 m
p90 over 6 withheld landmarks, with only 1 point inside the native fit hull. The separate
fisheye-rectified variant used by this local demo gives 1.29 m / 2.69 m over all 6, with
1.26 m / 1.89 m on the 4 withheld points inside the rectified fit hull. These small landmark
counts do not establish player-position accuracy or global certification. Independently
fitting image-to-pitch rather than inverting pitch-to-image differs from the native author
route by 1.23 m median / 2.76 m maximum on the same withheld points; both paths and their
per-landmark outputs are recorded in `author_projection_comparison.json`.

The full per-point residuals, fit-only correction parameters, and overlays are in each
replay folder's `global_residual_diagnosis.json`, `global_landmark_residuals.csv`, and
`*_fold_*_pitch_residuals.png`.

## Continuous experimental replays

Both use the same calibration, source video, detector output, and code path:

| Segment | Samples | Pixel detections | Projected inside local support | Tracklets |
|---|---:|---:|---:|---:|
| 1500–1560 s | 300 | 1,487 | 403 (27.1%) | 91 |
| 1560–1620 s | 300 | 1,320 | 347 (26.3%) | 46 |

The rendered H.264 files are 1920 x 1080, 5 fps, 60 seconds, 300 frames each. The HTML
viewers show a representative frame plus the video controls; the MP4s and posters were
independently probed/extracted and visually inspected. Tracklet IDs are local, adjacent-sample
candidates, not verified player identities.

- [1500–1560 s viewer](../outputs/assisted_local_video_demo_20261027_run3/assisted_local_demo.html)
- [1500–1560 s video](../outputs/assisted_local_video_demo_20261027_run3/assisted_local_demo.mp4)
- [1560–1620 s viewer](../outputs/assisted_local_video_demo_20261027_run4/assisted_local_demo.html)
- [1560–1620 s video](../outputs/assisted_local_video_demo_20261027_run4/assisted_local_demo.mp4)

## Frozen model integration result

For each segment, the canonical G1 and original v0.5 S4 checkpoints were loaded once and
their pins verified. G1 made **0 neural forward calls**; its input contract was unavailable
on all 300 samples because team and goalkeeper role are unknown. S4 received one diagnostic
query at a supplied actor/time with an 11-sample position history, but the unknown attributes
prevented a causal model tensor. S4 made **0 neural forward calls** and returned `unsupported`
(“The causal 11-sample player observation is not ready”). The underlying memory contract
identifies unknown team/role as the blocker. No action annotation or player attribute was
invented. Ball and possession remain unknown.

G1/S4 query evidence is in `experimental_g1_s4_status.jsonl`; it contains no successful
learned prediction. Optional passing geometry is disabled. None of these outputs enters the
training or benchmark corpus.

## Reproduction and checks

These commands assume this workspace already has the source video, fitting-only detector
tracklets, released calibration files, frozen checkpoints, portable ffmpeg, and configured
`.venv`. Each output directory must be new; the runner refuses to overwrite a non-empty one.

```powershell
.\.venv\Scripts\python.exe -m src.backbone.video_pitch_experimental `
  --start-s 1500 --duration-s 60 `
  --output outputs\assisted_local_video_demo_reproduction_1500

.\.venv\Scripts\python.exe -m src.backbone.video_pitch_experimental `
  --start-s 1560 --duration-s 60 `
  --output outputs\assisted_local_video_demo_reproduction_1560

.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_video_pitch_experimental.py `
  tests\test_regional_pitch_mapping.py `
  tests\test_backbone_v06_video.py
```

Final targeted regression result: **10 passed**. Pylance diagnostics and Python syntax checks
are clean. Both MP4s probe as H.264, 1920 x 1080, 5 fps, 300 frames, 60 seconds. The canonical
baseline configuration, G1 checkpoint, and S4 checkpoint hashes match the release manifest.
The tested local environment used Python 3.14.8 and CPU Torch; this is not a fresh-machine
reproducibility claim.

## Still blocked

- The original global video-to-pitch gate remains failed; no correction is promoted globally.
- Four in-hull validation landmarks are insufficient to validate real player positions.
- Track identity, teams, goalkeeper roles, gaze, ball, and possession are not measured here.
- G1 and S4 learned forward passes remain blocked; this is position-only, not action inference.
- The author’s per-frame camera JSON parameters and per-frame homographies are absent from
  the local release, so its full camera-parameter projection path cannot be reproduced.
- Scores, if a future supported action query becomes possible, are not tactical value or a
  best-action recommendation.
