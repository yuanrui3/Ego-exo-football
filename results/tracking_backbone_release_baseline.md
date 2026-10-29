# Tracking-based backbone release baseline

**Scoped status: READY.** This is a separately scoped reproducible local
baseline, not a retroactive completion claim. The historical v0.6/completion
acceptance remains **PARTIAL** and the video-to-pitch bridge remains
**BLOCKED**. The GNN queue remains paused. No training, calibration, data
expansion, annotation edits, commits, pushes, merges or tags are part of this
release closeout.

## What runs

The frozen path is:

`verified pitch tracking or validated external metre observations -> synthetic player-view observations -> causal memory/G1 hidden-player estimates and explicit fallbacks -> separate event-conditioned S4 action queries`

**G1 estimates do not feed S4.** Both models receive the causal observation/
memory stream through their existing interfaces, but S4 receives no G1 hidden
coordinates, no global player ordering and no true action labels. The actor and
query time are supplied by the caller. An absent annotation is unknown, not a
negative action. Recorded annotations are joined by stable sample ID/event ID
and annotation index, then kept evaluation-only. Optional passing geometry is
independent and disabled in the canonical demo.

The default models are the existing fitted G1 and original v0.5 S4. Checkpoint
hashes, training/development match provenance, source hashes, contracts,
environment and source inventory are in
[`backbone_release_manifest.json`](../configs/backbone_release_manifest.json);
the executable canonical input is
[`backbone_release_baseline.json`](../configs/backbone_release_baseline.json).
No progress file or expired orchestrator is needed: the release runner reads
the selected G1 integration manifest, its pinned checkpoint, and S4 directly.

The existing batch/corpus/export/training interfaces are retained, along with
the verified target-free 100,000-window lazy index. They are not rebuilt or
used as hidden inputs to the replay. G1 estimates are used only while the
11-step observation contract is met; otherwise the pipeline records explicit
unsupported reasons and applies the configured last-seen fallback. Memory
expires after6s; stale estimates are not presented as current observations.

## Running and verifying

Run from the repository root in PowerShell with the existing configured Python
environment. These example output names are new, immutable directories:

```powershell
$Python = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

# Generate one fresh, 120-second segment replay and self-contained HTML viewer.
& $Python -m src.backbone.release_baseline `
  --config configs\backbone_release_baseline.json `
  --output "outputs\backbone_release_demo_$Stamp"

# Run 10 targeted regression files, source-only import check, and a second
# fresh 600-sample mixed-action replay with machine-readable verification.
& $Python -m src.backbone.verify_release_baseline `
  --config configs\backbone_release_baseline.json `
  --output "outputs\backbone_release_verification_$Stamp"
```

The current verified demo is
[`viewer.html`](../outputs/backbone_release_demo_20261025_final2/viewer.html).
Open it locally in a browser; use Play/Pause, select the evaluation-only annotation
toggle, and seek to sample238 (frame1720) to see an actual supported
HIGH_PASS prediction beside a separately recorded PASS annotation. This real
browser interaction passed; there was no displayed JavaScript exception.
The recorded interaction is
[`browser_smoke.json`](../outputs/backbone_release_demo_20261025_final2/browser_smoke.json).
Machine verification is
[`verification.json`](../outputs/backbone_release_verification_20261025_final2/verification.json).

The release verification run passed **77 targeted tests**, including four new
release checks; a temporary source-only copy imported all runtime/transitive
modules from its copied `src/` tree without consulting workspace source.
It generated 600 samples and four supported event-conditioned queries across
DRIVE/HIGH_PASS/PASS. All four predictions were recorded as actual model
outputs. This is a local retrospective demonstration, not prospective
generalization.

The external-metre adapter was separately smoke-tested on50 rows from the
existing annotation-backed interoperability fixture. It produced G1 estimates
without inventing action targets or queries; the fixture is not video-derived.
Its scoped result is
[`replay_validation.json`](../outputs/backbone_release_external_smoke_20261025_final/replay_validation.json).

Output directories must be empty/new. The baseline runner validates pinned
checkpoint, model-manifest, source-cache, association and training-index
hashes; it does not overwrite results. To use another input or model, copy the
canonical config, update the source adapter/configuration and its pinned
hashes, and preserve its provenance. `soccertrack_cache` and
`external_jsonl` are supported input adapters. External inputs must declare
105m x68m pitch metres, axes and 0.2-second timing; no pixel projection,
rescaling or future resampling is inferred. External action queries require
explicit actor/time and stable query identifiers; unverified labels are
rejected. A replacement model must implement the existing G1/S4 contracts,
pass the scoped regressions, and have its checkpoint/config hash and training
provenance updated in a copied manifest. Do not change defaults merely by
repointing to an experimental artifact.

This is **not fresh-machine reproducibility**. The tested interpreter is
Python3.14.8 on Windows x64, Torch2.11.0+cu128, NumPy2.5.3, SciPy1.18.1; CPU
inference passed. `requirements.txt` lists the broader research environment,
and a clean install was not tested. Local checkpoints, caches, association
manifests and corpus-index shards are required and are excluded from the
source review bundle.

## Limits retained at release

- **Video-to-pitch observations are blocked.** The regional validation failed
  and accepted zero detector-derived pitch coordinates.
- Player visibility is simulated from tracking and a causal motion heading;
  it is not measured gaze.
- S4 needs supplied actor and event-query times. It is not a continuous action
  detector or a best-action recommender.
- Seven trained classes do not cover all major football actions. Unimplemented
  branches remain unsupported.
- Ball state and possession, unsupported targets, location/direction, timing
  and calibrated confidence remain unknown/unset unless independently supplied.
- Class scores are uncalibrated and are not tactical value or cross-branch
  action ranking.
- The match-excluded distillation experiment was harmful; the original S4
  remains the default. Robust and teacher/student alternatives remain available
  as experimental artifacts, not deployment defaults.
- The 93.45-minute full-stream score uses continuous-prefix memory. The
  3,796-event tensor benchmark uses a separately initialized event protocol;
  their metrics are not directly comparable.
- The source-only test verified imports with already installed dependencies;
  it did not verify clean-machine dependency installation or include private
  data/checkpoints.

The earlier acceptance records remain unchanged: original v0.6 and the
completion continuation both report **PARTIAL**, the video bridge is
**BLOCKED**, and the GNN queue is `PAUSED_FOR_INTEGRATION`.

## Small prioritized backlog (not release gates)

1. **Unblock the observation bridge:** independently validate a regional
   calibration, detector footpoints, synchronization and identity continuity
   before any video-derived pitch input is accepted.
2. **Improve transfer evidence:** resolve weak/collapsed teacher targets before
   considering another matched teacher/student comparison; do not promote C2.
3. **Expand action support and perception evidence:** separately annotate and
   validate missing action branches, ball/possession, and measured player-view
   observations rather than assigning unknowns guessed labels.
4. **Align evaluation protocols:** only compare streaming and event-tensor
   results after using common samples, targets and initialization/history
   protocol.

No backlog item is authorized by this release closeout.

## Files ready for version-control review

The release-specific source/documentation set is:

- `configs/backbone_release_baseline.json`
- `configs/backbone_release_manifest.json`
- `src/backbone/release_baseline.py`
- `src/backbone/verify_release_baseline.py`
- `tests/test_backbone_release.py`
- `results/tracking_backbone_release_baseline.md`
- `README.md` (release-baseline pointer only)

The required pre-existing inference modules and regression tests are
enumerated in the model/source manifest. Do not add local data, weights,
footage, annotations or generated replay outputs to the source bundle.
