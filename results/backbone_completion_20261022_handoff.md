# Backbone completion continuation — 2026-10-22

## Status: PARTIAL, with the video observation bridge still blocked

This is a separate continuation of the corrected v0.6 assignment, not a
retroactive completion of its expired deadline. The original v0.6 handoff and
outputs remain unchanged. No commits, pushes, human annotation edits, new
downloads, or GNN-queue resume occurred.

The actual connected prototype still uses **G1 plus the original v0.5 S4**.
Passing is optional and disabled in the full-half workload. Ball, possession,
restart context and outcomes remain unknown. S4 is an event-conditioned,
seven-class classifier, not a continuous detector or best-action recommender.

All new outputs are under `outputs/backbone_completion_20261022/`.
Start with `acceptance_record.json`, `requirement_ledger.json`, `viewer.html`,
and `model_source_registry.json`. The ledger preserves every non-heading,
nonempty line of the original assignment, its original line number, and a
status/result. `NOT_RUN` means not separately re-audited in this continuation;
it does not erase historical evidence or turn inherited work into a new pass.

## Measured work and decisions

| Requirement / branch | Actual result | Decision |
|---|---|---|
| Match-excluded action transfer | Three scratch-initialized teachers; 12,038 keyed fitting OOF targets; one six-fit teacher reference; matched seed-42 C1/C2 students | Implemented and validated; negative model result |
| Two complete tracking halves, four actors each | 118577 first half: 48.2833 min / 57,940 actor-steps; 128058 second half: 45.1667 min / 54,200 actor-steps | PASS |
| Full chronological workload | **93.45 native source minutes, 28,035 five-Hz source samples, 112,140 actor-steps**, 95 scheduled/supported S4 calls; 112,049 streamed G1 calls | PASS; no short-window substitution |
| Actual hidden-state use | 751,077 G1 hidden-entity estimates; 492,903 last-seen hidden estimates | Stream-use counts, not checkpoint-load counts |
| Recovery | Worker exited at a saved 300-sample boundary, then resumed; 1,200 four-actor prefix records match an independent pilot exactly; no duplicate actor/time records | PASS |
| State-restore numerics | Both halves: 25 comparisons after a 300-sample prefix, zero difference; identities and memory retained | PASS |
| Optimizer continuation | Disposable uninterrupted two-step control versus persisted boundary and one actual resumed step: zero parameter difference | PASS; canonical checkpoints unchanged |
| Lazy fitting index | **100,000 distinct actor/time windows**, 12 fitting halves, 63,267-byte compressed index; 11,932 missing/unknown actor-metadata windows retained | PASS; no semantic labels |
| Development failure tests | 49 class/condition streams, 5,929 actor-steps; seven classes; source gaps, noise, unknown metadata, camera cuts, identity changes after memory formation | PASS |
| Measured observation hotspot | Cached deterministic dropout draws: 4.40x observation-path and 1.39x full-replay speedup; exact stored output parity | Improvement measured; global-worst-bottleneck superlative NOT established |
| Another recording through public adapter | Unmodified CLI on 128058 second half: 600 samples, exact observation/inference parity, 3,537 learned hidden estimates and 1,614 available lane assessments | Same-dataset interoperability only |
| Narrow video-to-pitch attempt | Frozen left-region mapping rejected; zero accepted detector-derived pitch records | BLOCKED; do not call this video-based action inference |

The full streams use 49 and 46 compressed storage chunks. Memory is carried
across chunks; maximum retained memory entries were 22 and 25, and history
length stayed bounded at 11. These dictionary/state measurements are not an
OS/GPU peak-memory profile. Stream execution took approximately 1,174 seconds
combined, excluding parity probes, auxiliary audits and packaging.

No source intervals were trimmed for missingness. The selected four actors
happened to have no actor-missing samples in these two runs; separate
development tests explicitly remove actors/source packets and verify abstention.
At five-Hz sampling, each half has four final native frames after its last
sample; no future resampling was performed.

## Transfer: a working connection, not an accuracy improvement

Existing `PartialObservationStudent` architecture: width96, two layers, four
heads, dropout0.1. The teacher receives privileged tracked histories; students
receive only their permitted partial-observation histories. G1 predictions were
not injected into S4.

The frozen recipe used batch256, learning rate0.0001, weight decay0.0001,
seed42, development stopping with patience2 and minimum improvement0.0005.
C2 adds KL distillation with temperature2 and weight0.5 times temperature squared.
C1/C2 share initialization, corrupted arrays, sample IDs and deterministic order.

Teacher folds exclude, respectively:

1. 117092 and117093.
2. 118576 and118578.
3. 128057 and132831.

Each teacher fits the other four fitting matches from scratch. Fixed physical
normalization has no fitted cross-match statistics. Match132877 was reused for
bounded stopping, explicitly not an independent evaluation set. The six-fit
reference also produced keyed development and retrospective predictions.

Actual optimizer steps: teacher folds **256,256,192**; six-fit reference **240**;
C1 andC2 **192 each**. All completed fits retain best and last checkpoints,
optimizer/RNG state, specifications, histories, source hashes and keyed targets.

| Common examples / condition | Original S4 | C1 corruption control | C2 + distillation |
|---|---:|---:|---:|
| Development clean macro-F1 | 0.4766 | 0.4578 | 0.2435 |
| Development corrupted macro-F1 | 0.4151 | 0.4401 | 0.2423 |
| Retrospective clean macro-F1 | 0.5188 | 0.5167 | 0.2800 |
| Retrospective intensified-stress macro-F1 | 0.4702 | 0.4690 | 0.2599 |

The OOF teacher macro-F1 was only0.1376; its hard predictions collapsed to
PASS/DRIVE (7,660/4,378 examples), with zero hard predictions in five classes.
Distillation was harmful. No second teacher recipe, seed sweep, or automatic
specialist switch was launched. Original S4 was retained using development
evidence before the retrospective audit. Retrospective matches have been used
previously and are not a fresh test set.

The 95 exact-grid full-half queries are a **different observation protocol**
(complete-prefix memory), with macro-F1 0.2753 and accuracy0.5684. They cover
six annotation classes; HEADER has zero scheduled examples on this exact grid.
All seven output classes remain available, and all seven are exercised by the
development stress subset. Do not compare this95-query score directly with
the3,796-event tensor benchmark or omit the actual errors.

## Input and loading contracts

- S4:11 samples over2s,0.2s spacing,22 anonymous entity slots,10 features;
  simulated120-degree/45m observation support. Player-local observations and
  causal memory only; no global ordering or true action label is a feature.
- Normalization: actor-relative x/52.5, y/34, velocity/12; training age/5.
  The normal adapter explicitly converts its age/8 convention to training age/5.
- Visibility, ever-seen/memory age, team relation, role and actor flags are
  explicit. Unknown team/role or absent/discontinuous actor can block the
  trained tensor; they are not filled from privileged references.
- Teacher inputs use the original causal history builders, fixed physical
  scales and a deterministic anonymous permutation. Privileged histories are
  stored separately from student arrays and keyed soft targets.
- `ActionModelSession` loads original S4 and both experimental student
  checkpoints; direct/wrapper probability differences were exactly0.
- `ModelSession`, `ReplayEngine`, `SoccerTrackSource` and
  `ExternalObservationSource` remain the normal connected inference path.
- Returned action scores are uncalibrated class probabilities, not calibrated
  confidence or tactical value. Target/location/direction/timing/confidence
  fields remain unset when unsupported. Belief means/std and optional occupancy
  are separate uncertainty representations, not action-ranking scores.

The lazy index references immutable shared caches instead of duplicating dense
tensors. `LazyWindowDataset.observations()` supplies causal partial observations;
`privileged_window()` is a separate supervision-only accessor. No missing event
was converted to NO_ACTION, and no off-ball/goalkeeper/clearance labels were added.

## Video bridge: precise remaining blocker

One new, predeclared raw-image thin-plate interpolation attempt was restricted
to the left image region (not asserted to be a physical camera). It used25
released landmarks and three separated pitch-width validation bands.
Only **two** held-out points lay inside the fitted support hull; their diagnostic
median/p90 errors were **8.35m/10.99m**, failing both support and error gates.
The original panorama gate remains rejected; the narrow condition did not
replace or weaken that original requirement.

The reviewed continuous source minute is video seconds240–299.8:300 existing
five-Hz pixel records and1,615 detections. Only35 detections fall in the proposed
region, spread over16 unverified tracklets; the largest has11 ROI detections.
This does not establish continuous athlete identity. No new video-inference
frames or unique footage minutes are claimed for this review of cached outputs.

`review_only/index.html` contains three timestamped rejected-region images and
14 links to post-memory identity/unknown-role cases. It contains no filled human
labels. Physical footpoint accuracy, accepted region coverage, identity
continuity and team/role evidence remain prerequisites. More uncalibrated
detection volume was deliberately not substituted for this missing bridge.

## Verification and known limitations

- **73** relevant local tests passed, including explicit artifact integration.
- **9 portable fixture tests** passed in a source-only copy without data,
  checkpoints or videos. These overlap the73; do not count82 independent tests.
  Attempting the artifact tests without their data correctly failed; they were
  not reclassified as portable passes. A first incomplete copy also exposed a
  missing repository `train/` source folder, which was then included.
- Browser assertions used real Play/Pause clicks, both long source timelines,
  all eight actor selections, actual event queries, remembered uncertainty,
  unknown possession, truth-off default and unchanged predictions when truth,
  annotation, occupancy and reference controls were toggled.
- Browser failures led to bounded fixes: absent overlays now report unavailable
  instead of throwing; a sticky header no longer blocks controls in a short
  viewport. Both singular and multiple recorded-annotation formats display.
- SciPy was already installed; its direct use is now declared in the dependency
  manifest. No package installation or environment-policy change was performed.
- The remaining computation budget was not filled with repeated fits or
  recycled footage. The original16–19-hour target remains unmet, and the
  acceptance remains PARTIAL rather than presenting a fallback as the bridge.

## Exact Windows PowerShell commands

Run from the repository root with the existing local prerequisites:
the current v0.5 training index/shards, verified tracking caches/associations,
original S4/G1 artifacts, v0.6 corrupted-input artifacts, cached video records,
released calibration metadata and local source video. Large/private artifacts
are not included in source control.

These commands resume this saved local program, not a fresh-machine bootstrap.
The second-recording interoperability bundle and actual browser-validation
record are required existing evidence; the orchestration does not manufacture
them. If the packaged viewer changes, rerun the real browser assertions and
bind their record to its new hash before verification can pass.

```powershell
$py = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$out = 'outputs\backbone_completion_20261022'

# One main execution/resume command; completed fitting stages are not refitted.
& $py -m src.backbone.completion_audit --phase resume --output $out

# One verification command: actual artifacts, full streams, hashes and73 tests.
& $py -m src.backbone.completion_audit --phase verify --output $out

Start-Process (Join-Path (Get-Location) "$out\viewer.html")
```

The continuation compute deadline is2026-10-22T20:51:15.185Z, preserving two
hours before its separately recorded upper bound. `resume` refuses to reset an
expired deadline; `verify` remains available afterward. Native Python commands
are used directly; this machine's disabled PowerShell-script policy was not
changed or bypassed.

For an active full-half worker, create
`outputs\backbone_completion_20261022\full_halves\STOP_REQUESTED`.
It stops after an atomically saved chunk; remove only that named marker before
resuming. Source/model/implementation fingerprint changes intentionally reject
incompatible resumptions. All current full-half work is complete.

**Next prerequisite:** independently supported, denser regional calibration and
detector-footpoint/association validation. Do not promote C2, infer team/roles
from hidden truth, or claim video-to-player action inference from this run.
