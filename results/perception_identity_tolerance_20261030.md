# Perception, identity continuity and downstream tolerance

Research update, October 30, 2026. **The supported tracking release and its
G1/original S4 checkpoints are unchanged.** Experiment execution completion
is separate from scientific improvement and historical release acceptance.
The global video-to-pitch validation gate remains unsatisfied.

## What is published

- Shared finite-update/AMP accounting and a disposable CUDA stability runner.
- The bounded detector continuation, causal actor/cohort replay and standalone
  identity-program source, with their regression tests.
- A fixed-input association audit and frozen-model dose-response worker.
- The original tolerance [protocol](association_tolerance_20261030/protocol.json),
  [association comparison](association_tolerance_20261030/association_summary.json),
  [tolerance table](association_tolerance_20261030/tolerance_table.json),
  [per-match/seed aggregate metrics and block intervals](association_tolerance_20261030/sensitivity_metrics.json),
  [decision](association_tolerance_20261030/decision.json), dose-response SVGs,
  [GPU parity](association_tolerance_20261030/gpu_parity.json) and
  [workload](association_tolerance_20261030/inference_workload.json).

No datasets, annotations, footage, weights, crop corpus, embeddings, dense
predictions, local authorization attachments or return ZIPs are published.
Source programs intentionally reference excluded local prerequisites; this
is not a self-contained training/evaluation distribution.

## Preserved progression and negative results

### Numerical safeguards and detector continuation

Local run `outputs/as_20261029_134730` completed 1,000 committed CUDA updates
with no skips. Controlled overflow tests verified that skipped AMP updates
do not update optimizer/EMA state and that subsequent finite updates succeed.
The historical epoch-16 failure remains unexplained.

The separate run `outputs/pc_20261029_144726` completed 20 additional epochs:
8,996 committed updates and four AMP skips. Development selection preceded
retrospective evaluation. Recorded retrospective footpoint F1 changed from
0.718749 to 0.833147 on 117092 and 0.907414 to 0.914568 on 118576. These are
footpoint metrics, not box mAP. Local selected detector SHA256:
`641e19ecf639f7b0b628221c3c9c1b410433b4e4672de425e9babc0e491860f2`.
It is not a release-default replacement or certified pitch observation source.

### Actor continuity

In the earlier ten-minute diagnostic, original anchors had 359 available
actor-samples out of 6,000 opportunities, all in the first minute. Runtime-ID
loss alone did not establish physical disappearance: post-inference reference
matching found many detections under different runtime IDs.

`outputs/ac_20261029_195056` tested a global elapsed-time/margin association.
Development rejected it. On the ten-minute baseline replay, FIXED_ACTOR had
359/6,000 available samples, 213 G1 calls and 2/58 supported S4 queries.
OBSERVABLE_COHORT had 4,593/6,000, 2,548 G1 calls and 25/58 supported S4 queries.
The cohort uses separate causal sessions; it did **not** recover the original
people or splice different people into one memory.

### Appearance learning: no tracking improvement

`outputs/ip_20261029_202245` built 95,357 conservative crops, including
89,074 fit and 6,283 development crops. Full-body crops came from the local
COCO-pretrained YOLO11n detector; annotation footpoint boxes were not treated
as body crops. This backbone is not a certified pretrained person-ReID model.

R1/R2 same-kit temporal development retrieval rose from 18.3% to approximately
74-75% on 535 queries, within fitting matches, not new-match recognition.
R1 committed 54,230 updates with 19 AMP skips; R2 committed 57,011 with 20
skips. Both complete routes failed development tracking selection.
The old comparison also changed association mechanics, so it was **not**
an embedding-only ablation. The baseline remained selected.

The 13.91-hour worker evaluated 13 compatible complete halves (602.4 minutes).
117092 second half was excluded for a one-frame video/annotation clock mismatch.
For retrospective 118576 second half:

| Route | Switches / 279,239 matched transitions | Consistent 11-sample histories / 320,210 truth instances |
|---|---:|---:|
| Baseline | 30,532 | 145,213 |
| R1 | 108,923 | 39,540 |
| R2 | 100,619 | 43,698 |

These are imperfect post-inference annotation matches, not identity ground
truth certification. More neural calls or more persistent IDs are not accuracy.

## Association audit: same cached inputs and shared mechanics

`outputs/at_20261030_124211` first reproduced a stationary-player edge case:
valid appearance at 0, .2, .4, .6, .8 seconds, missing only at 1.0, then valid
at 1.4 caused a new original-tracker ID. A separate experimental implementation
retains the last valid feature and its original timestamp for at most 0.8s;
cuts reset state, missing crops do not refresh age, and uncertain relinks abstain.
Historical implementations and outcomes are unchanged.

On development, 6,834/17,272 detections lacked usable appearance. R1 erased
earlier valid features 1,315 times. Retention fixed the synthetic case but
worsened both real development comparisons:

| Development source | R1 switches before/after repair | Transition denominator | Fragmentation before/after |
|---|---:|---:|---:|
| 117092 | 2,205 / 2,403 | 9,368 | 2,654 / 2,878 |
| 117093 | 1,568 / 1,659 | 6,092 | 1,591 / 1,685 |

The edge case is material in frequency but does not explain away the full
tracking regression. Same assignment, motion model, timestamps, retention,
gap lifecycle and unmatched behavior were then used for motion/HSV/R1/R2.
HSV uses half-L1; embeddings use cosine. Two development-positive quantiles
per representation set distance scales before retrospective scoring.

Predetermined cached 600-1,200 second intervals, without rerunning detection:

| Source | Route | Switches / matched transitions | Fragmentation | Consistent histories / 66,000 truth instances |
|---|---|---:|---:|---:|
| 118576 second | Production reference | 6,695 / 56,397 | 3,191 | 28,034 |
| 118576 second | Shared motion | 8,442 / 56,397 | 8,342 | 26,790 |
| 118576 second | Shared HSV | 8,109 / 56,397 | 7,572 | 26,265 |
| 118576 second | Shared R1 | 10,650 / 56,397 | 10,932 | 20,684 |
| 118576 second | Shared R2 | 10,099 / 56,397 | 10,333 | 22,200 |
| 128057 first | Production reference | 3,623 / 50,963 | 2,431 | 27,637 |
| 128057 first | Shared motion | 4,644 / 50,963 | 5,520 | 26,451 |
| 128057 first | Shared HSV | 5,101 / 50,963 | 6,011 | 24,382 |
| 128057 first | Shared R1 | 6,794 / 50,963 | 8,163 | 19,521 |
| 128057 first | Shared R2 | 6,622 / 50,963 | 8,021 | 20,071 |

The production reference does not share the new mechanics and is labeled
separately. R1/R2 also hurt relative to shared controls. No candidate passed
development selection; the production baseline remains retained.

## Downstream sensitivity: availability is not survivor accuracy

All 5,954 stable-key indexed queries were evaluated: 2,158 development queries
from 132877, 1,773 retrospective queries from 118577 and 2,023 from 128058.
All seven trained action classes are included. These are not fresh confirmatory
matches. No training was performed.

Every condition starts from the same empty, up-to-eight-second causal prefix
(41 samples, exact native query offset), not a full-half memory prefix.
Clean simulated 120-degree/45m visibility is fixed. Controlled corruption
enters observations **before** released causal memory; no future interpolation.
The original focal token is never rebound after fragmentation. G1 scoring uses
fixed clean hidden targets and original token/segment keys; no nearest-person
rescue. G1 hidden estimates do not feed S4.

31 nonzero single-factor conditions, three seeds, cover focal gaps and
fragmentation; nonfocal same-team/opposing-team swaps and fragmentation;
independent position noise, fixed common translation and causal shared drift.
Rates are per eligible player/pair-second, not unspecified frame percentages.
Exposure totals sum overlapping windows, not unique full-half player-time.
Source-level visibility/heading feedback was not run and is not certified.

Working criteria were locked before scoring: retain >=90% of clean support,
S4 paired fixed-seven-class macro-F1 loss <=.03, G1 paired mean-error increase
<=20%. These are provisional engineering criteria, not scientific standards.
Newly supported examples do not offset lost clean-supported examples.
Reports include per-match/seed curves, missing reasons, all-query
correct-and-supported rates, fixed-target coverage, tails, uncertainty and
paired 60-second temporal-block bootstrap intervals.

| Condition | Measured conclusion across all three matches/seeds |
|---|---|
| Independent positional noise | 0.25m radial RMS passed both models; 0.5m failed G1's error envelope |
| Causal shared drift | 0.5m stationary radial RMS passed both; higher tolerances differ by model/match |
| Focal gaps | No tested duration .2-.4-1-2s at .1 starts/player-second passed the availability envelope |
| Focal fragmentation | No tested .02/.1/.25 starts/player-second passed the envelope |
| Nonfocal swaps/fragmentation | S4 passed all tested rates; G1 only passed .02 across all matches/seeds |
| Constant shared translation | S4 passed .25-4m; G1 absolute-position envelope passed only .25/.5m |

These are discrete tested levels, not calibrated safety cutoffs. For 118577
seed42, .2s focal gaps reduced S4 support to 1,419/1,773 (80.0%), while paired
survivor predictions were unchanged. At .5m independent RMS noise, G1 error
rose from 1.37m to 1.75m (+27.9%) with full support. Same-team swaps at
.25/eligible-pair-second raised G1 error to 3.35m, versus 2.97m for opposing-team
swaps; same-team errors are not harmless to identity-keyed reconstruction.

Clean retrospective S4 macro-F1 was only .424 (118577) and .502 (128058).
Clean G1 mean errors were 1.37m/1.40m, versus last-seen 1.84m/1.88m and
constant-velocity 2.18m/2.15m on fixed hidden targets. Preserving this clean
performance does not make the models suitable for coaching or tactical advice.
The next priority is original-focal continuity/detection coverage, followed by
position/projection accuracy, not another appearance fit.

## Validation and local reproduction

The tolerance assignment completed in 2.13 hours. Its final verification
passed 57 tests; protected hashes were unchanged. FP32 CPU/sequential/CUDA
batch parity differed by at most 1.91e-6 (tolerance 2e-4). The RTX A5000 processed
512,058 batched inputs for each model, in 4,704 forward calls per model.
Parity checks additionally exercised the sequential APIs.

Install the existing video dependency manifest only in your chosen environment.
Ultralytics retains its upstream license. The code imports no newly downloaded
model or package automatically for this research result.

Publication verification passed **100 tests** with the existing local assets.
An exact staged source checkout passed **96 fixture tests** without
those assets. Four integration checks require excluded local weights, detector
provenance or the annotated event index; they are not fixture-only checks.

Fixture-only regressions:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q `
  tests\test_amp_guard.py tests\test_gpu_perception.py `
  tests\test_perception_continuation.py tests\test_actor_continuity.py `
  tests\test_identity_program.py tests\test_association_sensitivity.py `
  tests\test_video_player_cross_match_replay.py tests\test_backbone_release.py `
  -k "not encoder_initialization and not manifest_protects_sources and not release_config_has_supported_mixed_action_queries and not event_annotation_join_uses_both_stable_keys"
```

The original full worker command, requiring all excluded historical caches,
checkpoints and frozen inputs:

```powershell
& .\.venv\Scripts\python.exe -u -m src.backbone.perception_sensitivity `
  --output outputs\at_20261030_124211 resume
```

Do not run that command against a completed run or reset an expired historical
deadline. For a newly authorized reproduction, create a new output directory,
copy the published protocol there, declare new start/cutoff/reserve/deadline
timestamps **before** generating predictions, preserve all condition definitions,
and record source/config hashes. The CLI `reproduce` command exercises only the
synthetic lifecycle case; `run` advances association then sensitivity/reporting;
`verify` checks local pins/tests. Worker locks are Windows-specific.

The complete local packages preserve richer keyed metrics and source patches.
The compact source-controlled evidence omits per-query identities, absolute
workstation paths and local authorization records.
