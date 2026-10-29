# Backbone v0.2: external observations and batch ingestion

**Working ingestion bridge, not a raw-video-to-pitch system. No training, research resume, annotation edits, commits or pushes.**

## Action-general system scope (October 20 correction)

Project target: a shared backbone by **October 31, 2026** and an integrated major-football-action prototype by **December 31, 2026**. Current scope is **all major actions**, not passing alone. This handoff corrects the earlier pass-centred framing without expanding the bounded ingestion sprint or changing learned checkpoints.

The backbone accepts any configured observed player, including an off-ball player; neither the source adapters nor training exporter filter for PASS events. `action`, `restart_context`, `outcome`, `ball_state`, and `possession` are distinct contracts. External observations do not require an event row. Missing action/event annotations mean **unannotated/unknown**, never negative or “no action.” Existing fitting-match training exports remain observation examples with optional labels absent; no synthetic off-ball labels were made.

The versioned general output is `football.action-output.0.1`: `action_type`; optional teammate/opponent target; optional pitch location/direction; optional timing interval and confidence; plus `supported`, `partial`, or `unsupported` status and reason. Current action inference returns `action_type=null`, `support_status=unsupported`, with an explicit “No action model is configured” reason. Ball position and possession independently take `observed`, `estimated`, or `unknown` status. Defaults are both **unknown**, with no coordinates, possessor, or selected-actor-possession assumption. Restart context and outcome independently default to unknown. No action branch is silently represented as an empty passing result.

Passing-lane assessment is a **named optional passing module** controlled by each batch job's `passing_assessment` flag. Its compatibility default is `true`; set it `false` and both the assessment and candidate ordering are absent, with module status `UNSUPPORTED`. The recorded example batch sets this false for `retro118577_t60`; all other jobs keep the compatibility default. It is not a field required by the general action contract. P0 receiver ranking remains an optional reference/module, never a general action choice. Passing, receiver selection, shooting, carrying/dribbling, defending/pressing, goalkeeper, set-piece/restart actions, and other branches have not acquired action models from this change. Their output must stay explicitly unsupported until implemented. Different branch scores must not be compared as though they rank actions or establish a best action.

## What is connected

[External JSONL adapter](../src/backbone/external.py) -> the existing `TrackingFrame` / `Observation` contracts -> synthetic provider OR already player-visible packets -> causal anonymous memory -> frozen two-second G1 -> optional passing geometry -> action-general output shell -> the existing offline viewer.

The transport envelope `backbone.observations.0.2` extends the existing records; it does not introduce a competing memory/model interface. Unknown goalkeeper roles are now represented by `null`. An unknown team or role disables learned inference explicitly; those observations remain in memory, not silently discarded. Existing known-role input behavior is covered by regressions.

[Batch runner](../src/backbone/batch.py), [configuration](../configs/backbone_v0_2_batch.json), [actual output directory](../outputs/backbone_v0_2_20261020_1558/), [current batch summary](../outputs/backbone_v0_2_20261020_1558/batch/batch_summary.json), [training shard index](../outputs/backbone_v0_2_20261020_1558/batch/training_index.json).

### Persistent measured results

- Ten predeclared 20-second segments, 100 chronological samples each: **1,000 actor frames**, 6,168 released measurements, 5,560 learned hidden-entity estimates, 4,598 explicit fallbacks, 2,187 available and 4,415 unavailable candidate assessments. The first replay explicitly disables the optional passing module; the other nine preserve the compatibility default.
- Eight retrospective external-file replay jobs: matches 118577 and 128058, fixed starts 60/120/180/240 seconds, one previously configured actor each. Selection did not use model predictions.
- Two training-export jobs: fitting matches **117092 and 117093 only**, fixed 60-second starts. Ten warm-up samples then 90 records each: **180 unique student/privileged/optional-target triples**.
- Successful action-general batch execution: **19.21 s**, including source/model loading and export. Shared G1 and P0 loaded once in one compatible model session. Hash-matched resume: **0.37 s**, all ten jobs skipped, zero model sessions loaded. Concurrent regression tests were running during the successful execution; these are functional timing measurements, not standalone throughput benchmarks.
- Current complete generations contain approximately **82.43 MB** of outputs before completion-marker overhead. Superseded and failed attempts are retained, not included in this size.
- Actual public-file round trip: **100 frames, exact observation and CPU prediction equality excluding latency**; numerical acceptance tolerance is absolute/relative `1e-5`. It exercised 391 learned hidden estimates and 238 available lanes. [Evidence](../outputs/backbone_v0_2_20261020_1558/roundtrip_validation.json).
- **65 tests passed, zero failed/errors/skipped**, covering external ingestion, action-general contracts and existing backbone/dynamic-football/hidden-defender regressions. [JUnit](../outputs/backbone_v0_2_20261020_1558/acceptance_tests.xml).
- Deliberately malformed count and forbidden retrospective training jobs both produced explicit failure records and CLI exit 1, not a successful batch. [Negative acceptance configuration](../configs/backbone_v0_2_malformed.json), [failure summary](../outputs/backbone_v0_2_20261020_1558/malformed_batch/batch_summary.json).

These results establish software interoperability and execution, **not physical source validity or tactical accuracy**.

## External JSONL schema

UTF-8; one metadata object followed by one frame object per line. [Real example](../outputs/backbone_v0_2_20261020_1558/inputs/roundtrip118577.jsonl) contains all 100 frames and annotation provenance.

Minimal structural example (a short packet example, not enough history for G1):

```jsonl
{"record_type":"metadata","schema_version":"backbone.observations.0.2","source_id":"camera_stream_A","match":"example","half":"1st","observation_kind":"exo_source","sampling_seconds":0.2,"coordinates":{"system":"pitch_centred","units":"metres","axes":"x_length_y_width","pitch_length_m":105,"pitch_width_m":68},"origin":{"type":"external_provider"},"quality":{"position_accuracy":"UNVALIDATED"},"calibration":{"status":"PROVIDER_DECLARED_NOT_CERTIFIED"},"synchronization":{"status":"PROVIDER_DECLARED_NOT_CERTIFIED"}}
{"record_type":"frame","frame_index":0,"timestamp_s":0.0,"segment_id":"camera_A_take_1","identity_reset":false,"camera_cut":false,"measurements":[{"track":"local_actor","segment":0,"position_status":"observed","xy":[0.0,0.0],"team":1,"goalkeeper":false,"quality":{"detector_confidence":0.9}},{"track":"local_other","segment":0,"position_status":"missing","xy":null,"team":"UNKNOWN","goalkeeper":null}]}
```

- Required metadata: source ID, match, half, origin, quality, calibration and synchronization provenance objects. Provider declarations are recorded, **not independently certified**.
- `exo_source`: camera/tracking-source measurements before simulated actor visibility. The synthetic FOV provider releases only permitted measurements to `ReplayEngine.step(Observation)`.
- `player_visible`: packets already released by an external actor observation provider. Header must also declare `actor_track`, which must equal the configured actor. They bypass synthetic FOV; no privileged positions are added.
- Privileged reference is not an accepted observation kind and cannot be mixed into this stream. Source overlays on external replays explicitly say **source observations only, not complete truth**. Player-visible sources do not get a hidden source overlay.
- Coordinates: declared `metres` only. `pitch_centred` passes xy unchanged; `pitch_top_left` subtracts `(52.5,34)`. For top-left, positive x/right follows pitch length, positive y/down follows width; centred y retains that sign. Model-compatible pitch is exactly 105 by 68 m. No inferred axis flips, pitch rescaling, image-pixel conversion or homography fitting.
- `team`: affiliation 1/2, or `UNKNOWN` / 0. Actor-relative team relation is derived only when affiliations are known. `goalkeeper`: true/false, or null when unknown. Never infer either from detector confidence.
- `quality` on measurements is optional and preserved separately in replay diagnostics. A detector confidence is **not** interpreted as calibrated metre uncertainty. Current visible geometry retains v0.1's point-measurement assumption; quality metadata does not calibrate that assumption.
- `position_status="missing"` requires `xy=null`; it is retained as source missingness, never released as a current position or filled from future samples. Omitted tracks are also permitted. Optional `absent_count` must be at least the number of explicit missing positions. Inventory completeness is always unknown/partial.
- Strict timestamp and frame-index order; duplicates rejected. `frame_index` is a **logical 25Hz grid**, not a claim about camera FPS. Successive supplied packets have index increments divisible by five and matching elapsed seconds (`delta_index/25` within `1e-4 s`). Gaps are allowed, but clear model history; there is no resampling/interpolation.
- Changes in `segment_id`, `camera_cut=true`, or `identity_reset=true` reset viewpoint memory/history, including when the actor is missing. Measurement `segment` changes retire the associated track's old memory. There is no cross-cut association without evidence.
- G1 still needs 11 consecutive actor-observed samples at 0.2 s, known team/role, and at most 22 first-sight tracks. Unsupported inventory/history remains explicit fallback. Age >2 s uses v0.1 fallback; expiry remains 6 s. Never-observed entities are not named.

## Batch and export contracts

Each job declares adapter/path (for external files), match/half, start frame and sample count (end exclusive is `start_frame + samples*5`), actors, intended use, observation settings, backend/checkpoint experiment and output destination. Defaults are expanded per job. `--only` selects one job without editing Python.

No event/action-type selector exists in source or export; arbitrary observation times are accepted. Jobs may select any source-local player ID without declaring possession. The replay emits a general unsupported action output unless a future action module supplies it. Restart and outcome remain separately unknown unless actual provenance-backed labels are provided.

Jobs are bounded to 11..1500 samples and 1..4 actors. External files are parsed line-by-line; only the selected bounded segment is retained for viewer construction. Existing SoccerTrack caches load one match/half per job; there is no collection-wide source cache. The model session is reused across consecutive compatible jobs, with only one session active.

Completion fingerprints cover the resolved job configuration, external-file hash OR cache/metadata/association hashes, selected checkpoint/config/result hashes, integration manifest and relevant implementation hashes. Source/model/research/code hashes are checked again before atomic completion. Markers include artifact hashes; missing/corrupt artifacts are rebuilt, never appended. Changing one job does not invalidate an unchanged independent job.

Directories use a 16-character hash prefix to avoid Windows path-length failures; the full SHA256 is recorded and checked for collisions. An initial real execution exposed this Windows rename limit in training exports; those attempts were recorded as FAILED, fixed and successfully rerun. Earlier attempts and invocation summaries remain for audit.

Resume is **per segment**, not mid-frame. A failed/interrupted segment is recomputed chronologically from its start and atomically replaces its record files. Completed jobs are skipped only with matching hashes. Failures are persisted and make batch status FAILED / CLI exit 1; independent jobs still run. Progress, execution time, frame/coverage/fallback/unavailable counts and output sizes are persisted.

**Consume only the current `training_index.json` shards, not a recursive glob of attempt directories.** Each training shard has:

- `student.jsonl`: allowed anonymous observations, causal memory and exact optional G1 input tensors. Source association tokens are excluded from learning fields. `sample_id`/timestamp/frame index are join metadata, not model features.
- `privileged_exo.jsonl`: separately keyed available annotation states and provenance. External observation-only sources get unavailable privileged targets, never invented full-state supervision.
- `optional_targets.jsonl`: action/event/receiver targets and teacher scores remain null with `target_availability.action=false` and `meaning=UNANNOTATED, not a negative action or no-event label`. Arbitrary timestamps are not pass decisions. In-sample P0 or G1 output is **not** claimed as out-of-fold teacher supervision.
- `manifest.json`: permitted fitting-match membership, source/checkpoint provenance, record count and separation. Development and retrospective matches are rejected for training export; external match/half metadata must match configuration.

Passing lane outputs in an optional replay are geometry diagnostics only; they are not action labels, a discrete pass prediction, or comparable to scores from future shooting/defending branches. An unknown ball/possession state does not suppress general memory or imply the actor is on-ball.

P0 remains optional, loaded once, in a separate full-state reference panel only. External files have no supported full-state P0 state builder, so P0 is explicitly unavailable there. Annotation-backed training replays can display the existing P0 reference; it never enters student features or player assessment.

## Bounded video pilot

[Pixel manifest](../outputs/backbone_v0_2_20261020_1558/video_pilot/manifest.json), [genuine image detections](../outputs/backbone_v0_2_20261020_1558/video_pilot/pixel_observations.jsonl), [diagnostic overlay](../outputs/backbone_v0_2_20261020_1558/video_pilot/overlay_1500.jpg).

Existing local 117092 first-half video, fixed 60..72 s interval; 12 actual decoded images at 1 Hz. Existing YOLO11 nano COCO pose weights, CPU, four overlapping image tiles, confidence 0.15 and image-space NMS. **25 detections in 13.86 s** of detection/export execution, before final source-file hashing. Pixel boxes may be false positives; no detection-accuracy claim. No annotations supplied to the detector, no new downloads, no new homography, no GSR positions/IDs substituted.

| Stage | Result |
|---|---|
| Pixel observations | OBTAINED; decoded frame ordinals/decoder times and source/weight hashes saved |
| Compatible camera-to-pitch projection | BLOCKED; no accepted per-camera/seam mapping for these detector observations |
| Independent projection/synchronization validation | NOT VALIDATED; video ordinal is not physical sync evidence |
| Identity continuity and teams | NOT ESTABLISHED / UNKNOWN |
| Pixel-only input to pitch backbone | REJECTED explicitly; error retained |

Existing exploratory panorama-thirds calibration is retained as negative evidence, not applied. Its withheld regional median errors were approximately 113.1/11.6/12.9 px, with large tails; image thirds are not verified physical cameras.

**Next missing adapter capability:** independently validated camera-to-pitch and timing correspondence, with source-local identity continuity and team/role availability. Actual actor visibility must remain distinct from broadcast source missingness. This sprint does not solve gaze, facing, uncertain receiver endpoints or a receiver policy.

## Exact Windows PowerShell commands

From the repository root:

```powershell
$py = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$cfg = 'configs\backbone_v0_2_batch.json'
$out = 'outputs\backbone_v0_2_reproduce'

# One external file: input path and actor are declared in the selected job.
& $py -m src.backbone.batch --config $cfg --only retro118577_t60 --output "$out\single"

# Run all jobs; repeat with --resume to skip only hash-matched completed work.
& $py -m src.backbone.batch --config $cfg --output "$out\batch" --resume

# Open a generated offline replay; no server required.
$summary = Get-Content "$out\batch\batch_summary.json" -Raw | ConvertFrom-Json
Start-Process (Join-Path $summary.jobs[0].output 'viewer.html')

# Produce only the fitting-match training shard.
& $py -m src.backbone.batch --config $cfg --only train117092_t60 --output "$out\training_only" --resume

# Validate an external file schema, without claiming physical validity.
& $py -m src.backbone.external --validate outputs\backbone_v0_2_20261020_1558\inputs\roundtrip118577.jsonl

# Repeat the actual real-inference parity check.
& $py -m src.backbone.roundtrip --input outputs\backbone_v0_2_20261020_1558\inputs\roundtrip118577.jsonl --output "$out\roundtrip.json"

# Deliberate negative test: expected CLI exit code 1.
& $py -m src.backbone.batch --config configs\backbone_v0_2_malformed.json --output "$out\malformed"

# Regression suite and persistent final-bundle acceptance.
& $py -m pytest tests\test_backbone_ingestion.py tests\test_backbone.py tests\test_dynamic_football.py tests\test_hidden_defender.py -q --junitxml=outputs\backbone_v0_2_20261020_1558\acceptance_tests.xml
& $py -m src.backbone.validate_v02 --output outputs\backbone_v0_2_20261020_1558
```

The eight public files are already supplied. To regenerate fixtures from available verified local annotations, use `python -m src.backbone.prepare_demo --config configs\backbone_v0_2_batch.json --output outputs\backbone_v0_2_20261020_1558`; existing fixtures are validated, not overwritten. To ingest genuinely new observations, edit a copy of the JSON configuration's adapter/path/actor/segment/provenance settings, not Python code.

Original v0.1 outputs and paused research artifacts are preserved. [Final acceptance](../outputs/backbone_v0_2_20261020_1558/acceptance.json) records protected hashes, actual browser checks and measured wall time. No claim that schema/tests establish camera validity.

## December 31 action-integration plan boundary

The October 31 deliverable is still the shared, action-agnostic backbone: source and player observation, ball/possession state with provenance, temporal memory, actor/context selection, uncertainty, timestamps, support/abstention and evaluation boundaries. Each later action head should plug into the general `ActionOutput`, declare its required evidence and supported contexts, and separate action predictions from restart context and eventual outcome. Add and validate branches against branch-appropriate targets/protocols; do not compare unlike raw scores across actions or infer an overall best action without a separately defined calibrated decision layer. This sprint adds contracts and explicit unsupported statuses only—no training, action labels or scoring claims.
