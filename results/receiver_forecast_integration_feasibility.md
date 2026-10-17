# Receiver / trajectory integration feasibility

Date: 2026-10-16.

**Decision: technically feasible, but NOT READY TO TRAIN AS-IS.**
Proceed to an integration experiment only after implementing a documented frame
contract, association adapter, missing-feature policy and match-level out-of-fold
(OOF) forecasts. Do not attach the existing seven-match encoder to its own training
matches and treat those features as out-of-sample.

The forecasting application-level leakage audit is **PASS**, with selection and
upstream-provenance qualifications detailed in the
[leakage audit](trajectory_leakage_audit.md).
No augmented ranker or forecasting model was trained in this task.

## Existing interfaces actually checked

- [Receiver builder](../src/build_pass_receiver_dataset.py#L235) reads exact decoded
  GSR frames `f-10,f-5,f`, where `f` is the frame in the example ID.
- [Anonymous association](../src/make_temporal_sft_dataset.py#L75) locates the
  passer by identity, sorts teammates/opponents by distance at the current frame,
  breaks distance ties by original annotation order, and keeps those identities
  fixed across history. IDs and jerseys are not prompt features.
- [Receiver tensor parser](../src/receiver_tensor_dataset.py#L44) produces three
  anonymous 22-slot states with observation masks. There may be fewer than ten
  observed candidate labels; absent slots are masked.
- [Candidate features](../src/models/candidate_ranker_receiver.py#L27) already
  contain current/history relative distances, movement and opponent pressure.
  New features should measure incremental forecast information, not duplicate
  these features and attribute all gains to social reasoning.
- [Forecast forward pass](../src/models/trajectory_forecasting.py#L91) operates on
  51 historical times and predicts all currently observed players simultaneously.
  It is permutation-equivariant and does not require receiver labels.

All **4,799 train + 1,224 validation** frozen prompts were reproduced exactly from
raw GSR at the stored decision frames. Identity/label mapping used only present
and past records, `passer_id` and the existing distance/tie rule, not `receiver_id`,
`response`, next action, gap duration or actual future positions.

An inference-only check of **14 real receiver decisions** used the saved social
seed-42 checkpoint with all source arrays truncated at `t` and no target fields
in the model input. Predicted positions and nearest-observed-defender features
were finite for 12 sampled full scenes, one nine-candidate scene and one
ten-defender scene. See [inference evidence](../outputs/trajectory_audit_20261016/receiver_inference_checks.json).
This establishes interface feasibility, not OOF validity or a receiver-accuracy gain.

## Exact compatibility counts

"Usable all-ten" below requires a full two-second frame span, all ten current
candidate tracks, a valid present scene of at most 22 players, and exact agreement
with the frozen prompt. Missing observations could be masked by the network; in
the retained all-ten examples, every currently present player's 51-frame history
in fact remains observed and in the same segment.

| Property | Train, denominator 4,799 | Validation, denominator 1,224 |
|---|---:|---:|
| Exact raw-to-frozen anonymous mapping | 4,799 (100%) | 1,224 (100%) |
| Two-second frame span available | 4,794 (99.8958%) | 1,223 (99.9183%) |
| All ten candidates observed at current frame | 4,799 (100%) | 1,223 (99.9183%) |
| **Usable trajectories for all ten candidates** | **4,794 (99.8958%)** | **1,222 (99.8366%)** |
| Complete history of every currently observed player, all-ten subset | 4,794 | 1,222 |
| Usable all-ten plus all 11 opponent tracks | 4,783 (99.6666%) | 1,217 (99.4281%) |
| Current scene has all 11 opponents, irrespective of history span | 4,788 | 1,219 |
| Strict all-ten failure rate | **5 (0.1042%)** | **2 (0.1634%)** |

If nine candidates plus a masked absent tenth are permitted, **1,223 validation
examples** have usable history for all *observed* players; this must not be
described as forecasting an unobserved tenth candidate. No extrapolated or invented
track is needed. There are 11 training and 5 validation all-ten-usable scenes with
fewer than 11 observed opponents; distances to available defenders can be computed,
but unobserved defenders cannot be declared irrelevant or nonexistent.

| Match | Receiver examples | Usable all-ten histories |
|---|---:|---:|
| 117092 | 600 | 600 |
| 117093 | 691 | 689 |
| 118576 | 680 | 680 |
| 118578 | 608 | 608 |
| 128057 | 612 | 612 |
| 132831 | 820 | 819 |
| 132877 | 788 | 786 |
| 118577 | 528 | 526 |
| 128058 | 696 | 696 |

### All strict failures

| Example | Reason |
|---|---|
| 117093_p1_f17_0 | Half starts fewer than 50 frames before the decision |
| 117093_p2_f20_1138 | Same |
| 132831_p2_f49_1250 | Same |
| 132877_p1_f31_1 | Same |
| 132877_p2_f25_1184 | Same |
| 118577_p1_f37_0 | Same |
| 118577_p1_f27597_324 | Only nine current teammate candidates; 21-player scene |

The six early decisions are a history-span issue, not six broken identity tracks.
Left-padding with explicitly missing history could preserve them, but needs a
tested adapter and must not be described as two seconds of observations. For an
initial controlled experiment, preserve every frozen receiver example and use
explicit forecast-availability masks / original-feature-only fallback rather than
dropping the examples. Report full-denominator and forecast-covered metrics.
Never filter inference examples by their actual future target availability.

Evidence:
[per-example CSV](../outputs/trajectory_audit_20261016/receiver_coverage.csv),
[machine-readable summary](../outputs/trajectory_audit_20261016/audit_summary.json).
Raw identity sidecars are in that same ignored audit directory and contain
association metadata only, not learned features.

## Frame contract: a concrete pre-existing timing problem

The forecasting model's `t` is a zero-based GSR array index, and all its features
are at or before that index. The frozen receiver builder passes BAS `frame`
directly to this index. In all **6,023** examples the resulting current frame is
**20--60 ms after** the BAS event's documented milliseconds from the first released
video frame. For example, `117092_p1_f205_3` uses raw frame 205, **8.200 s**, while
its BAS event is **8.160 s**.

Source: [frame selection, line 235](../src/build_pass_receiver_dataset.py#L235)
and [zero-based decoding, line 21](../src/extract_soccertrack_events.py#L21).
This does not invalidate forecasting causality at its own anchor, but prevents
calling the combined legacy benchmark a strictly pre-contact/BAS-time prediction.
No video synchronization or actual contact time was assessed here.

Choose and record one contract before implementing integration:

1. **Legacy benchmark compatibility:** use exactly frozen GSR index `f`, reproducing
   the existing ranker's state. Evaluate against the 64.3% legacy receiver result
   at the same index, with the 20--60 ms BAS-time limitation prominently stated.
   This audit establishes that this mapping is technically reproducible.
2. **Strict event-time research:** make a separate versioned experiment at the
   latest verified source frame not later than BAS milliseconds (normally
   `floor(position_ms / 40)` under the verified first-frame convention).
   Reconstruct and rebaseline the entire state/anonymous association consistently;
   do not silently shift frozen examples or compare mismatched frame contracts.

Blindly subtracting one frame is not sufficient to guarantee causal BAS timing:
the event milliseconds are fractional relative to the 25 fps grid.

## Association adapter needed

The [existing cache serialization](../src/trajectory_dataset.py#L220) omits
player-ID/track-ID-to-slot metadata. Therefore the NPZs alone cannot safely map
forecast outputs to receiver labels. This audit recovered the exact slot mapping
from raw annotation encounter order and checked 421,007 raw player-frame records.
Do not infer identities by nearest coordinate, shirt color or proximity to the ball.

The adapter should:

1. Verify raw-source/cache provenance and use a persistent verified identity sidecar.
2. Recreate `ACTOR`, `TEAMMATE_1..10`, `OPPONENT_1..11` at the chosen exact frame,
   including the established distance-tie order.
3. Build historical features only through `t`, independently of any future fields.
   The production training `make_window` reads targets and requires future frames,
   so it should not be the inference API.
4. Feed all currently observed players in any consistent order, apply padding
   masks, then gather output displacements by association outside the network.
5. Preserve missing-actor/candidate/defender and history-quality flags explicitly.
   Team/role are allowed inputs; IDs/jerseys are never passed into learned layers.

## Recommended forecast-derived candidate features

For each horizon 0.4/1/2/3 s, define predicted positions
`p_hat_i(h) = p_i(t) + predicted_displacement_i(h)`. Every position on the right
comes from present observations or a causal forecast, never actual future GSR.

| Feature group | Suggested features | Qualification |
|---|---|---|
| Defender separation | Minimum predicted candidate-to-opponent distance; change from current separation; nearest-k distances | Mask unavailable opponents; nearest observed defender is not guaranteed nearest real defender |
| Pressure / space | Predicted opponent counts within 3/5/10 m; minimum separation over the four forecast times | Discrete-horizon forecast proxies, not verified interception opportunities |
| Receiver movement | Predicted dx/dy/speed; motion toward/away from actor; relative motion and separation change | Absolute direction is not automatically attacking direction |
| Candidate/actor geometry | Predicted relative dx/dy/distance; change in passing angle; candidate vs actor displacement | No assumed ball speed or actual receiver label |
| Corridor proxy | Predicted opponent distance to a current-actor / predicted-candidate line segment | Geometric proxy only; no independently verified ball flight or arrival time |
| Availability / reliability | Historical coverage, velocity-support flags, observed defender count, missing candidate mask | Required to distinguish absent information from genuine open space |
| Optional uncertainty | Development/OOF-calibrated position scale or conservative separation margins | Existing calibration is transferred across model refits; do not assume it is valid for every seed or scenario |

Start with low-dimensional metric features and an equal-budget CV-feature control.
A social-forecast gain must beat simply appending independent/CV movement features.

## Critical in-sample bias and leakage-safe procedure

The saved final encoders were supervised on future player positions from all seven
training matches. Freezing those weights does **not** make predictions on those
same matches out-of-sample. Overlapping trajectory windows can directly include a
receiver event's post-pass motion in encoder training targets. Using those forecasts
for ranker training but only genuinely unseen forecasts for validation creates an
optimistic stacking/training-distribution mismatch.
This is **not direct leakage of the two held-out matches into the final encoder**:
those matches were excluded. It is supervised in-sample feature bias on receiver
training rows, plus development contamination if the seven-match encoder is used
on 132877 during receiver selection.

The existing seven-match encoder is also unsuitable for receiver development on
132877: its weights have seen that match. A six-match selection checkpoint did not
fit its weights on 132877, but its duration was selected there. An unbiased outer
development assessment should keep encoder duration/calibration selection inside
the outer fitting partition as well.

### Selection stage: retain 132877 as receiver development

1. Use the six other training matches as the outer receiver-fitting partition.
2. Produce leave-one-match-out forecasts for each of those six matches using
   encoders fit on the other five. Choose epochs, any normalization and calibration
   only within those five, using inner match-level folds. No frames or future targets
   from the forecasted match enter weights, stopping or calibration.
3. Generate 132877 features with a separate encoder fit on the six outer fitting
   matches, with its duration/calibration selected internally. Do not use the
   saved seven-match encoder or tune it on receiver-development trajectories.
4. Fit forecast-feature normalization and the receiver ranker on six-match OOF
   features only; use 132877 to select the augmented receiver configuration.
   Use the same example/fallback policy and tuning budget for each comparator.

### Frozen final stage

1. After choosing the pipeline, generate seven-match leave-one-match-out forecast
   features: every training match is forecast by a model that excluded it from
   fitting, inner selection and calibration.
2. Train the receiver ranker on all seven matches' OOF features. Use training-only
   normalization; keep original labels and frozen example IDs unchanged.
3. For validation matches 118577/128058, use an encoder refit on all seven training
   matches with training-only selected configuration/duration/calibration. No
   held-out trajectories enter fitting or feature normalization.
4. Record fold/checkpoint/feature-source provenance per example. Compare the
   existing-feature ranker, +CV, +OOF independent, and +OOF social features with
   matched receiver seeds and tuning budget. Report full-set and coverage-subset
   top-1/top-3/MRR plus both validation matches.

OOF encoders see five/six matches, whereas final inference sees six/seven; monitor
that residual training-size/distribution difference. Match-level cross-fitting is
preferred to random window folds, which would leak overlapping adjacent history
and future targets. The two validation matches already have published forecasting
and receiver results: they are a fixed reused benchmark, not a fresh untouched
confirmatory test set. Do not repeatedly tune the augmented pipeline on them.

### Why not concatenate frozen encoder embeddings immediately?

The temporal GRU and post-social hidden context could eventually augment the
ranker's actor/candidate representation through an explicit `encode` interface.
However, independently trained OOF encoders have arbitrary latent-axis
orientations. Naively mixing their raw embeddings into one ranker is not guaranteed
to produce a consistent feature space. Prefer physical-coordinate predictions and
derived scalar features for the first OOF experiment. Latent alignment or a
strictly disjoint shared representation would require a separately validated design.

## Go / no-go

**No-go for direct reuse of the existing frozen seven-match encoder as training
features. Conditional go for a new controlled OOF integration experiment after
the adapter, explicit frame contract and fallback tests are implemented.**
Coverage is high enough; no further downloads, videos or human answers are needed.
This is feasibility, not demonstrated receiver-accuracy improvement, pass
completion prediction, interception prediction or tactical decision quality.
