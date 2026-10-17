# Trajectory forecasting leakage and methodology audit

Date: 2026-10-16.

**Leakage audit: PASS for application-code input causality and held-out isolation.**
This is not a certificate that upstream GSR annotations were generated causally.
Their original release commit and possible upstream temporal processing remain
**UNVERIFIED**. Future-conditioned window selection is present and quantified below.
Receiver integration is **not ready to train as-is**; see the
[integration feasibility report](receiver_forecast_integration_feasibility.md).

No forecasting weights, completed benchmark results, frozen datasets, video
material or human annotations were changed. No training, downloads or video
processing were performed.

## Evidence and scope

- Read the complete relevant dataset construction, model forward pass, training,
  evaluation, checkpoint selection, calibration and receiver-association paths.
- Matched the current forecasting source/configuration SHA256s to those recorded
  at experiment launch, not merely the repository HEAD (the experiment code is
  uncommitted). Verified all 18 cache SHA256s and their raw-source size/mtime.
- Independently reconstructed **20 uniformly randomly selected eligible windows**
  over the nine-match population, seed `20261016`, without using the production
  window builder to construct the comparison tensors.
- Streamed all 18 raw GSR halves once per successful association extraction using
  `ijson`; retained only needed records and identity mappings. Verified **421,007
  raw player-frame coordinate/team/role records** against cache slots. All 6,023
  receiver prompts were reproduced exactly from raw states, including distance
  ordering/ties and historical coordinates.
- All reconstructed timestamps, masks, current coordinates and target tensors
  matched. The maximum floating-point difference was **0.000024706** in a
  velocity calculation, within the documented `1e-4` tolerance; discrete fields
  and targets matched exactly.
- Erased/changed all post-anchor coordinates, observation flags and segment
  values for each sampled window: every model input remained exactly unchanged.
- Changed targets and target masks supplied to the actual trained social seed-42
  model: maximum forecast change **0.0**.
- Ran inference on **14 actual receiver decisions** with arrays truncated at the
  decision frame and with no target tensors supplied. Forecasts and predicted
  nearest-observed-defender distances were finite, including explicit nine-candidate
  and missing-defender cases. This tests interfaces, not receiver accuracy.
- Reconstructed the completed held-out window list and order independently:
  **21,605 windows**, identical to the saved evaluation records.
- Five new focused deterministic tests passed. Existing production code and
  benchmark measurements were deliberately left unchanged.

Reproducible audit implementation:
[audit utility](../src/audit_trajectory_integration.py),
[focused tests](../tests/test_trajectory_integration_audit.py).
Evidence: [audit summary](../outputs/trajectory_audit_20261016/audit_summary.json),
[20 reconstructions](../outputs/trajectory_audit_20261016/sample_reconstruction.json),
[raw/cache checks](../outputs/trajectory_audit_20261016/raw_cache_verification.json),
[target-free receiver inference](../outputs/trajectory_audit_20261016/receiver_inference_checks.json),
[selection verification](../outputs/trajectory_audit_20261016/saved_selection_verification.json).
The final full scan took **450.5 seconds**, following one failed serialization
attempt in the new audit utility that was corrected; completed associations were
reused. No production defect was silently patched.

## Ten required checks

| Check | Result | Evidence / qualification |
|---|---|---|
| 1. Inputs at or before prediction time | PASS | [History indexing](../src/trajectory_dataset.py#L260) is `[t-50, ..., t]`; current positions, roles and team are at `t`. Velocities use historical observations only. Here `t` is the zero-based GSR anchor, not a BAS event clock. |
| 2. Future positions confined to supervision | PASS for input isolation; sampling caveat | [Targets](../src/trajectory_dataset.py#L282) read `t+[10,25,50,75]`. Future segment availability also controls [window eligibility](../src/trajectory_dataset.py#L325), so the literal statement "future data is used only in targets/loss/evaluation" is not true. This is future-conditioned selection, not a hidden input feature. |
| 3. Exactly 51 history times | PASS | Inclusive `[t-50,t]` has 51 samples and a 2.00 s first-to-last span at 25 fps. Missing times are masked, not interpolated or presented as observations. |
| 4. Exact target offsets | PASS | +10/+25/+50/+75 frames equal +0.4/+1/+2/+3 s. All 20 independent reconstructions passed, using zero-based cache indices. |
| 5. Split isolation | PASS, stage-specific | Selection fits six matches `117092,117093,118576,118578,128057,132831`; development is `132877`; held-out is `118577,128058`. [Final refits](../train/run_trajectory_forecasting.py#L815) intentionally include all seven training matches after selection. Development is therefore not disjoint from the final refit, by design. |
| 6. No validation-fitted normalization | PASS | Fixed divisors are 10/12 for trajectory features and 60/30 for location/geometry. LayerNorm uses the current token's activations. Cache quality percentiles on validation are diagnostics, not learned normalization statistics. |
| 7. IDs association-only | PASS | [Raw identity handling](../src/trajectory_dataset.py#L164) associates `player_id` and splits continuity on track changes. [Model inputs](../src/models/trajectory_forecasting.py#L91) contain no player ID, track ID, jersey or permanent slot embedding. Team and role embeddings are allowed inputs. |
| 8. Masks/coordinate processing causal | PASS for inputs | [Segment construction](../src/trajectory_dataset.py#L63) is prefix-causal: only adjacent present/past records determine a segment start. Historical masks compare against the current segment. Future target masks are used for supervision/metrics, not attention. Future-counterfactual tests passed. |
| 9. Selection excludes held-out metrics | PASS | [Development early stopping](../train/run_trajectory_forecasting.py#L342) and [configuration selection](../train/run_trajectory_forecasting.py#L794) use development ADE only. Recomputed all selected epochs/configurations from saved histories and checked all 15 final-fit configs. Held-out metrics are published before later families finish, but no selection code consumes them, and source/config launch hashes match. |
| 10. Calibration development-only | PASS for split isolation | [Calibration](../train/run_trajectory_forecasting.py#L806) uses the six-match model's best checkpoint on development. Reconstructed temperature **0.9838814062** from saved best-development uncertainty statistics. It is transferred to all seven-match final refits, not fitted on held-out data. Calibration transfer is not evidence of per-seed or held-out calibration. |

The selected social model is width 128, three social blocks, learning rate 0.001,
24 refit epochs; development ADE 1.528633 m. Independent/static/short/probabilistic
selected durations were 13/11/19/19 epochs. None used validation early stopping.

## Selection-bias assessment

Denominator: every stride-13 anchor in `[50, N-75)` for each half, before player
eligibility checks. This yields **96,200 candidate windows** over 1,252,749 source
frames. First two/last three seconds are structurally unavailable for a complete
history/future window; the other frames are also subsampled by the 0.52 s stride.
Those boundary/stride omissions are not included as eligibility exclusions.

| Eligibility rule, applied in code order | Excluded | Fraction of all candidates |
|---|---:|---:|
| Fewer than 18 or more than 22 current players | 30 | 0.03119% |
| More than 11 current players on either team, after count rule | 0 | 0.00000% |
| Fewer than 12 same-segment valid targets at +75 frames, after current rules | 1 | 0.00104% |
| **Total excluded** | **31** | **0.03222%** |
| **Retained** | **96,169** | **99.96778%** |

Independent, overlapping rule counts are 30/22/1 respectively. All 22 team-count
violations already have more than 22 players and are excluded by the first rule;
adding independent counts would double-count them. Of the 30 current-count
exclusions, 8 are below 18 and 22 are above 22.

| Match | Stage | Candidates | Retained | Excluded |
|---|---|---:|---:|---:|
| 117092 | selection/final training | 10,361 | 10,355 | 6 |
| 117093 | selection/final training | 10,605 | 10,605 | 0 |
| 118576 | selection/final training | 11,182 | 11,182 | 0 |
| 118578 | selection/final training | 10,852 | 10,852 | 0 |
| 128057 | selection/final training | 10,595 | 10,595 | 0 |
| 132831 | selection/final training | 10,454 | 10,454 | 0 |
| 132877 | development/final training | 10,524 | 10,521 | 3 |
| 118577 | held-out | 11,128 | 11,128 | 0 |
| 128058 | held-out | 10,499 | 10,477 | 22 |

Only one candidate fails the future-target threshold, in development match
132877. **No held-out window is removed solely by that future rule**. This does
not make future-conditioned selection suitable for online use; it limits its
measured numerical impact on this particular held-out population.

Observed current-player histogram, before exclusions:

| Players | Windows |
|---:|---:|
| 14 | 2 |
| 16 | 4 |
| 17 | 2 |
| 18 | 6 |
| 19 | 13 |
| 20 | 24 |
| 21 | 664 |
| 22 | 95,463 |
| 23 | 15 |
| 24 | 2 |
| 25 | 5 |

### Difficult situations are disproportionately excluded, but are rare here

- Of 221 candidate windows with any missing/discontinuous historical observation,
  **13 (5.88%)** are excluded. Only **0.2163% of retained windows** have any missing
  history, versus **41.94% of excluded windows**.
- Retained/excluded historical observation coverage averages **99.9926%/98.2952%**.
  Mean current counts are **21.9919/21.4839**; mean valid 3 s player counts are
  **21.9872/19.8710**.
- All 22 over-capacity scenes and the one scene with fewer than 12 continuous
  future players are excluded. The present dataset is overwhelmingly clean:
  95,463/96,200 candidate anchors already have 22 current players.
- Target masking additionally conditions each horizon's error on an observed,
  same-segment future target. It cannot measure error where ground truth is
  missing. Small overall window exclusion is not proof of robustness to occlusion,
  track switches, partial observation or deployment conditions.

The 1.600 m social / 1.822 m independent / 2.794 m CV ADE values remain the
unchanged **conditional retrospective benchmark**, not an estimate for arbitrary
online football scenes. Spatial quantization (1.05 m by 0.68 m in the pilot) and
unverified annotation processing further limit sub-meter/contact interpretations.

## Concrete issues and interpretation limits

| Finding | Source and line | Consequence |
|---|---|---|
| Legacy receiver frame is later than BAS first-frame-relative event time | [Receiver target-frame selection, line 235](../src/build_pass_receiver_dataset.py#L235), [GSR decoding, line 21](../src/extract_soccertrack_events.py#L21) | All 6,023 receiver examples use raw zero-based frame `f` from their ID, 20--60 ms after BAS milliseconds. This is a concrete timing-contract defect if called a strictly BAS-time prediction, not future leakage in forecasting at its own anchor. No frozen data was shifted. |
| Cache loses identity-to-slot metadata | [Cache serialization, line 220](../src/trajectory_dataset.py#L220) | Coordinates/segments/team/role are saved but player/track identities are not. An explicit verified association sidecar is needed for receiver integration; this audit recovered it from raw GSR rather than guessing from positions. |
| Eligibility requires future continuity | [Future selection, line 325](../src/trajectory_dataset.py#L325) | Known selection bias; cannot be used to decide which examples receive inference features. Documented and quantified, not silently removed. |
| Social comparison is not parameter-/epoch-matched | [Social/no-social construction, line 78](../src/models/trajectory_forecasting.py#L78) | Social has 469,620 parameters and 24 selected epochs; independent has 269,832 and 13. The ~12.2% ADE improvement supports the tested social architecture, but does not isolate attention from capacity/optimization. |
| Same-match forecaster reuse can contaminate receiver feature generation | [Seven-match refit, line 815](../train/run_trajectory_forecasting.py#L815) | Freezing weights does not remove supervised in-sample trajectory bias. Out-of-fold generation is required for an unbiased stacking experiment. |

## Reproduce without training

```powershell
py -m unittest tests.test_trajectory_integration_audit -v
py src\audit_trajectory_integration.py --minutes 75
```

The audit reuses its own source-verified association sidecars on restart. Evidence
lives in a new ignored output directory; existing forecasting evidence is read-only.
