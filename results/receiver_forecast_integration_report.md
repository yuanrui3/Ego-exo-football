# Social-forecast receiver integration

Status: **TIME_LIMIT**. Elapsed: 12.00 hours.
Experiment: `C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\outputs\receiver_forecast_integration_20261016`. Completed upstream forecast fit sets: 24.
Stage: forecast_independent_199196d577ec5e56ae6e_refit.

## Protocol and caveats

Seven frozen training matches, development 132877, held-out 118577/128058. Upstream forecasts are match-excluded: development fitting rows use leave-one-out within six (five-match models), development uses a six-match model, final fitting uses six-match OOF models and validation uses seven-match refits. Each fit set selects duration on an inner match in its own fitting partition, then refits every allowed fitting match. Predicted matches are forbidden in both fitting and stopping. No forecast uncertainty calibration is used.
The width128/3-layer/lr0.001 forecasting architecture was selected previously using 132877. This is disclosed reused-development context, not nested fresh architecture selection. New epoch stopping never sees the predicted match. Validation benchmarks have previously published results and are not a fresh confirmatory test set.
Primary timing preserves legacy GSR matched frames, approximately 20--60 ms after BAS time. It must not be called strictly pre-pass. Separate as-of sensitivity regenerates base features, candidate order/labels and history at floor(BAS_ms/40), not later than the documented event timestamp; contact/video synchronization and upstream GSR causal generation remain unverified.
Features contain no IDs, actual future positions, receiver target, gap duration, ball speed or future reception time. Inputs end at t; displacements are predicted at .4/1/2/3 s. Movement direction is relative to the current actor-to-candidate line, not an assumed attacking direction. Space is distance to observed defenders, not certified free space.
All rankers have the same shared candidate MLP form and receive the same width128/256 x lr.001/.0003 development grid, MRR early stopping, and seeds42/7/123. Different feature counts change parameter counts; baseline is retuned with the same budget. Selection normalization uses six fitting matches; final normalization uses seven-match OOF features only.
A=base engineered, B=+CV, C=+independent, D=+social, E=D without explicit forecast pressure, F=D without explicit forecast movement. Removal does not eliminate correlated information.
Primary comparison is the same 1,222 covered validation rows, including re-scored historical v2 predictions. Full 1,224-row metrics use same-seed A fallback for uncovered rows; no label or future availability is used for inference filtering. The as-of sensitivity has its own reported observed-target/coverage denominator and must not be conflated with legacy scores.

## Coverage

{"legacy": {"train": 4794, "validation": 1222, "train_total": 4799, "validation_total": 1224, "base_valid_train": 4799, "base_valid_validation": 1224, "reasons": [{"id": "117093_p1_f17_0", "reason": "less_than_2s_history"}, {"id": "117093_p2_f20_1138", "reason": "less_than_2s_history"}, {"id": "132831_p2_f49_1250", "reason": "less_than_2s_history"}, {"id": "132877_p1_f31_1", "reason": "less_than_2s_history"}, {"id": "132877_p2_f25_1184", "reason": "less_than_2s_history"}, {"id": "118577_p1_f37_0", "reason": "less_than_2s_history"}, {"id": "118577_p1_f27597_324", "reason": "missing_candidate_or_candidate_history_or_target"}]}}

## Matched receiver comparison

| Timing / config | Covered n | Top1 mean +/- seed SD | Top3 | Top5 | MRR | Macro F1 | Features | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_A | 1222 | 64.375 +/- 0.635 | 0.9285 | 0.9869 | 0.7886 | 0.3660 | 49 | 74881 |
| legacy_B | 1222 | 62.793 +/- 0.695 | 0.9285 | 0.9831 | 0.7801 | 0.3460 | 112 | 107137 |
| legacy_D | 1222 | 66.148 +/- 0.739 | 0.9403 | 0.9850 | 0.8025 | 0.3717 | 112 | 312577 |
| legacy_E | 1222 | 64.484 +/- 0.134 | 0.9364 | 0.9828 | 0.7914 | 0.3612 | 84 | 283905 |
| legacy_F | 1222 | 65.712 +/- 0.468 | 0.9367 | 0.9847 | 0.7974 | 0.3777 | 80 | 279809 |
| historical_v2 (unchanged) | 1222 | 64.266 +/- 0.806 | 0.9255 | 0.9853 | 0.7868 | 0.3654 | historical | historical |
| historical_engineered_only (unchanged) | 1222 | 64.184 +/- 0.343 | 0.9253 | 0.9845 | 0.7857 | 0.3691 | historical | historical |

Full fallback, per-rank recall, prediction distributions, per-match and subgroup metrics are in the companion JSON. Shuffle controls permute forecast values/masks within covered examples of each validation match, keeping base states unchanged. No seed SD is a population confidence interval; there are only two held-out matches.

## Research answers

1. D minus matched A top1: **+1.773 percentage points**. The old 64.30% score has a different model/full denominator; see separately recomputed historical-v2 common-subset results.
Against unchanged v2 predictions rescored on the common subset: +1.882 pp (historical matched top1 64.266%). This is a contextual reference; A/B/D share the new equal-budget selection protocol.
2. D minus CV B top1: +3.355 pp.
3. Independent-motion comparison PENDING.
4. D minus ablation without explicit forecast pressure: +1.664 pp; retraining comparison, not causal feature importance.
4. D minus ablation without explicit forecast movement: +0.436 pp; retraining comparison, not causal feature importance.
Best measured legacy configuration so far: legacy_D; mean ranker fit 2.3 s, mean validation ranker inference 0.0081 s (excludes forecast training/inference, separately recorded).
5. Compute cost: 12.00 wall hours so far, 24 completed fit sets. Per-fold checkpoint/epoch/parameter provenance and per-run inference cost are saved; no padded runtime.
6. Verdict: **INCONCLUSIVE**. A positive mean alone does not establish robust tactical value.
7. Next-actor pseudo-label accuracy is not pass completion, intended receiver, pass feasibility or tactical value. Verified ball flight/contact, receiver control/intention, defender reach, uncertainty under missing observations and task-specific evaluation are still needed.

## Historical matched-subset reference

See `historical_v2_matched_subset.json` under the experiment directory. It rescored the unchanged existing full-v2 predictions on exactly the covered legacy rows, with no retraining.

## Resume

```powershell
& "C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe" train\run_receiver_forecast_integration.py --output "C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\outputs\receiver_forecast_integration_20261016" --config "C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\configs\receiver_forecast_integration.json"
```

Blocker: Training budget reached; batch/optimizer/RNG saved
