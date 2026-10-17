# Multi-agent football trajectory forecasting

Status: **COMPLETE**. Updated 2026-10-16 03:50:29.
Experiment directory: `C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\outputs\trajectory_forecasting_20261016`.
Elapsed wall time: 3.11 hours.

## Protocol

Raw local GSR pitch coordinates in meters; 51 samples over 2 seconds at 25 fps. Exact targets at +10/+25/+50/+75 frames. Stride 13 frames (0.52 s).
Selection uses six training matches with 132877 as development. Final fits use all seven training matches; held-out 118577 and 128058 are never used for selection.
No ball observations, action labels, identity numbers, track IDs, jerseys or future positions are model features. BAS Pass is used only for a +/-1 s evaluation subset.
Unknown identities are excluded. Duplicate observations are invalidated; missing frames, track/team/role changes and >1.4 m adjacent jumps split continuity. No interpolation.
Only anchor scenes with 18--22 observed players, <=11 per team and >=12 valid 3 s targets are evaluated. This conditions results on continuity and excludes harder missing-track scenes. All models use identical windows and target masks.
CV uses least-squares historical velocity over 0.4 s (>=6 real samples), capped at 12 m/s. Neural models predict metric displacement corrections to this same CV; static ablation has zero velocity and no history. Independent model has no player interaction. Social attention has learned relative geometry/team bias, padding masks and no slot embeddings.
The independent-player baseline is also the no-social-attention ablation: it matches the selected social model's width, block count, learning rate and seeds, with development-selected duration. Attention blocks become per-player residual MLPs; parameter counts are recorded and are not identical.
Coordinate grid/repetition statistics and source size/mtime/cache SHA256 are in cache metadata. Local GSR release commit is unknown (info.version is not a release SHA). Coarse staircase tracking is not independently verified continuous ground truth.
The feasibility pilot has a 1.05 m x-axis / 0.68 m y-axis coordinate lattice and 85.6% identical adjacent positions; instantaneous apparent speeds are therefore misleading. Short-horizon error is limited by quantization. These tracks are usable for coarse movement experiments, not precise ball contact or interception timing.

## Held-out results

| Model | Seeds/runs | MDE 0.4 s | MDE 1 s | MDE 2 s | MDE 3 s | ADE |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 1 | 0.845 +/- 0.000 | 1.967 +/- 0.000 | 3.722 +/- 0.000 | 5.321 +/- 0.000 | 2.964 +/- 0.000 |
| constant_velocity | 1 | 0.760 +/- 0.000 | 1.600 +/- 0.000 | 3.375 +/- 0.000 | 5.440 +/- 0.000 | 2.794 +/- 0.000 |
| social | 3 | 0.490 +/- 0.000 | 0.925 +/- 0.004 | 1.915 +/- 0.011 | 3.070 +/- 0.018 | 1.600 +/- 0.008 |
| independent | 3 | 0.514 +/- 0.004 | 1.031 +/- 0.003 | 2.202 +/- 0.005 | 3.542 +/- 0.007 | 1.822 +/- 0.004 |
| static | 3 | 0.852 +/- 0.005 | 1.851 +/- 0.011 | 3.473 +/- 0.020 | 4.958 +/- 0.028 | 2.783 +/- 0.016 |
| short_history | 3 | 0.593 +/- 0.003 | 1.120 +/- 0.000 | 2.215 +/- 0.004 | 3.426 +/- 0.008 | 1.838 +/- 0.003 |
| probabilistic | 3 | 0.474 +/- 0.002 | 0.919 +/- 0.003 | 1.904 +/- 0.006 | 3.053 +/- 0.008 | 1.587 +/- 0.005 |

### Match / role / PASS-subset ADE (meters)

| Model | Subset | ADE mean +/- seed SD |
|---|---|---:|
| persistence | goalkeeper | 1.387 +/- 0.000 |
| persistence | outfield | 3.121 +/- 0.000 |
| persistence | match_118577 | 2.811 +/- 0.000 |
| persistence | around_PASS | 3.222 +/- 0.000 |
| persistence | match_128058 | 3.126 +/- 0.000 |
| constant_velocity | goalkeeper | 2.066 +/- 0.000 |
| constant_velocity | outfield | 2.867 +/- 0.000 |
| constant_velocity | match_118577 | 2.732 +/- 0.000 |
| constant_velocity | around_PASS | 3.107 +/- 0.000 |
| constant_velocity | match_128058 | 2.859 +/- 0.000 |
| social | goalkeeper | 1.039 +/- 0.007 |
| social | outfield | 1.656 +/- 0.008 |
| social | match_118577 | 1.531 +/- 0.008 |
| social | around_PASS | 1.889 +/- 0.010 |
| social | match_128058 | 1.673 +/- 0.008 |
| independent | goalkeeper | 1.152 +/- 0.004 |
| independent | outfield | 1.889 +/- 0.005 |
| independent | match_118577 | 1.728 +/- 0.004 |
| independent | around_PASS | 2.159 +/- 0.005 |
| independent | match_128058 | 1.922 +/- 0.004 |
| static | goalkeeper | 1.380 +/- 0.012 |
| static | outfield | 2.924 +/- 0.017 |
| static | match_118577 | 2.711 +/- 0.025 |
| static | around_PASS | 3.065 +/- 0.014 |
| static | match_128058 | 2.860 +/- 0.008 |
| short_history | goalkeeper | 1.134 +/- 0.002 |
| short_history | outfield | 1.909 +/- 0.003 |
| short_history | match_118577 | 1.779 +/- 0.004 |
| short_history | around_PASS | 2.082 +/- 0.005 |
| short_history | match_128058 | 1.901 +/- 0.005 |
| probabilistic | goalkeeper | 1.033 +/- 0.008 |
| probabilistic | outfield | 1.643 +/- 0.005 |
| probabilistic | match_118577 | 1.519 +/- 0.007 |
| probabilistic | around_PASS | 1.870 +/- 0.005 |
| probabilistic | match_128058 | 1.660 +/- 0.003 |

All errors are meters. ADE is the equal-weight mean of four horizon MDEs; FDE is 3 s MDE. Detailed JSON contains each match, role, PASS subset and valid-target/player coverage per run. Charts live in the experiment's `charts/` directory.

## Scientific questions

1. **Beyond constant velocity:** Social mean ADE minus CV is -1.194 m. A negative difference favors the neural model; inspect both matches and horizons.
2. **Interactions:** Social minus independent mean ADE is -0.222 m. Matched-seed 30 s block CIs are in JSON. With only two validation matches, population-level statistical significance is not established.
3. **Prohibitive horizon:** No universal acceptable error threshold was specified. Report measured errors through 3 s; do not invent a tactical usability cutoff. Spatial quantization and unverified tracking limit fine interception interpretation.
4. **History vs snapshot:** Static minus full-history ADE is +1.184 m; 0.4 s minus 2 s history is +0.238 m.
5. **Pass feasibility:** At most a movement-encoding foundation, not measured pass feasibility, completion, intention or tactical value. Verified ball state/flight, synchronized contact, receiver intention/control, uncertainty under missing observations and task-specific labels are still needed.

## Reusing the encoder

The shared temporal encoder and social player embeddings could be joined to the existing PASS candidate ranker's actor/candidate features using identity matching outside the network. Fine-tuning or frozen-feature tests must use the existing split and avoid future-target leakage. No receiver ranker has been retrained or tactical benefit measured here.

## Execution and limitations

Completed entries: constant_velocity, independent_seed123, independent_seed42, independent_seed7, persistence, probabilistic_seed123, probabilistic_seed42, probabilistic_seed7, short_history_seed123, short_history_seed42, short_history_seed7, social_seed123, social_seed42, social_seed7, static_seed123, static_seed42, static_seed7.
Best development checkpoints and selected epochs are retained. Final fits use that frozen duration, not validation early stopping. CPU threads=4, loader workers=0, CUDA allocator fraction=0.72, mixed precision, checkpoints every <=100 steps or approximately 60 seconds.
Per-player identities are matching-only; storage order is not encoded. Attention permutation equivariance, target masks, splits and metric arithmetic are tested before launch.
The optional probabilistic head uses isotropic Gaussian NLL; a scalar radial temperature is estimated on development only and transferred without held-out fitting.
Within-match block bootstrap accounts partially for overlapping windows but cannot establish generalization beyond the two held-out matches. Seed SD and match differences are distinct.

## Resume

```powershell
& "C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe" train\run_trajectory_forecasting.py --output "C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\outputs\trajectory_forecasting_20261016" --config "C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2\configs\trajectory_forecasting.json"
```
