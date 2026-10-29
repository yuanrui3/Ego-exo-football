# Football Backbone v0.6 Handoff

**Corrected status: PARTIAL completion of the original v0.6 assignment.** The delivered subset passed its validators, but that is not equivalent to completing the original requirements. The local stage plan incorrectly substituted 10- and 5-minute replay windows for two complete tracking halves and did not exhaust the original expansion backlog. The previous completion claim is withdrawn. No GNN training was resumed; no annotations were changed; no commit or push was made. The fixed deadline was 2026-10-22 00:47:47.841 UTC; it has passed and is not reset by this correction.

## Original requirements and measured status

The original attachment was reread for this correction: `C:\Users\flexadmin\Downloads\copilot_backbone_v0_6_autonomous_20h.md`, SHA-256 `4875b886856b29dfe3a98ba400896b61b2145397acf34f9994e64b48c0befcb8`. Requirement text below is quoted from it; evidence is from the existing artifacts, not a fresh pipeline rerun. Superseded reports are archived under [completion_claim_correction_20261022](../outputs/backbone_v0_6_20261021/completion_claim_correction_20261022/).

| Original wording | Measured result | Status |
|---|---|---|
| "Run twelve REAL 120-second sequences" | 12 sequences; 7,200 unique sampled frames, 14,400 actor-steps | PASS for this workload |
| "target 36,000 actual inference frames covering 120 unique source minutes" | 36,000 fitting frames / 120 fitting source minutes, plus 6,000 development frames / 20 minutes | PASS for pixel ingestion, not video-based action inference |
| "Feed accepted detector-derived pitch positions through the public observation adapter." | Neither attempted mapping passed; no accepted detector-derived pitch observations reached the backbone | BLOCKED; permitted pixel-only fallback retained |
| "Otherwise fit one compact existing structured action-teacher architecture." | No compact teacher was fitted; no match-excluded teacher probabilities were generated | NOT_RUN; learning path unimplemented |
| "If compatible teacher targets cannot be produced correctly within the stage budget, mark distillation BLOCKED and continue with the corruption-training control." | One C1 corruption-trained student fitted and validated; C2 not run | PASS for authorized fallback only; not proof teacher training is infeasible |
| "Run the integrated system on two complete available retrospective tracking halves, with four configured actors per half" | Only 10- and 5-minute windows, 4,500 sampled frames / 15 nominal source minutes / 18,000 actor-steps | FAIL against original duration requirement; full-half operation unverified |
| "Storage chunks must carry memory state" | Safe-prefix reconstruction passed for one actor over 3,000 samples; no complete-half chunked workload delivered | NOT_VERIFIED at required scale |
| "Evaluate a fixed development-only subset" | Training comparison used development; the expanded seven-condition audit used retrospective matches and does not substitute for the requested development stress matrix | NOT_VERIFIED for the full requested matrix |
| "Expand successful video ingestion across the remaining locally available training halves, initially ten minutes per uncovered half." | Initial 140-minute corpus delivered; remaining-half expansion not established | NOT_VERIFIED |
| "Build a lazy index of up to 100,000 distinct fitting actor/time windows" | Historical 4,456 unlabeled windows exist; no new lazy index meeting this expansion requirement was delivered | NOT_RUN |
| "Remove the single worst measured integration bottleneck" | Throughput measured, but no corresponding before/after optimization benchmark with prediction parity delivered | NOT_RUN |
| "Run the resulting source-independent adapter on another available recording/configuration without Python edits." | Existing external-adapter parity demonstrated; separate expansion result not established | NOT_VERIFIED |
| "Build a small review-only diagnostic pack" | Calibration diagnostics exist; complete rejected-region/identity-break/unknown-role pack not established | NOT_VERIFIED |
| "Generate acceptance from output contents and event logs, not hand-edited success flags." | Individual artifact validators ran; aggregate acceptance was manually assembled and overclaimed overall completion | FAIL for original aggregate-acceptance requirement |
| "Plan for 16–19 hours of useful development and computation." | Recorded acceptance after 4h02m20s; final status after 4h03m21s | NOT_MET; unfinished work, not elapsed time alone, invalidates COMPLETE |

Other Stage F evidence remains narrowly scoped: 69 targeted tests and browser checks passed, but they do not certify unexecuted workloads. A v0.6 clean source-only audit, complete model/source registry and single execution/resume/verification orchestration were not established by the submitted acceptance. These remain unverified, not implicit passes.

## Integration result

- **Stage A — sustained replay:** 12 causal 120-second sequences passed. The batch covers 7,200 unique source frames (24 source minutes), 14,400 actor-steps, 14,160 G1 calls and 12 event-conditioned action calls. External-observation parity and safe-prefix recovery are recorded in the Stage A artifacts.
- **Stage B — video ingestion:** passed on 42,000 unique sampled frames: 36,000 fitting and 6,000 development, 140 source minutes total, 280 chunks and 251,464 detections. Measured average processing throughput was 4.01 frames/s. Video frames and detections are not pitch-calibrated player observations.
- **Calibration bridge remains blocked and undeployed:** the held-out annotation-assisted mapping measured 8.27 m median and 20.30 m p90 pitch error against 2 m/5 m limits; the local homography fit failed. No pitch mapping, team assignment, or physical-camera claim is deployed.
- **Stage C — authorized C1 fallback, not completed teacher transfer:** trained on 12,038 fitting examples from `117092`, `117093`, `118576`, `118578`, `128057`, `132831`; development used `132877` (2,158 examples). Early stopping selected epoch 1 of 3. Optimizer/RNG resume state and adapter parity passed. Best checkpoint: [`best.pt`](../outputs/backbone_v0_6_20261021/stage_c/robust_training/best.pt), SHA-256 `7d3ea78f20bac82f6d3291e4a446ac728820dce9feef6759260c5897d4e73f84`; resume state [`last.pt`](../outputs/backbone_v0_6_20261021/stage_c/robust_training/last.pt), SHA-256 `c3fd58eeba76707b906ab6f6b2c1c62499edc59f03d39ba9e9f93cad4aee2b7f`. Missing reusable teacher code/checkpoints did not establish that the authorized compact-teacher fit could not be implemented.
- **Stage D — partial long replay and recovery:** two short retrospective jobs passed their local validators: 4,500 unique source frames (15 nominal source minutes), 18,000 actor-steps, 8 fully covered actor sequences and 9 event queries over PASS, HIGH_PASS, DRIVE and CROSS. These are not two complete halves. One action session served all queries. The separate interruption probe stopped after sample 1,500, reconstructed memory from the released observation prefix, and matched all 3,000 outputs to uninterrupted control with zero numeric difference and zero duplicate frames.
- **Additional retrospective robustness audit:** passed on 3,796 common evaluation examples from `118577` and `128058`, with no fitting, checkpoint selection or tuning. Target-independence logits were identical (maximum difference 0.0); corrupted inputs remained target-free and causal. This additional audit did not exhaust or replace the original Stage E backlog.
- **Stage F — verification of delivered subset:** local Stage A, B, C, D and timing validators passed; the additional audit validator passed; 69 targeted backbone and resume regressions passed; Pylance reported no errors in the checked Python files. Browser assertions passed for the tested viewer paths. These results do not establish original-assignment completion.

## Robustness comparison and deployment decision

These are matched comparisons: both checkpoints use the same evaluation examples and protocol within each condition. The robust checkpoint is **not a universal upgrade**.

| Held-out condition | Baseline macro-F1 | Robust macro-F1 | Change |
|---|---:|---:|---:|
| Clean | 0.5188 | 0.5063 | -0.0124 |
| Training-regimen corruption | 0.5087 | 0.5010 | -0.0077 |
| Intensified stress | 0.4702 | 0.4646 | -0.0055 |
| Provider dropout only | 0.4752 | 0.5020 | +0.0268 |
| Correlated gap only | 0.5132 | 0.4992 | -0.0140 |
| Coordinate noise only | 0.4913 | 0.4855 | -0.0058 |
| Identity break only | 0.5188 | 0.5063 | -0.0125 |

The provider-dropout gain is concentrated in match `118577` (macro-F1 0.4242 → 0.5057); it does not replicate on `128058` (0.5017 → 0.4974). **Keep the original v0.5 S4 as the default.** Retain robust S4 only as an explicitly selected experimental alternative; there is no validated automatic dropout-switching rule. The existing v0.5 batch and Stage A configs select the original S4; the preserved Stage D experiment explicitly selects robust S4. The adapter requires an explicit checkpoint and does not perform automatic checkpoint switching. Rare-class metrics are uncertain: HEADER has only 5 evaluation examples; DEFENSIVE_INTERVENTION has 158 and its robust-model recall falls from 0.1519 to 0.0633 on clean inputs. The classifier's softmax is uncalibrated and is not tactical value or an action ranking.

The identity-break policy changes one non-actor segment at the first model-history sample. It does not establish recovery from an identity switch after a player's memory has already formed. The historical condition and its metrics are preserved; no post-memory-switch recovery claim is made.

Development comparison was directionally consistent with this caution: robust macro-F1 rose 0.0210 on corrupted development inputs (0.4151 → 0.4361) but fell 0.0168 on clean development inputs (0.4766 → 0.4598).

## Replay and interface

Open the [`long-sequence replay viewer`](../outputs/backbone_v0_6_20261021/stage_d/stress_replay_long_v06/viewer.html). The browser check exercised queries from both sequences: `DRIVE` at frame 5191 and `PASS` at frame 764. Recorded annotations are behind a separate evaluation-only toggle; enabling the source/evaluation overlay did not change player-facing action status or scores. The full-state receiver panel remains separately labeled exo-reference-only, with optional P0 disabled. Ball and possession remain unknown. Scores are explicitly uncalibrated; passing geometry remains optional.

The viewer is a retrospective tracking prototype with synthetic visibility, not raw-video perception, gaze, a continuous action detector, or a best-action recommender. The selected actor is not assumed to possess the ball. GNN research remains `PAUSED_FOR_INTEGRATION`; replay inference is not GNN training.

## Failures and recoveries preserved

- The first 10-minute batch attempt was rejected safely by the preflight's 1,500-sample contract; its output was preserved. A bounded, explicit retrospective-only 3,000-sample mode was added for the continuous-memory replay.
- The first completed-replay validation expected the non-batch schema. The validator was corrected to read the batch schema, then the replay passed. The historical stress config and selection evidence were preserved; the long config is separately versioned.
- The VS Code test runner did not discover this pytest suite. Direct execution with the selected Python 3.14 interpreter passed the focused v0.6 regressions and the final 69-test suite.

## Evidence

- Machine acceptance record: [`acceptance_record.json`](../outputs/backbone_v0_6_20261021/acceptance_record.json)
- Run progress, stage plan and decisions: [`progress.json`](../outputs/backbone_v0_6_20261021/progress.json), [`stage_plan.json`](../outputs/backbone_v0_6_20261021/stage_plan.json), [`decision_log.jsonl`](../outputs/backbone_v0_6_20261021/decision_log.jsonl)
- Stage B: [`video_validation.json`](../outputs/backbone_v0_6_20261021/stage_f/video_validation.json), [`regional_mapping_117092.json`](../outputs/backbone_v0_6_20261021/stage_b/regional_mapping_117092.json)
- Stage C: [`fresh_robustness_validation.json`](../outputs/backbone_v0_6_20261021/stage_f/fresh_robustness_validation.json), [`training_result.json`](../outputs/backbone_v0_6_20261021/stage_c/robust_training/training_result.json)
- Stage D: [`stress_validation.json`](../outputs/backbone_v0_6_20261021/stage_f/stress_validation.json), [`resume_validation.json`](../outputs/backbone_v0_6_20261021/stage_d/resume_probe_long_v06/resume_validation.json)
- Stage E: [`heldout_robustness_evaluation.json`](../outputs/backbone_v0_6_20261021/stage_e/heldout_robustness_v06/heldout_robustness_evaluation.json), [`heldout_robustness_validation.json`](../outputs/backbone_v0_6_20261021/stage_e/heldout_robustness_v06/heldout_robustness_validation.json)
- Browser evidence: [`browser_validation.json`](../outputs/backbone_v0_6_20261021/stage_f/browser_validation.json)

## PowerShell reproduction

Run from PowerShell. The commands use the tested Python 3.14 interpreter. Use fresh output paths; evaluators and resume probes intentionally refuse to overwrite existing results. These commands validate or replay existing artifacts; **they do not train**.

```powershell
$Python = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
Set-Location 'C:\Users\flexadmin\Downloads\football_tactical_llm_starter_v2\football_tactical_llm_starter_v2'

# Recheck the sustained Stage A batch.
& $Python -m src.backbone.validate_sustained_batch_v06 `
  --config outputs\backbone_v0_6_20261021\stage_a\sustained_batch_v2.json `
  --summary outputs\backbone_v0_6_20261021\stage_a\sustained_batch\batch_summary.json `
  --output outputs\backbone_v0_6_20261021\stage_f\recheck_sustained.json

# Recheck Stage B ingestion and Stage A causal timing reconciliation.
& $Python -m src.backbone.validate_video_ingest_v06 `
  --config outputs\backbone_v0_6_20261021\stage_b\video_config_v06.json `
  --output outputs\backbone_v0_6_20261021\stage_b\video_corpus `
  --root . `
  --report outputs\backbone_v0_6_20261021\stage_f\recheck_video.json
& $Python -m src.backbone.validate_timing_reconciliation_v06 `
  --corpus-root outputs\backbone_v0_5_20261020\corpus `
  --report outputs\backbone_v0_6_20261021\stage_f\recheck_timing.json

# Reproduce the historical robust-S4 experiment (not the default model).
# These jobs cover 15 source minutes, not two full halves.
$ReplayOutput = 'outputs\backbone_v0_6_20261021\stage_d\reproduction'
$ReplayReport = 'outputs\backbone_v0_6_20261021\stage_d\reproduction_validation.json'
& $Python -m src.backbone.batch `
  --config outputs\backbone_v0_6_20261021\stage_d\stress_config_long_v06.json `
  --output $ReplayOutput
& $Python -m src.backbone.validate_stress_v06 `
  --phase completed `
  --config outputs\backbone_v0_6_20261021\stage_d\stress_config_long_v06.json `
  --output $ReplayOutput `
  --report $ReplayReport

# Reproduce the controlled 1,500-sample interruption and safe-prefix resume.
$ResumeOutput = 'outputs\backbone_v0_6_20261021\stage_d\resume_probe_reproduction'
$Baseline = 'outputs\backbone_v0_6_20261021\stage_d\stress_replay_long_v06\batch_summary.json'
& $Python -m src.backbone.replay_resume_probe_v06 `
  --phase interrupt `
  --config outputs\backbone_v0_6_20261021\stage_d\stress_config_long_v06.json `
  --job stress_118577_1st `
  --output $ResumeOutput `
  --baseline-summary $Baseline
& $Python -m src.backbone.replay_resume_probe_v06 `
  --phase resume `
  --config outputs\backbone_v0_6_20261021\stage_d\stress_config_long_v06.json `
  --job stress_118577_1st `
  --output $ResumeOutput `
  --baseline-summary $Baseline

# Reproduce and independently validate the frozen Stage E audit.
$EvaluationOutput = 'outputs\backbone_v0_6_20261021\stage_e\reproduction'
& $Python -m src.backbone.evaluate_robustness_v06 `
  --phase evaluate `
  --output $EvaluationOutput
& $Python -m src.backbone.evaluate_robustness_v06 `
  --phase validate `
  --output $EvaluationOutput

# Final focused regression command (69 tests passed for this exact selection).
& $Python -m pytest `
  tests\test_backbone.py `
  tests\test_backbone_ingestion.py `
  tests\test_backbone_v0_3.py `
  tests\test_backbone_v0_4.py `
  tests\test_backbone_v05.py `
  tests\test_backbone_v06_video.py `
  tests\test_backbone_v06_robust_action.py `
  tests\test_dynamic_football.py::test_resume_optimizer_rng_exact_equality `
  -q
```

The executed C1 fallback fit is complete; teacher fitting, match-excluded target generation and the matched distillation student were not executed. The commands above reproduce only delivered work, not the missing original requirements. This correction starts no training and does not reset the expired budget. The dynamic-GNN queue remains paused.

## Next completion targets

1. Implement the compact multi-action teacher path with three match-excluded folds, keyed probabilities and a matched student comparison; preserve original S4 as the default unless development evidence justifies replacement.
2. Complete two full tracking halves with chronological state across bounded shards, explicit missing-data fallback/abstention, the development stress matrix and the fitting-only lazy unlabeled-window index. Measure and remove one bottleneck with before/after timing and prediction parity.
3. Attempt one narrowly supported region and continuous video-to-pitch segment under a declared held-out acceptance gate. Rejected projection remains rejected; unknown identity/team/role must not be silently filled to force inference.

Future acceptance must retain original wording, measured evidence and PASS/FAIL/BLOCKED/NOT_RUN/NOT_VERIFIED per requirement. An authorized fallback can pass its branch without turning a missing capability into a passed requirement.
