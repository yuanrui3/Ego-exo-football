# Temporal PASS-receiver experiment

Completed 2026-10-15. Exact metrics and run settings are preserved in
[pass_receiver_experiment.json](pass_receiver_experiment.json).

## Scope and evidence

- Task: anonymous temporal player state -> TEAMMATE_1 through TEAMMATE_10.
- Targets are high-confidence next-action pseudo-labels, not verified intended
  receivers. The receiver must be a different same-team player acting within
  50 frames, with no intervening opponent/restart/outcome/block/tackle or
  flagged ambiguity, and must map to a teammate at pass time.
- PASS only; no HIGH_PASS or CROSS. Reconstructed ball coordinates are not
  receiver-label evidence or model inputs. Future events construct targets only.
- Existing temporal player-state serialization and match split were preserved.
  States are at t-10, t-5, and t, with temporally consistent anonymous entities.
- Training matches: 117092, 117093, 118576, 118578, 128057, 132831, 132877.
  Validation matches: 118577, 128058.
- Of 6,590 training and 1,760 validation PASS decisions, 4,799 and 1,224
  respectively survived filtering/history requirements (1,791 and 536 excluded).
- The initial 300-event feasibility audit found 117/150 PASS candidates HIGH,
  versus 4/100 HIGH_PASS and 11/50 CROSS. Its 30-example manual review was
  circumstantial and included weak reconstructed-ball checks, not independent
  receiver ground truth. Those checks are not part of dataset construction.
- The dataset-building sanity review inspected 24 examples across all nine
  matches. Automated checks covered all 6,023 receiver examples for target
  mapping, unchanged temporal state blocks, and past/present-only prompts.

## Configuration and provenance

Each run started fresh from Qwen/Qwen3-4B-Base with the natural, unbalanced
training distribution: one epoch, batch size 1, accumulation 16, learning rate
0.0002, max length 2048. LoRA: rank 16, alpha 32, dropout 0.05, bias none,
targeting q/k/v/o/gate/up/down projections. Loss covers response tokens plus EOS
only. Seeds/data seeds: 42, 7, 123.

All runs recorded parent Git commit
`8b5b5b6b438d6f8e67513682de2d0e48eaa3676b` and dirty working tree `true`:
the receiver scripts had not yet been committed. This report's commit archives
those scripts; it does not retroactively change the recorded training provenance.

Dataset SHA256:

- Train: `4795616041231020a85edded0d25e497a3723cebfc32db9d043c6393b596c730`
- Validation: `063f5f45ea2441a8e0dc06c8c63c0a0f90183589af0c0b398d25938f00a80b11`

Recorded environment: NVIDIA RTX A5000, PyTorch 2.11.0+cu128,
Transformers 5.19.0, PEFT 0.21.2, CUDA 12.8.

| Seed | Optimizer steps | Training runtime (seconds, includes epoch validation) | Mean training loss | Final validation loss |
|---|---:|---:|---:|---:|
| 42 | 300 | 12061.4584 | 0.257749 | 0.206453 |
| 7 | 300 | 11992.3002 | 0.258433 | 0.206283 |
| 123 | 300 | 11990.6291 | 0.263460 | 0.213110 |

Checkpoint interval: 100 steps, retention: latest two. Each run retained
checkpoint-200 and checkpoint-300 and saved a final adapter normally.
The two-replica overnight branch, including ranking evaluations, took about
7 hours 55 minutes; it was not extended artificially to consume ten hours.

## Ranking results

All validation prompts were scored against all ten candidates. The score is the
mean log-probability over candidate tokens plus EOS, excluding prompt tokens.
Each run produced 1,224 predictions with zero non-finite scoring cases.

| Model / baseline | Top-1 | Top-3 | Top-5 | MRR | Mean true rank | Median true rank | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geometric distance ranking | 35.78% | 79.82% | 95.10% | 0.591363 | 2.411765 | 2 | - |
| Seed 42 | 55.07% | 88.48% | 97.63% | 0.724706 | 1.888889 | 1 | 0.278007 |
| Seed 7 | 55.07% | 88.40% | 97.55% | 0.723393 | 1.897059 | 1 | 0.295835 |
| Seed 123 | 51.88% | 87.99% | 97.06% | 0.703324 | 1.967320 | 1 | 0.252107 |
| Mean | 54.00% | 88.29% | 97.41% | 0.717141 | - | - | 0.275316 |
| Population SD (3 seeds) | 1.50 pp | 0.21 pp | 0.25 pp | 0.009785 | - | - | 0.017953 |

The geometric baseline ranks teammates by Euclidean distance computed from
rounded prompt coordinates at t. Its top-1 (35.78%) differs slightly from always
predicting TEAMMATE_1 (437/1224 = 35.70%) because prompt-coordinate rounding can
change near-ties. Existing baseline definitions and dataset labels were not
changed. All three seeds beat both definitions of nearest top-1 and geometric
MRR. The uniform-random validation result at seed 42 was 11.27%.

Seed 42 met both automatic success thresholds (top-1 >= 39.0%, MRR >= 0.610),
selecting fresh receiver replicas rather than the unbalanced action control.
The unbalanced 7-class action control was not run in this branch.

Generated-label agreement with ranked top-1 on the first 20 prompts:
seed 42: 20/20; seed 7: 20/20; seed 123: 18/20. These are diagnostic checks,
not substitutes for the full ranking evaluation.

### Per-target seed-42 results and prediction distributions

| Target | Train support | Validation support | Precision | Recall | F1 | Predicted seed 42 | Predicted seed 7 | Predicted seed 123 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TEAMMATE_1 | 1672 | 437 | 58.82% | 75.51% | 66.13% | 561 (45.83%) | 471 | 503 |
| TEAMMATE_2 | 1233 | 321 | 54.49% | 56.70% | 55.57% | 334 (27.29%) | 364 | 348 |
| TEAMMATE_3 | 855 | 219 | 51.74% | 47.49% | 49.52% | 201 (16.42%) | 212 | 226 |
| TEAMMATE_4 | 499 | 125 | 47.56% | 31.20% | 37.68% | 82 (6.70%) | 124 | 114 |
| TEAMMATE_5 | 260 | 62 | 44.00% | 17.74% | 25.29% | 25 (2.04%) | 29 | 20 |
| TEAMMATE_6 | 145 | 30 | 41.67% | 16.67% | 23.81% | 12 (0.98%) | 15 | 10 |
| TEAMMATE_7 | 77 | 21 | 33.33% | 14.29% | 20.00% | 9 (0.74%) | 9 | 3 |
| TEAMMATE_8 | 33 | 4 | 0.00% | 0.00% | 0.00% | 0 (0.00%) | 0 | 0 |
| TEAMMATE_9 | 16 | 1 | 0.00% | 0.00% | 0.00% | 0 (0.00%) | 0 | 0 |
| TEAMMATE_10 | 9 | 4 | 0.00% | 0.00% | 0.00% | 0 (0.00%) | 0 | 0 |

Exact per-target precision/recall/F1 for all seeds are in the JSON record.

## Interpretation and limitations

The improvement over a fixed distance ranking is reproducible on these two
held-out matches. Every seed also exceeds the stricter automatic success
thresholds. Seed 123 is weaker, so the result is not numerically seed-invariant.
These runs measure optimization-seed variability, not uncertainty across new
match splits or independent ground-truth annotations.

The model is not merely always choosing the nearest teammate: it improves
top-1, top-3, top-5, and MRR. However, predictions are still concentrated on
nearby players. At seed 42, TEAMMATE_1-3 receive 89.54% of predictions versus
79.82% of true targets. Ranks 4-7 have nonzero recall but remain difficult;
ranks 8-10 have only nine validation examples combined and no top-1 predictions.
Macro-F1 is over all ten supported labels and exposes these weaknesses.

The gain supports continuing receiver modeling on the frozen pseudo-label task.
It does not establish reliable intended-receiver prediction for contested,
unsuccessful, high, or crossing passes, or for unseen competitions.

**Verdict: RECEIVER QWEN RESULT IS REPRODUCIBLE.**

**Next recommendation: PROCEED WITH PASS-RECEIVER MODELING.**

## Local artifacts (not committed)

- Seed 42: `outputs/qwen3_4b_pass_receiver`
- Seed 7: `outputs/qwen3_4b_pass_receiver_seed7`
- Seed 123: `outputs/qwen3_4b_pass_receiver_seed123`

Each contains the original run configuration, final adapter, checkpoints,
validation metrics, and per-example predictions. Logs are outside model output
directories. Only code, tests, documentation, and this compact result record
are archived in Git; datasets, predictions, logs, and weights remain local.
