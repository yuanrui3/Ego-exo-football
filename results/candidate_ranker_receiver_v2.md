# Structured PASS-receiver candidate ranker v2

Frozen high-confidence next-action pseudo-labels, not verified intended receivers.
No labels, prompts, split, ball inputs, Qwen results, or geometry baselines changed.

## Protocol

- 4,799 training examples; 1,224 validation examples on matches 118577/128058.
- Selection fit: six training matches (4,011 examples); development: match 132877
  (788 examples). Final fits use all seven original training matches.
- Input: unchanged anonymous player states at frames -10, -5, 0.
- 49 engineered candidate features plus validity masks: relative dx/dy/distance
  at all frames; actor/candidate displacement and velocity; distance changes;
  nearest-opponent distances and counts within 3/5/10 m; teammate distance rank;
  goalkeeper and observed indicators.
- Continuous features/scene coordinates standardized using fitting data only.
  Missing values remain masked; no IDs or future-event metadata enter the model.
- Shared candidate MLP 98 -> 256 -> 128, combined with current actor/candidate
  representations and observed-token scene pooling from a three-layer,
  four-head Transformer (dimension 128, feed-forward 256, dropout 0.1).
  Shared scoring head 512 -> 128 -> 1. **526,209 trainable parameters.**
- AdamW, lr 0.001, weight decay 0.0001, batch 64, gradient clipping 1.0,
  natural unbalanced cross-entropy, deterministic FP32.
- Fresh seed-42 selection for alpha 0/0.25/0.5; maximum 100 epochs, patience 15.
  Training subtracts alpha * log(fitting target frequency) from logits.
  Inference and development scoring use raw logits.
- Select maximum development MRR; ties prefer lower alpha, then earlier epoch.
  All final fits are fixed before held-out prompts/targets are read.

| Alpha | Best epoch | Development MRR | Completed epochs |
|---|---:|---:|---:|
| 0.0 | 21 | 0.773461 | 36 |
| 0.25 | 21 | 0.770600 | 36 |
| 0.5 | 19 | 0.763582 | 34 |

Selected **alpha 0.0, epoch 21**. Fresh final seeds each trained exactly 21
epochs on all seven matches, then evaluated once. No validation retuning.

## Results

| Seed | Fit seconds | Top-1 | Top-3 | Top-5 | MRR | Mean rank | Median rank | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 28.311 | 64.22% | 92.73% | 98.77% | 0.786252 | 1.647876 | 1 | 35.10% |
| 7 | 28.270 | 63.32% | 92.40% | 98.45% | 0.780643 | 1.675654 | 1 | 37.34% |
| 123 | 28.290 | 65.36% | 92.57% | 98.37% | 0.793944 | 1.629085 | 1 | 37.24% |
| Mean | - | 64.30% | 92.57% | 98.53% | 0.786946 | 1.650871 | 1 | 36.56% |
| Population SD | - | 0.84 pp | 0.13 pp | 0.18 pp | 0.005452 | 0.019129 | 0 | 1.03 pp |

Total selection/fits/evaluation: **223.689 seconds**. Zero non-finite cases.

| Comparison | Top-1 | Top-3 | Top-5 | MRR | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Frozen geometry | 35.78% | 79.82% | 95.10% | 0.5914 | - |
| Entity Transformer v1 mean | 36.85% | 80.23% | 95.32% | 0.5984 | 7.78% |
| Candidate ranker v2 mean | 64.30% | 92.57% | 98.53% | 0.7869 | 36.56% |
| Qwen three-seed mean | 54.00% | 88.29% | 97.41% | 0.7171 | 27.53% |

TEAMMATE_1 prediction share: 40.52%, 40.36%, 40.03%, versus true share 35.70%
and v1 shares 85-93%. Recall improves substantially for ranks 2-5; ranks 6-7
remain difficult. Ranks 8-10 have nine validation examples total and zero
correct top-1 predictions. V2 exceeds Qwen mean top-1 by 10.29 pp and MRR by
0.0698. This does not isolate feature causality or establish wider generalization.

Seed-42 evaluation-only coordinate shuffle recomputes all engineered features,
retaining training normalization: top-1 64.22% -> 10.29%, top-3 92.73% -> 29.74%,
MRR 0.786252 -> 0.295260. This disrupts trajectories and is out of distribution;
it demonstrates sensitivity, not causal feature attribution.

**Verdict: STRUCTURED MODEL CLOSES THE GAP.**

## Verification and provenance

- 63 unit tests passed. Frozen hashes/split, selection histories, training-only
  normalization, saved weights/parameter count, aligned finite predictions,
  metrics recomputed from rankings, population SD, and shuffle verified.
- Recorded run Git SHA: `5e1fd126702ce7e0f8848de3ed1f861f761d2a6b`;
  dirty tree true (new v2 implementation uncommitted at training).
- Python 3.14.8; PyTorch 2.11.0+cu128; CUDA 12.8; NVIDIA RTX A5000.
- Train SHA256: `4795616041231020a85edded0d25e497a3723cebfc32db9d043c6393b596c730`.
- Validation SHA256: `063f5f45ea2441a8e0dc06c8c63c0a0f90183589af0c0b398d25938f00a80b11`.
- Reproduce using the [README workflow](../README.md#football-feature-candidate-ranker-v2).
  Detailed metrics, predictions, weights, histories, and the full report remain
  local under `outputs/candidate_ranker_receiver_*`, not in this archive.

Seed SD is not a confidence interval across matches. The final validation set
contains only two matches with correlated events. Feature ablations are needed
before attributing the gain to explicit distance rank, motion, pressure, or context.
