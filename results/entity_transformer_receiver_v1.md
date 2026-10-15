# Entity Transformer receiver baseline v1

Completed 2026-10-15. Frozen PASS-only high-confidence next-action pseudo-labels,
not verified intended-pass ground truth. No datasets, prompts, label rules,
match splits, Qwen code/results, or geometric-baseline definitions were changed.
No ball data was used.

## Model and protocol

- Input: [3, 22, 8] anonymous player tensor at t-10, t-5, t frames.
  Features: x/52.5, y/34, observed mask, actor/teammate/opponent indicators,
  goalkeeper/player indicators. Slots follow anonymous labels across time.
  Missing slots/coordinates are zero-masked; no permanent or match IDs are features.
- 435,585 trainable parameters; projection dimension 128; learned time, slot,
  and relation embeddings; 3 encoder layers, 4 heads, feed-forward 256,
  dropout 0.1; flattened 66-token attention.
- Shared 259 -> 128 -> 1 MLP scores current actor/teammate representations plus
  normalized relative x/y and Euclidean distance. Missing candidates receive
  a finite logit of -10000 and unobserved tokens are masked as attention keys.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 64,
  gradient clipping 1.0; natural unbalanced cross-entropy; deterministic FP32.
- Epoch selection: seed 42; train on six matches (4,011 examples), develop on
  match **132877** (788 examples). Maximum 50 epochs, highest development MRR,
  earliest epoch breaks ties.
- Selected **epoch 49**, development MRR **0.5705352107001853**.
  Selection runtime: **52.210 seconds**.
- Fresh final fits at seeds 42, 7, 123 used all seven training matches
  (4,799 examples) for exactly 49 epochs. Each was evaluated once on the same
  two validation matches (1,224 examples); no final-validation tuning.
- Training matches: 117092, 117093, 118576, 118578, 128057, 132831, 132877.
  Validation matches: 118577, 128058.

## Results

Ranking metrics reuse the existing Qwen receiver metric function unchanged.
All ten labels have validation support and are included in macro-F1.
Each seed produced 1,224 predictions with zero invalid/non-finite scoring cases.

| Seed | Train seconds | Top-1 | Top-3 | Top-5 | MRR | Mean rank | Median rank | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 55.716 | 38.07% | 80.72% | 95.42% | 0.606825 | 2.348856 | 2 | 9.04% |
| 7 | 53.458 | 35.95% | 79.82% | 95.42% | 0.592230 | 2.404412 | 2 | 7.50% |
| 123 | 52.462 | 36.52% | 80.15% | 95.10% | 0.596268 | 2.392157 | 2 | 6.82% |
| Mean | - | 36.85% | 80.23% | 95.32% | 0.598441 | - | - | 7.78% |
| Population SD | - | 0.90 pp | 0.37 pp | 0.15 pp | 0.006153 | - | - | 0.93 pp |

| Baseline / model mean | Top-1 | Top-3 | Top-5 | MRR | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Frozen geometric ranking | 35.78% | 79.82% | 95.10% | 0.5914 | - |
| Entity Transformer v1 | 36.85% | 80.23% | 95.32% | 0.5984 | 7.78% |
| Qwen | 54.00% | 88.29% | 97.41% | 0.7171 | 27.53% |

The structured control exceeds geometric top-1 by about 1.06 percentage points
and MRR by 0.0071; statistical significance was not established. Qwen exceeds
this control by about 17.16 top-1 percentage points and 0.1187 MRR.

### Target behavior

| Target | Validation support | Seed-42 recall | Predictions seed 42 / 7 / 123 |
|---|---:|---:|---|
| TEAMMATE_1 | 437 | 91.30% | 1041 / 1073 / 1139 |
| TEAMMATE_2 | 321 | 15.58% | 138 / 134 / 83 |
| TEAMMATE_3 | 219 | 6.85% | 42 / 17 / 2 |
| TEAMMATE_4 | 125 | 1.60% | 3 / 0 / 0 |
| TEAMMATE_5 | 62 | 0.00% | 0 / 0 / 0 |
| TEAMMATE_6 | 30 | 0.00% | 0 / 0 / 0 |
| TEAMMATE_7 | 21 | 0.00% | 0 / 0 / 0 |
| TEAMMATE_8 | 4 | 0.00% | 0 / 0 / 0 |
| TEAMMATE_9 | 1 | 0.00% | 0 / 0 / 0 |
| TEAMMATE_10 | 4 | 0.00% | 0 / 0 / 0 |

TEAMMATE_1 receives 85.05%, 87.66%, and 93.06% of predictions, versus a true
share of 35.70%. Ranks 2-3 have poor recall; ranks 4-10 are effectively
unlearned. Ranks 8-10 also have only nine validation examples combined.

## Seed-42 coordinate-shuffle ablation

Evaluation only, no retraining. Seed 42 shuffles complete x/y pairs among
observed entities independently per sample/state. Labels, targets, masks,
and role/relation indicators remain unchanged; missing coordinates stay zero.

| Metric | Original | Shuffled | Change |
|---|---:|---:|---:|
| Top-1 | 38.07% | 14.46% | -23.61 pp |
| Top-3 | 80.72% | 36.19% | -44.53 pp |
| MRR | 0.606825 | 0.338071 | -0.268755 |

The model is geometry-sensitive but heavily favors the nearest-ranked entity.
Shuffling also disrupts temporal trajectories and is out of distribution:
this does not establish robust spatial reasoning. Qwen has a clear measured
advantage over this fixed control; scale/pretraining effects cannot be isolated
from architecture, representation, and optimization differences.

**Verdict: QWEN OUTPERFORMS ENTITY TRANSFORMER.**

## Reproducibility and validation

- Recorded Git SHA: `693a06c1ad8ac1a0e5a09193d061ee3510a7eaf5`;
  dirty working tree: true (new v1 source/docs were uncommitted during training).
  The later archive commit does not retroactively change this provenance.
- Python 3.14.8; PyTorch 2.11.0+cu128; CUDA 12.8; NVIDIA RTX A5000.
- Train SHA256: `4795616041231020a85edded0d25e497a3723cebfc32db9d043c6393b596c730`.
- Validation SHA256: `063f5f45ea2441a8e0dc06c8c63c0a0f90183589af0c0b398d25938f00a80b11`.
- All 55 unit tests passed, including 11 new tests and a train/evaluate smoke test.
  All 6,023 examples parsed successfully, with fixed anonymous slots.
- Prediction IDs/targets aligned to validation, rankings/metrics were recomputed,
  dataset hashes were unchanged, and saved full-fit best/final weights were equal.

Reproduce with the command in the [README](../README.md#compact-entity-transformer-control).
The parser, model, training script, and tests are archived with this report.
Original weights, histories, run configs, per-example predictions, exact metric
JSONs, and the longer report remain local under `outputs/`. None are committed.
The final-fit best weights equal final weights, with no final-validation
checkpoint selection. V2 was not started as part of this archival task.
