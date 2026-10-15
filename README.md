# Football Tactical LLM System

This repository contains our football tactical modeling pipeline for turning SoccerTrack event annotations and game-state snapshots into a supervised fine-tuning dataset for a tactical action model.

Pipeline:

`SoccerTrack BAS + GSR -> canonical decision JSONL -> structured text -> match-level SFT split -> baselines -> Qwen3-4B LoRA -> classification metrics`

## System overview

Our system is designed to:

- ingest BAS event annotations and GSR state data from SoccerTrack-v2
- align events to the correct frame using the BAS `frame` field with a ±1-frame tolerance
- create canonical decision records that capture the action, actor context, and player state
- validate data quality before training
- build train/validation datasets across multiple matches
- compare against a majority baseline
- fine-tune a Qwen3-4B model with LoRA adapters
- evaluate the model using task-relevant classification metrics instead of raw language loss

## Why this repository exists

The extractor supports the released SoccerTrack-v2 BAS format, where events are stored in an `actions` array with uppercase labels, as well as annotation-style BAS files. It also handles GSR files in SoccerNet-COCO-style format and aligns events to the BAS `frame` field with a ±1-frame tolerance.

## Project structure

- `src/extract_soccertrack_events.py` — builds canonical decision records from BAS + GSR inputs
- `src/inspect_dataset.py` — inspects the generated JSONL dataset
- `src/serialize_state.py` — serializes decision records into state-aware structured text
- `src/make_sft_dataset.py` — creates train/validation splits for supervised fine-tuning
- `src/make_temporal_sft_dataset.py` — builds a separate, temporally aligned anonymous SFT dataset
- `src/majority_baseline.py` — establishes a simple baseline for comparison
- `train/train_lora_qwen.py` — fine-tunes a Qwen3-4B model with LoRA
- `src/evaluate_action_model.py` — evaluates model predictions with classification metrics
- `src/audit_pass_receivers.py` — audits event-sequence receiver candidates without changing datasets
- `src/build_pass_receiver_dataset.py` — builds separate high-confidence PASS-receiver datasets and geometric baselines
- `train/train_pass_receiver_qwen.py` — trains a fresh, unbalanced temporal receiver model
- `src/evaluate_pass_receiver_model.py` — ranks all ten receiver candidates by conditional token scores
- `results/` — compact, verified experiment summaries; no datasets or model weights
- `data/` — stores extracted and prepared datasets
- `outputs/` — stores trained adapters and evaluation artifacts

## Requirements

Install the Python dependencies:

```powershell
py -m pip install -r requirements.txt
```

The project depends on:

- PyTorch
- Transformers
- Datasets
- PEFT
- Accelerate
- ijson
- NumPy
- scikit-learn

## Quick start

### 1) Extract a match decision dataset

```powershell
py src\extract_soccertrack_events.py `
  --bas "D:\SoccerTrack\BAS\117093\117093_12_class_events.json" `
  --gsr "D:\SoccerTrack\GSR\117093\117093_1st.json" `
  --match-id 117093 `
  --period 1 `
  --output data\117093_1st_decisions.jsonl
```

This creates a per-event decision file that includes:

- match and period metadata
- frame and timestamp information
- matched GSR state
- event label
- player and team context (possession is unknown and left unset)

### 2) Check dataset quality before training

```powershell
py src\inspect_dataset.py --decisions data\117093_1st_decisions.jsonl
py src\serialize_state.py data\117093_1st_decisions.jsonl --mode anonymous
```

Before any training run, verify the key quality signals:

- `state match rate`
- `actor-presence rate`

If these rates are weak, the extracted dataset should be reviewed before model training.

### 3) Combine multiple matches and build the SFT split

```powershell
py src\make_sft_dataset.py `
  --input data\all_decisions.jsonl `
  --train data\train.jsonl `
  --val data\val.jsonl
```

The split is match-level and preserves the existing train/validation assignment. Anonymous serialization is the default for SFT generation, and records without a valid actor state are excluded. Extraction also excludes events whose actor is missing from the matched GSR frame by default. Use `--include-actor-missing` on extraction only when retaining such records for diagnostics.

### 4) Run the reference baseline

```powershell
py src\majority_baseline.py --train data\train.jsonl --val data\val.jsonl
```

### 5) Fine-tune the tactical model

```powershell
py train\train_lora_qwen.py `
  --train data\train.jsonl `
  --val data\val.jsonl `
  --output outputs\qwen3_4b_football
```

This uses Qwen3-4B with LoRA adaptation for the tactical action classification task.

### 6) Evaluate the trained adapter

```powershell
py src\evaluate_action_model.py `
  --val data\val.jsonl `
  --adapter outputs\qwen3_4b_football `
  --predictions outputs\qwen3_4b_football\val_predictions.jsonl
```

### Optional: build the short-history temporal experiment

This creates separate temporal JSONL files while reusing the existing match-level
train/validation assignment. GSR is streamed and states are taken only from
exact frames `t-10`, `t-5`, and `t`; no adjacent-frame fallback or future frame
is used. Decisions missing any required frame are skipped. Anonymous player
labels are assigned at event time and held fixed across all three states.

```powershell
py src\make_temporal_sft_dataset.py `
  --decisions data\all_decisions.jsonl `
  --train-reference data\train.jsonl `
  --val-reference data\val.jsonl `
  --gsr-dir data `
  --train data\train_temporal.jsonl `
  --val data\val_temporal.jsonl

py train\train_lora_qwen.py `
  --train data\train_temporal.jsonl `
  --val data\val_temporal.jsonl `
  --output outputs\qwen3_4b_football_temporal `
  --epochs 1

py src\evaluate_action_model.py `
  --val data\val_temporal.jsonl `
  --adapter outputs\qwen3_4b_football_temporal `
  --predictions outputs\qwen3_4b_football_temporal\val_predictions.jsonl
```

An optional mild-balancing screen can continue from that trained adapter using
inverse-square-root class sampling for half an epoch. This is a screening run,
not a clean comparison against training with balancing from initialization.
It writes a separate adapter and does not alter the temporal baseline:

```powershell
py train\train_temporal_mild_balance.py `
  --train data\train_temporal.jsonl `
  --val data\val_temporal.jsonl `
  --start-adapter outputs\qwen3_4b_football_temporal `
  --output outputs\qwen3_4b_football_temporal_mildbalance `
  --examples 7568 `
  --lr 2e-4
```

### Clean 7-class temporal + mild-balance run

Direct-action labels only (PASS, HIGH_PASS, DRIVE, CROSS, SHOT, HEADER,
DEFENSIVE_INTERVENTION = Block + Tackle). Out, Goal, Throw In and Free Kick are
excluded. The original label is kept as `source_label` metadata.

```powershell
py src\recode_primary_actions.py

# Fresh run from Qwen/Qwen3-4B-Base (no adapter), 1 epoch, seed 42,
# replacement sampling with weight = 1/sqrt(N_class); validation unbalanced.
py train\train_temporal_7class_mild_balance.py `
  --train data\train_temporal_7class.jsonl `
  --val data\val_temporal_7class.jsonl `
  --output outputs\qwen3_4b_football_temporal_7class_mildbalance

# Labels are inferred from the validation file (or pass --labels A,B,...).
py src\evaluate_action_model.py `
  --val data\val_temporal_7class.jsonl `
  --adapter outputs\qwen3_4b_football_temporal_7class_mildbalance `
  --predictions outputs\qwen3_4b_football_temporal_7class_mildbalance\val_predictions.jsonl
```

The run writes `run_config.json` (config, seed, class/sampled counts, dataset sizes,
Git SHA and dirty flag, dataset SHA256 hashes, library/CUDA/GPU versions). It saves a
checkpoint every 200 optimizer steps (latest 2 kept) and refuses a non-empty output
directory. To resume an interrupted run:

```powershell
py train\train_temporal_7class_mild_balance.py `
  --train data\train_temporal_7class.jsonl `
  --val data\val_temporal_7class.jsonl `
  --output outputs\qwen3_4b_football_temporal_7class_mildbalance `
  --resume-from-checkpoint outputs\qwen3_4b_football_temporal_7class_mildbalance\checkpoint-NNN
```

Run tests with `py -m unittest discover -s tests -v`.

## PASS-receiver modeling

This separate task predicts `TEAMMATE_1` through `TEAMMATE_10` from anonymous
player states at `t-10`, `t-5`, and `t` frames. Entity identities are held fixed
over the three states. It does not change the action-classification datasets.

Labels are **high-confidence pseudo-labels, not verified intended-pass ground
truth**: the next same-team direct action must be by a different player within
50 frames (2 seconds), without an intervening opponent action, restart, outcome,
block, tackle, or other ambiguity. The receiver must map to a teammate present
at pass time. Future events construct targets only; prompts contain no future
event information or permanent player IDs. HIGH_PASS, CROSS, and reconstructed
ball coordinates are not used by the receiver dataset or model.

The fixed match split is:

- Train: 117092, 117093, 118576, 118578, 128057, 132831, 132877 (4,799 examples).
- Validation: 118577, 128058 (1,224 examples).

With the original per-half decision, BAS, and GSR files available locally, build
the receiver data once. This command refuses to overwrite its output files;
do not rerun it to change labels during an experiment. Create the report directory
first if it does not exist:

```powershell
New-Item -ItemType Directory -Force analysis | Out-Null
py src\build_pass_receiver_dataset.py --data-dir data
```

Train from `Qwen/Qwen3-4B-Base`, not an existing adapter:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
py train\train_pass_receiver_qwen.py `
  --output outputs\qwen3_4b_pass_receiver `
  --seed 42 2>&1 |
  Tee-Object -FilePath outputs\qwen3_4b_pass_receiver_training.log
```

Defaults match the validated temporal LoRA configuration: one epoch, natural
unbalanced targets, learning rate `2e-4`, batch size 1, gradient accumulation 16,
maximum length 2048, LoRA rank 16 / alpha 32 / dropout 0.05. Loss is computed
only on response tokens plus EOS. Checkpoints are saved every 100 optimizer
steps, retaining the latest two. Non-empty output directories are rejected unless
resuming. Resume an interrupted run without restarting from scratch:

```powershell
py train\train_pass_receiver_qwen.py `
  --output outputs\qwen3_4b_pass_receiver `
  --seed 42 `
  --resume-from-checkpoint outputs\qwen3_4b_pass_receiver\checkpoint-NNN
```

Use a new output directory and `--seed 7` or `--seed 123` for fresh replicas.
Each run records its configuration, dataset SHA256 hashes, Git state, environment,
target counts, and training metrics in `run_config.json`.

```powershell
py src\evaluate_pass_receiver_model.py `
  --adapter outputs\qwen3_4b_pass_receiver `
  --val data\val_pass_receiver.jsonl
```

The primary evaluation scores all ten candidates, using **mean conditional
log-probability over candidate tokens plus EOS**, not prompt-token loss or
generated text alone. This normalizes for the different token length of
`TEAMMATE_10`. The evaluator writes `val_predictions.jsonl` and `val_metrics.json`
inside the adapter directory. It also checks generated text on the first 20
examples; generation agreement is diagnostic, not the ranking metric.

### Verified results

The clean 7-class mildly balanced action run reached **52.13% accuracy** and
**0.4715 macro-F1**, with no invalid generations on 3,800 validation examples.
This is a different task from receiver prediction and is not directly comparable
to receiver metrics.

Receiver validation on the same 1,224 examples for every seed:

| Model / baseline | Top-1 | Top-3 | Top-5 | MRR | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Geometric distance ranking | 35.78% | 79.82% | 95.10% | 0.5914 | - |
| Qwen receiver, seed 42 | 55.07% | 88.48% | 97.63% | 0.7247 | 0.2780 |
| Qwen receiver, seed 7 | 55.07% | 88.40% | 97.55% | 0.7234 | 0.2958 |
| Qwen receiver, seed 123 | 51.88% | 87.99% | 97.06% | 0.7033 | 0.2521 |

All three seeds beat geometric top-1 and MRR. This supports continuing
PASS-receiver modeling, not claiming ground-truth intended-pass accuracy.
Ranks 8-10 have only nine validation examples combined and no top-1 predictions.
See the [experiment report](results/pass_receiver_experiment.md) for limitations,
per-target results, provenance, and three-seed variability, and the
[machine-readable record](results/pass_receiver_experiment.json) for exact metrics.

Only source, tests, documentation, and compact result summaries belong in Git.
Datasets, per-example predictions, logs, adapters, and checkpoints remain local.

## Evaluation standards

We treat language-model loss as a secondary signal, not the main product metric.

The evaluation should report:

- action accuracy
- macro-F1
- per-class F1
- invalid-generation rate

These metrics reflect the actual system objective: tactical classification quality in context.

## Recommended workflow

For a production-quality run, the sequence should be:

1. extract one or more matches
2. inspect frame-state alignment and actor presence
3. aggregate at least 6 matches before model training
4. build the SFT split
5. establish the majority baseline
6. train the LoRA adapter
7. evaluate against the baseline using classification metrics

## Notes

This project is intended as a starter pipeline for tactical action modeling in football. It is built to be extended as needed for richer player context, stronger event representation, more complex labels, or additional downstream evaluation tasks.
