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
- `src/majority_baseline.py` — establishes a simple baseline for comparison
- `train/train_lora_qwen.py` — fine-tunes a Qwen3-4B model with LoRA
- `src/evaluate_action_model.py` — evaluates model predictions with classification metrics
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
- player and possession context

### 2) Check dataset quality before training

```powershell
py src\inspect_dataset.py --decisions data\117093_1st_decisions.jsonl
py src\serialize_state.py data\117093_1st_decisions.jsonl
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

The split should be match-level and should preserve realistic tactical variation across training and validation windows.

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
