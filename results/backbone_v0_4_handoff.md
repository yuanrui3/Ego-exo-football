# Backbone v0.4 integration handoff

## Status

The configured nine-match corpus, one action-only S4 fit, checkpoint reload, and shared replay are complete. The existing pixel-only video pilot was extended with conservative image-space tracklet proposals. The pitch-facing video adapter remains blocked. This is a compact event-conditioned classifier prototype, not a continuous action detector or a best-action policy.

## Corpus and labels

The full configured intervals were read from the existing local cache for both halves of all nine matches. The split is:

| Split | Matches | Eligible examples |
|---|---|---:|
| Fitting | 117092, 117093, 118576, 118578, 128057, 132831 | 6,850 |
| Development | 132877 | 1,714 |
| Retrospective evaluation | 118577, 128058 | 2,031 |

The fitting set is strictly separated from development and retrospective evaluation. The `128057` source was ingested in full but produced zero eligible examples under the input/label-coverage gates; it remains part of the declared fitting split, not a source of fabricated labels.

Across all splits, 19,311 raw events produced 10,595 eligible event-conditioned examples and 4,457 unlabeled arbitrary windows. Recorded exclusions include 7,433 unsupported event/frame alignments, 1,196 unsupported/context labels, 74 missing actor identities, 12 insufficient histories, and one actor missing at query time. The excluded events remain exclusions; unsupported alignments were not silently shifted.

Labels are joined by the stable event ID plus annotation index, and BAS frame/period and label mapping are checked. Original BAS labels, canonical class, and mapping provenance are stored only in each shard's `optional_targets.jsonl`. Student inputs have no action label. Arbitrary windows have no target and are never treated as negative/NO_ACTION examples.

The retained seven-class taxonomy and order are `PASS`, `HIGH_PASS`, `DRIVE`, `CROSS`, `SHOT`, `HEADER`, `DEFENSIVE_INTERVENTION`. Fitting counts are 3,133; 413; 2,850; 153; 108; 11; and 182 respectively. Sparse `HEADER` and low-frequency classes warrant caution.

## Selected checkpoint and inference contract

- Trained checkpoint: `outputs/backbone_v0_4_20261020/training/model/final.pt`
- SHA-256: `5a9d46eade21ebd5019f32d4ada4e20c6909754fc0e7e06af0add0f146fcb25a`
- Model: S4, width 96, 2 layers, 4 heads, dropout 0.1, seed 42.
- Initialization: `outputs/partial_observation_20261017/students/development/S4/width96/best.pt`, SHA-256 `f308b679ad0104ebde27b648c6b25dcf887ab3c2697e723fa2e93a3f0b43d5c9`; its fitting provenance is the six fitting matches. It was selected using development match `132877`, not either retrospective-evaluation match.
- One action-only cross-entropy fit ran on the RTX A5000 for four epochs, early-stopped at patience 3, and selected epoch 1 with development macro-F1 0.4796. `training/last.pt` retains model, optimizer and RNG state at the last completed epoch; `training/best.pt` retains the selected development checkpoint.
- Inputs: `[11,22,10]`, two seconds of history at 0.2-second intervals, inclusive of the query time. The event query is `floor(position_ms / 40)`; post-event frames are excluded. Samples failing the documented frame alignment are excluded rather than shifted.
- Observation: 120-degree synthetic FOV, 45 m range, causal motion-heading visibility, no artificial dropout/noise in this export, continuously observed actor, and legacy current-roster/team/role compatibility gates. Features are relative coordinates normalized by 52.5 m/34 m, velocity by 12 m/s, visibility, ever-seen memory, age capped at 5 s and divided by 5, team relation, goalkeeper flag, and actor flag. Model mask is causal ever-seen memory. S4 has no global slot embeddings; student slot order is anonymous.
- `ActionModelSession` loads once and accepts the existing v0.2 `WorldInput`; it converts the v0.2 `/8s` memory-age feature to S4's `/5s` scale. Direct full-precision checkpoint inference and adapter reload matched across all 2,031 retrospective examples: maximum absolute probability difference `4.77e-7`.
- The shared replay calls `ReplayEngine(..., action_model=...)`. It contains fitting examples for PASS, HIGH_PASS, and DRIVE, plus three unlabeled exploratory windows. Annotation labels are evaluation-only. Model softmax scores are uncalibrated class scores, not confidence or tactical value. Passing geometry remains a separate optional module; action target/location/direction/timing are not inferred. Ball, possession, restart, and outcome remain unknown absent their own evidence. Other action branches remain unsupported.

Retrospective metrics are diagnostic only: accuracy 0.5977 and macro-F1 0.5045. Both evaluation matches appeared in earlier research work, so this is not a newly unseen test set and should not be compared with unrelated historical scores. `HIGH_PASS`, `SHOT`, `HEADER`, and `DEFENSIVE_INTERVENTION` have low supports and weak/unstable recall.

## Video pilot

The existing local 12-frame/25-detection pilot was reused; no footage was downloaded and the detector was not rerun. The extension emits 17 unverified image-space candidate tracklets and three overlays. Associations use only adjacent sampled frames; empty frames and heuristic potential cuts break links. No potential cut was detected by the heuristic, but it is not a validated shot detector and the tracklets are not athlete identities.

`video_pilot_extension/pixel_calibration_sidecar.json` supports only dimensionless normalized image coordinates. Pitch projection and synchronization remain blocked; no pixel coordinates are sent to the pitch backbone. Team and role remain unknown.

## Acceptance evidence

`outputs/backbone_v0_4_20261020/acceptance_record.json` records status, counts, hashes and focused tests. The five-test v0.4 acceptance file covers export/split/replay/video behavior. In the working environment, the broader 47-test regression command also passed v0.3 adapter and v0.2 backbone/ingestion tests; those existing tests rely on prior local research artifacts. Corpus `--resume` completed all 18 hash-verified shards without appending duplicate rows.

## Reproduction (Windows PowerShell, repository root)

Choose a fresh `$run` directory for a clean reproduction. Do not overwrite existing outputs.

```powershell
$py = 'C:\Users\flexadmin\AppData\Local\Python\pythoncore-3.14-64\python.exe'
$run = 'outputs\backbone_v0_4_repro'

& $py -m src.backbone.corpus --config configs\backbone_v0_4_corpus.json --output "$run"
# Safe after interruption or to verify indexed shards:
& $py -m src.backbone.corpus --config configs\backbone_v0_4_corpus.json --output "$run" --resume

& $py -m src.backbone.train_action --index "$run\training_index.json" --config configs\backbone_v0_4_training.json --output "$run\training"
# Resume only from the last safe epoch checkpoint:
& $py -m src.backbone.train_action --index "$run\training_index.json" --config configs\backbone_v0_4_training.json --output "$run\training" --resume
& $py -m src.backbone.train_action --index "$run\training_index.json" --output "$run\training" --evaluate

& $py -m src.backbone.v0_4 --index "$run\training_index.json" --checkpoint "$run\training\model\final.pt" --output "$run\replay"
& $py -m src.backbone.video_tracklets --pilot outputs\backbone_v0_2_20261020_1558\video_pilot --output "$run\video_pilot_extension"

$env:BACKBONE_V0_4_OUTPUT = "$run"
& $py -m pytest tests\test_backbone_v0_4.py -q
```
