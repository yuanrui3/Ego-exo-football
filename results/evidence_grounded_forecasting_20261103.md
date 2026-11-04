# Evidence-grounded football forecasting

**Scientific status: PARTIAL_WITH_NEW_FINDINGS.** This publication records
executed experiments, not validated physical touch detection or a deployment.
Frozen release defaults and protected checkpoints are unchanged.

## What is worth retaining

- Six-match exploratory, match-excluded evaluation: adding opponent/full
  spatial context raises 1-2 second annotated-action macro AP from 0.1963
  (kinematics) to 0.2346 (full context), an absolute 0.0383 and relative 19.5%
  gain. All six held matches improve. There are 729,942 supported actor-query
  rows and 11,680 positive annotations; these are not independent samples.
- Conditional actor ranking improves with a scene-softmax objective, including
  a matched architecture/data/seed BCE control. The corresponding unconditional
  allocation experiment produces essentially no AP gain (0.167755 to 0.167826)
  and worse NLL. Better ranking does not establish better occurrence forecasting.
- Two pinned public Metrica sample matches replicate temporal-geometric
  predictive signal without external fitting. Simple trajectory projection
  remains competitive or better on external switch-only top-1 accuracy.
- Seven result-driven branches actually ran: quantization-aware velocities,
  conditional ranking, calibration/abstention, coordinate-resolution
  interventions, unconditional factorization, frozen second-match replication,
  and matched training-objective control.
- Final scheduling scores every current query before checking future-label
  support. Invalid earlier scheduling and external mapping results are
  preserved and explicitly labeled, not silently discarded.

Actual execution retained 384 CPU fold fits, 72 CUDA fits and 55,050 GPU optimizer
steps in approximately 62 minutes. Validation passed 26 automated tests, seven
browser checks, and 1,117 protected-file hash comparisons. Figures include
17 diagnostics and eight anonymous scene diagrams.

## Read the evidence

- [Scientific report](../outputs/next_touch_evidence_20261103T215415Z/scientific_report.md)
- [Negative results and repairs](../outputs/next_touch_evidence_20261103T215415Z/failure_analysis.md)
- [Data quality and limitations](../outputs/next_touch_evidence_20261103T215415Z/data_quality.md)
- [Updated roadmap](../outputs/next_touch_evidence_20261103T215415Z/roadmap.md)
- [Execution manifest](../outputs/next_touch_evidence_20261103T215415Z/execution_manifest.json)
- [Acceptance and remaining blockers](../outputs/next_touch_evidence_20261103T215415Z/acceptance.json)
- [Code, reproduction commands and review tool](../outputs/next_touch_evidence_20261103T215415Z/README.md)
- [Verified compact ZIP](../outputs/return_next_touch_evidence_20261103T215415Z.zip)

The ZIP has 128 members and is 3,098,514 bytes. Its final-byte SHA-256 is:

```text
46d99a011fcb0152aca90dc37552972d3b418401d441057bc248f5ca57c976fa
```

Independent verification checks CRC, every member's SHA-256, and the archive
hash against its [sidecar](../outputs/return_next_touch_evidence_20261103T215415Z.zip.sha256).

## Reproduction and publication boundaries

From the repository root:

```powershell
& '.\.venv\Scripts\python.exe' '.\outputs\next_touch_evidence_20261103T215415Z\resume.py'
& '.\.venv\Scripts\python.exe' -m pytest '.\outputs\next_touch_evidence_20261103T215415Z\test_evidence.py' -q
& '.\.venv\Scripts\python.exe' '.\outputs\next_touch_evidence_20261103T215415Z\verify_package.py' '.\outputs\return_next_touch_evidence_20261103T215415Z.zip'
```

Full experiment reproduction adds `--all` to the first command and requires
the existing authorized local datasets, pinned historical assets and caches.
This publication is **not a standalone dataset or clean-clone training demo**.
The original byte-pinned research files retain their line endings deliberately.
The ZIP includes dependency snapshots; the browsable publication places those
same dependency scripts at their original repository paths.

Only the selected research code, aggregate metrics, provenance metadata,
anonymous plots, tests, reports and verified ZIP are published. Raw tracking,
videos, human label rows, private per-event player mappings, review queues,
feature shards, predictions and model weights remain local. The general
`outputs/` ignore rule remains in place; only these selected files are tracked.

The review tool is prepared and tested, but new human review remains paused.
Sixty historical single-reviewer judgments concern pass visibility and
touch/control team, not exact contact actor/time. No new human labels were
created. Physical synchronization, exact touches, representative negatives,
human perception/intent, optimal tactics, operational alarms and sealed
SoccerTrack generalization remain unvalidated.

**Next research gate:** independently synchronized, event-complete evaluation
of the opponent-aware 1-2 second forecasting benefit, rather than more
conditional-ranking optimization alone.
