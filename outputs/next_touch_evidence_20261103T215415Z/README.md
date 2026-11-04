# Extended evidence-grounded football mission

Status: **PARTIAL_WITH_NEW_FINDINGS**. Read scientific_report.md, failure_analysis.md and data_quality.md before interpreting scores. Original source data/checkpoints/results are preserved; this package contains no footage, private per-event IDs, human label rows, weights or large feature shards.

## Reproduce in the existing repository

Keep this directory under outputs/next_touch_evidence_20261103T215415Z. The existing repository, pinned historical scripts and authorized local datasets are required; this is not a standalone data distribution.

```powershell
& '.\.venv\Scripts\python.exe' '.\outputs\next_touch_evidence_20261103T215415Z\resume.py' --all
& '.\.venv\Scripts\python.exe' -m pytest '.\outputs\next_touch_evidence_20261103T215415Z\test_evidence.py' -q
& '.\.venv\Scripts\python.exe' '.\outputs\next_touch_evidence_20261103T215415Z\verify_package.py' '.\outputs\return_next_touch_evidence_20261103T215415Z.zip'
```

Default resume verifies protected hashes and prints persisted status without fitting. --all first verifies the original stage0 hash pins without replacing that baseline, then executes stages1-9, longer horizons, conditional action types, source-contact/historical-team audits, importance, negative controls and adaptive D-G, then tests/figures/report/package. Existing fingerprinted fold outputs resume without retraining. The two official small Metrica downloads are pinned, bounded-retry and local only. PowerShell execution policy is not changed; entrypoints are native Python.

## Contents

- scientific_report.md: measured comparisons, sample sizes, seven adaptive branches and scientific interpretation.
- execution_manifest.json / acceptance.json / source_integrity_after.json: actual execution and protected-input gates.
- protocol and adaptive locks: prospective definitions, evidence hashes and Game2 freeze.
- metric JSON: six-match OOF, source data quality, external per-match pilot, scene quotas and negative findings.
- feature_dictionary.md / dataset_comparison.md: semantics, domain mapping and access boundaries.
- figures/: anonymized diagnostics/cases only; no video frames.
- review.html / review_core.js: working offline two-reviewer preparation tool. Load private review_local/queue.json plus the local evidence folder; nothing is pre-labeled. Human review remains paused.
- dependency_snapshots/: related immutable source-code snapshots for reference; execution resolves the existing repository's original files and verifies their hashes.

Archive verification is independent: ZIP CRC, SHA-256 for every member, final-byte archive SHA-256 and sidecar consistency. The final archive hash cannot be embedded in the archive itself; see the adjacent .sha256 and .verification.json.
