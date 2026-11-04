# Handoff

**PARTIAL_WITH_NEW_FINDINGS**. Actual elapsed time at report generation: 1.03 hours. Retained 456 fold fits, 72 on CUDA; 55,050 GPU steps. Seven distinct adaptive branches completed. Main physical labels remain unavailable; no fake completion is claimed.

Best practical result: 1-2 s any-BAS full-context macro AP 0.2346 versus 0.1963 kinematic and 0.0739 distance. Opponent-only 0.2262 is stronger than same-team-only 0.2112. All six held matches improve with full context.

Strong negative result: learned conditional actor allocation does not improve unconditional AP convincingly over equal-scene-probability HGB allocation, despite better switch top1. Retain the simple opponent-aware baseline.

Independent evidence: two pinned official Metrica samples evaluated without external fitting; Game2 parser/input hashes and official blob SHAs locked before scoring. Historical60human pass judgments recovered and precisely scoped; zero new labels, review paused.

Validation:26pytest tests;7browser core checks and real4sclip playback;17diagnostics/eight anonymous diagrams;1117protected hashes unchanged. Scheduling version1 and external mapping1 are retained as invalid attempts, with verified repairs.

Return archive: outputs/return_next_touch_evidence_20261103T215415Z.zip. Final actual SHA-256/CRC/member hashes are in its adjacent .sha256 and .verification.json; verify_package.py independently recomputes them. Exact reproduction commands are in README.md.

Best next question: does the opponent-aware1-2s benefit survive independently synchronized, event-complete evaluation with verified outcomes for all sampled intervals?
