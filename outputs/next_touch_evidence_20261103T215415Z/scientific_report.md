# Evidence-grounded football forecasting

**OVERALL_STATUS: PARTIAL_WITH_NEW_FINDINGS.** Executed 1.03 hours of wall-clock work, not the aspirational 12-20 hours. No compute was run merely to fill a duration. Retained 456 fold-fit records (72 CUDA fits, 55,050 optimizer steps), two independent public-source sample matches, seven post-results branches, and 17 diagnostics plus eight anonymous case diagrams. Original protected inputs/checkpoints/results: **1117 unchanged hashes**.

## What is actually grounded

Fourteen native halves were audited and fourteen private four-second source-clock clips extracted. FFprobe verified decode clocks/dimensions; source BAS floor-time and ball as-of conventions were retained without optimizing an offset. Median adjacent player-position repeat fraction across halves is0.883. Positions lie on the source's coarse1.05m by0.68m grid. Small images, panorama distortion and night glare limit visual adjudication; no independently verified physical touch or generic verified negative was fabricated.

The earlier statement of zero human labels was too broad: **60 historical single-reviewer pass judgments and60 corresponding local clips exist**.39 say the target pass is visible. Their questions are first-touch/control **team**, not exact touch actor/time. A strictly scoped36-case visible-pass/known-team association audit ran. On the matched35 covered cases, retrospective future geometry agrees with human touch team27/35, versus20/35 for always-passer-team. That method uses future tracking, so it is **not a forecast**. Enriched historical selection, one reviewer and uncertain physical sync preclude a new contact benchmark. New review remains paused with **zero new human labels**. The ready134-case viewer plays local clips, hides predictions until an immutable first judgment, blocks unverified negatives, exports JSON/CSV and preserves independent reviewer/arbitration records.

Three fixed contact-candidate algorithms were executed on12fitting halves and independently checked against Metrica source PASS/SHOT timestamps. Native changepoint candidates occupy39.1%of valid ball frames, versus13.3%in Metrica. These are repeated **candidate states**, not distinct certified touches. In Metrica, proximity agrees with the same source PASS actor within0.48s for726/768eligible events; median source-start actor-ball distance is0.088m. Strong annotation association is not independent video certification, and unannotated frames are not verified negatives.

## Prospective targets and longer horizons

Fixed current rosters, identities, feature clocks and masks are preserved. Future availability, ambiguous actors, missing targets and0.2m geometry ties cause target-only censoring, not future inputs.

| Target | Eligible actor rows | Positive rows | Nearest macro AP | Interpretable macro AP |
|---|---|---|---|---|
| geometry_1s | 674333 | 30660 | 0.3994 | 0.7273 |
| geometry_2s | 673353 | 30615 | 0.2211 | 0.4890 |
| next_BAS_0_1s | 731519 | 6618 | 0.1927 | 0.3343 |
| next_BAS_1_2s | 730635 | 6609 | 0.0781 | 0.1794 |
| geometry_3s | 672523 | 30577 | 0.1513 | 0.3515 |
| geometry_5s | 670575 | 30488 | 0.1008 | 0.2229 |
| next_BAS_3_5s | 729404 | 12474 | 0.0390 | 0.1068 |

Geometric targets mean future nearest player, not possession/touch. The3-5s BAS target is a **two-second interval**, unlike one-second earlier bins; prevalence changes prohibit interpreting its AP as a pure one-second lead curve. The previous mission's future-nearest target was+2s, not+1s. The small change from its roughly0.474AP to this mission's0.489 at+2s also changes tie censoring; it is not an architecture-only improvement.

For any annotated BAS action, inherited interpretable AP is0.4980/0.2317/0.1266 in0-1/1-2/2-3s. A fully joint0.48s-stale35channel arm drops AP from0.4681to0.3254,0.1963to0.1301,and0.0966to0.0754. Recent ball/player state carries useful information beyond proximity. These are **annotation-relative** forecasts, not human decisions inferred from gaze.

## Tactical context is useful, but not optimal tactics

| Task | Distance | Kinematic | Same team | Opponent | Full context |
|---|---|---|---|---|---|
| 1-2 s any BAS action | 0.0739 | 0.1963 | 0.2112 | 0.2262 | 0.2346 |
| Future nearest at 2 s (all) | 0.2193 | 0.4632 | 0.4719 | 0.4910 | 0.4962 |
| 1-2 s next BAS actor | 0.0765 | 0.1660 | 0.1697 | 0.1765 | 0.1788 |

For1-2s action forecasting, full context adds0.0383AP over identical-capacity kinematics (19.5%relative); every one of six held matches improves. Descriptive paired match-bootstrap interval:[0.02661192995836504, 0.04683707790954145]. With only six known matches, that is not population significance. Opponent pressure accounts for more of the controlled lift than same-team-only context. Correlated grouped importance is a sensitivity analysis, not a causal explanation.

Separate conditional action-type classifiers also ran: full-context macro F1 is0.291at1-2s, versus0.265kinematic and0.088class-prior. Dominant categories and absent rare classes limit the stronger-looking accuracy. This is type **given annotated occurrence**, not an unconditional seven-way alarm.

## Seven adaptive branches, actually executed

**A: quantization-aware velocities.** One-second backward smoothing modestly changes geometry2s AP0.4632to0.4680; next-BAS1-2s changes0.1660to0.1700. Helpful but not a major breakthrough.

**B: GPU conditional scene-softmax.** Three matched seeds, six folds and two targets were trained, with epoch selection only on 132877. On 6,609 supported future-BAS scenes, top1 changes 0.5957 to 0.6031. On 2,928 current-nearest-switch scenes it changes 0.2158 to 0.2466; conditional NLL nevertheless worsens 1.4042 to 1.4293. Geometry 2 s switch top1 improves 0.1549 to 0.2037 over 17,045 scenes. Conditional occurrence is supplied for this diagnostic only.

**C: development calibration/evidence abstention.** Raw next-BAS macro NLL0.035395becomes0.035313; Brier slightly worsens. ECE changes0.000551to0.000459. This is a small annotation-relative reliability change, not operational calibration.

**D: external coordinate-resolution intervention.** Identical native Metrica targets/candidates were kept while player/ball inputs were independently rounded to the source grid. Rounding both lowers Game1 geometry1s AP0.6825to0.6662 and2s0.3993to0.3841. Resolution mismatch matters but does not explain away all useful temporal information.

**E: unconditional factorization.** B's conditional improvement was tested without supplying future occurrence at inference. The same development-calibrated scene probability was allocated by normalized HGB or learned softmax over **all current scenes**. Macro AP changes0.167755to0.167826; delta0.000072, descriptive interval[-0.0019097555853850867, 0.002015188950782329]. NLL is worse. **The conditional ranking gain did not produce a convincing unconditional forecast gain.**

**F: untouched second external sample replication.** Game2's official file sizes and Git blob SHAs, parser, source features and target hashes were locked before its download/scoring. Exactly the same models/mapping were applied without Metrica fitting. It replicates geometric transfer but does not constitute a new SoccerTrack sealed test.

**G: matched training-objective control.** B's architecture/ensemble confound was addressed by another 36 CUDA fits with identical MLP, input, positive training scenes, seeds, initializations, batches, epochs and conditional-development-NLL selection. Only training loss/link differ: independent BCE versus scene softmax. Next-BAS switch top1 is 0.1513 versus 0.2466; overall conditional top1 is 0.5729 versus 0.6031. Conditional representation/objective matters, but E prevents promoting this to an unconditional deployed improvement.

## Independent domain evidence

| Game | Target | Eligible scenes | Nearest AP | Kinematic AP | Projection switch top1 | HGB switch top1 |
|---|---|---|---|---|---|---|
| 1 | geometry_1s | 3158 | 0.4673 | 0.6825 | 0.4194 | 0.3202 |
| 1 | geometry_2s | 3060 | 0.2361 | 0.3993 | 0.1912 | 0.1872 |
| 2 | geometry_1s | 3038 | 0.4313 | 0.6518 | 0.4049 | 0.3399 |
| 2 | geometry_2s | 2945 | 0.1974 | 0.3528 | 0.1940 | 0.1830 |

Game1's initial mapper used an inconsistent ball derivative interval and missed the source pitch-bounds mask. That invalid version is retained; mapping revision2 reuses the **exact same causal producer** as SoccerTrack. Game2 was frozen only after that repair and evaluated once. Neither is used for model selection. Current-ball-available eligibility and domain taxonomy differences are explicit. Source PASS is not silently combined with SoccerTrack's broader contact-like BAS target. No public raw data/video was repackaged.

## Scene workload and adversarial gates

At 10 scenes/hour and top3 actors, early full context reviews 90 scenes/270 actor candidates, recovering 68/11680 distinct supported annotations (0.58%). This favors selective offline review, not comprehensive alarms. Annotation-relative misses are not physical false positives.

Final audit caught a real scheduling flaw: version1 removed future-label-unsupported queries. It is preserved as invalid. Version2 separately refits frozen decoders to score **every** held-match current query and never consults future validity for scheduling; unknown selected outcomes are unverifiable, not false. Other model comparisons are unchanged.

Within-scene label permutation controls reduce geometry2s AP to0.0463/0.0512/0.0574, versus0.4632 true labels. Tests cover exact clock/prefix causality, future perturbations, identity permutations, duplicate joins, segment transitions, missing/bounded balls, negative raw scores, window boundaries/ties/censoring, distinct-event deduplication, caches, split integrity and ZIP tampering. Browser tests verify blinding/export and actual clip playback without creating labels.

## Decision

Keep the compact interpretable **opponent/full-context HGB** as the prospective baseline. Recent kinematics and source-team pressure contain useful predictive signal; the limitation is not simply absence of useful position information. Better conditional objectives improve who-to-rank, but do not solve when-an-action-occurs. Exact physical touches, representative negatives, human perception/intent, optimal tactics, deployment alarms and sealed SoccerTrack generalization remain **unvalidated**.

The best next question is: **On independently synchronized, event-complete footage, does opponent-aware causal context retain its1-2s benefit when every candidate interval has a verified outcome?** Human review remains paused; no new review is requested in this handoff.
