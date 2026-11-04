import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from common import ROOT, RUN, PRIOR, FIT, read, write, digest, status
from verify_package import verify


def md(name,text):
    (RUN/name).write_text(text.strip()+"\n",encoding="utf-8")


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join(["---"]*len(headers))+"|"]+["| "+" | ".join(map(str,row))+" |" for row in rows])


def report():
    required=["adaptive_quantization.json","adaptive_listwise.json","adaptive_calibration.json","adaptive_domain_resolution.json","adaptive_unconditional_factorization.json","external_results_game2.json","objective_control_comparison.json"]
    for name in required:
        if not (RUN/name).exists():raise RuntimeError("Missing actual experiment "+name)
    stages=read(RUN/"stage_status.json")
    if any(stages[str(k)]["status"]!="COMPLETE" for k in (0,3,4,5,6,7,8,9)):
        raise RuntimeError("Required executed stage incomplete")
    source_hashes=read(RUN/"source_hashes_before.json")
    mismatches=[name for name,expected in source_hashes.items() if not (ROOT/name).exists() or digest(ROOT/name)!=expected]
    write(RUN/"source_integrity_after.json",{"status":"PASS" if not mismatches else "FAIL","files":len(source_hashes),"mismatches":mismatches})
    if mismatches:raise RuntimeError("Protected sources changed")
    tests=ET.parse(RUN/"tests.xml").getroot()
    suites=tests.findall(".//testsuite")
    test_result={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ("tests","failures","errors","skipped")}
    if test_result["tests"]<26:raise RuntimeError("Missing complete targeted test results")
    if test_result["failures"] or test_result["errors"]:raise RuntimeError("Tests failed")
    write(RUN/"test_results.json",test_result)
    figures=read(RUN/"figure_inventory.json")
    if figures["diagnostics"]<8 or not 6<=figures["anonymous_case_diagrams"]<=12:raise RuntimeError("Figure gate failed")
    protocol=read(RUN/"protocol.lock.json")
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(protocol["started_utc"].replace("Z","+00:00"))).total_seconds()
    records={p.stem:read(p) for p in (RUN/"fits_local").glob("*.json")}
    gpu={k:v for k,v in records.items() if "gpu" in v}
    execution={"status":"PARTIAL_WITH_NEW_FINDINGS","started_utc":protocol["started_utc"],"finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":elapsed,"retained_fold_fit_records":len(records),"retained_CPU_fold_fits":len(records)-len(gpu),"retained_CUDA_fits":len(gpu),"CUDA_optimizer_steps":sum(v["steps"] for v in gpu.values()),"GPU_models_local":len(list((RUN/"models_local").glob("*.pt"))),"protected_files_unchanged":len(source_hashes),"fit":FIT,"development":["132877"],"external_matches":2,"external_final_source_only_models":12,"additional_resolution_source_only_models":2,"adaptive_branches_executed":list(read(RUN/"adaptive_decisions.json")),"tests":test_result,"figures":figures,"fit_records":records,"scope_note":"Retained unique fold records, not an inflated count of cached resumes. External mapping-repair attempts and failed integration executions are separately documented; no protected model retraining."}
    write(RUN/"execution_manifest.json",execution)
    context=read(RUN/"tactical_context_results.json")
    next_actor=read(RUN/"next_interactor_results.json")
    long=read(RUN/"longer_horizon_results.json")
    action=read(RUN/"anticipation_challenges.json")
    scene=read(RUN/"scene_prioritization.json")
    if any(v.get("scheduler_revision")!=2 for v in scene.values()):raise RuntimeError("Future-mask-free scheduler required")
    a=read(RUN/"adaptive_quantization.json");b=read(RUN/"adaptive_listwise.json");c=read(RUN/"adaptive_calibration.json")
    e=read(RUN/"adaptive_unconditional_factorization.json");g=read(RUN/"objective_control_comparison.json")
    external=[read(RUN/"external_results.json"),read(RUN/"external_results_game2.json")]
    quality=read(RUN/"clock_and_sync_audit.json")["halves"]
    ext_contact=read(RUN/"external_contact_association.json")
    native_contact=read(RUN/"contact_candidate_audit.json")["halves"]
    native_rate=sum(v["candidate_frames"]["changepoint"] for v in native_contact.values())/sum(v["valid_ball_frames"] for v in native_contact.values())
    semantic=read(RUN/"conditional_action_type.json")
    candidate_rows=[]
    for name,r in {**next_actor,**long}.items():
        candidate_rows.append([name,r["pooled"]["interpretable"]["rows"],r["pooled"]["interpretable"]["positives"],f'{r["macro_match"]["nearest"]["AP"]:.4f}',f'{r["macro_match"]["interpretable"]["AP"]:.4f}'])
    context_names={"early_action":"1-2 s any BAS action","geometric_switch_risk":"Future nearest at 2 s (all)","next_BAS_actor":"1-2 s next BAS actor"}
    context_rows=[[context_names[name]]+[f'{r["macro_match"][m]["AP"]:.4f}' for m in ("distance","kinematic","same_team","opponent","full_context")] for name,r in context.items()]
    external_rows=[]
    for game,r in enumerate(external,1):
        for target in ("geometry_1s","geometry_2s"):
            t=r["results"][target]
            external_rows.append([game,target,t["positive_rows"],f'{t["models"]["nearest"]["all"]["AP"]:.4f}',f'{t["models"]["kinematic"]["all"]["AP"]:.4f}',f'{t["models"]["constant_velocity"]["switch_only"]["scene"]["top1"]:.4f}',f'{t["models"]["kinematic"]["switch_only"]["scene"]["top1"]:.4f}'])
    context_delta=context["early_action"]["context_minus_kinematic"]
    early=context["early_action"]["macro_match"]
    relative=(early["full_context"]["AP"]/early["kinematic"]["AP"]-1)*100
    nn=b["next_BAS_1_2s"];geob=b["geometry_2s"]
    quota=scene["lead_1_2s"]["workload"]["full_context"]["10_top3"]
    main=f"""
# Evidence-grounded football forecasting

**OVERALL_STATUS: PARTIAL_WITH_NEW_FINDINGS.** Executed {elapsed/3600:.2f} hours of wall-clock work, not the aspirational 12-20 hours. No compute was run merely to fill a duration. Retained {len(records)} fold-fit records ({len(gpu)} CUDA fits, {execution['CUDA_optimizer_steps']:,} optimizer steps), two independent public-source sample matches, seven post-results branches, and {figures['diagnostics']} diagnostics plus eight anonymous case diagrams. Original protected inputs/checkpoints/results: **{len(source_hashes)} unchanged hashes**.

## What is actually grounded

Fourteen native halves were audited and fourteen private four-second source-clock clips extracted. FFprobe verified decode clocks/dimensions; source BAS floor-time and ball as-of conventions were retained without optimizing an offset. Median adjacent player-position repeat fraction across halves is{np.median([v['actor_repeated_position_fraction'] for v in quality.values()]):.3f}. Positions lie on the source's coarse1.05m by0.68m grid. Small images, panorama distortion and night glare limit visual adjudication; no independently verified physical touch or generic verified negative was fabricated.

The earlier statement of zero human labels was too broad: **60 historical single-reviewer pass judgments and60 corresponding local clips exist**.39 say the target pass is visible. Their questions are first-touch/control **team**, not exact touch actor/time. A strictly scoped36-case visible-pass/known-team association audit ran. On the matched35 covered cases, retrospective future geometry agrees with human touch team27/35, versus20/35 for always-passer-team. That method uses future tracking, so it is **not a forecast**. Enriched historical selection, one reviewer and uncertain physical sync preclude a new contact benchmark. New review remains paused with **zero new human labels**. The ready134-case viewer plays local clips, hides predictions until an immutable first judgment, blocks unverified negatives, exports JSON/CSV and preserves independent reviewer/arbitration records.

Three fixed contact-candidate algorithms were executed on12fitting halves and independently checked against Metrica source PASS/SHOT timestamps. Native changepoint candidates occupy{native_rate:.1%}of valid ball frames, versus{ext_contact['fraction_of_valid_frames']['changepoint']:.1%}in Metrica. These are repeated **candidate states**, not distinct certified touches. In Metrica, proximity agrees with the same source PASS actor within0.48s for726/768eligible events; median source-start actor-ball distance is0.088m. Strong annotation association is not independent video certification, and unannotated frames are not verified negatives.

## Prospective targets and longer horizons

Fixed current rosters, identities, feature clocks and masks are preserved. Future availability, ambiguous actors, missing targets and0.2m geometry ties cause target-only censoring, not future inputs.

{table(["Target","Eligible actor rows","Positive rows","Nearest macro AP","Interpretable macro AP"],candidate_rows)}

Geometric targets mean future nearest player, not possession/touch. The3-5s BAS target is a **two-second interval**, unlike one-second earlier bins; prevalence changes prohibit interpreting its AP as a pure one-second lead curve. The previous mission's future-nearest target was+2s, not+1s. The small change from its roughly0.474AP to this mission's0.489 at+2s also changes tie censoring; it is not an architecture-only improvement.

For any annotated BAS action, inherited interpretable AP is0.4980/0.2317/0.1266 in0-1/1-2/2-3s. A fully joint0.48s-stale35channel arm drops AP from0.4681to0.3254,0.1963to0.1301,and0.0966to0.0754. Recent ball/player state carries useful information beyond proximity. These are **annotation-relative** forecasts, not human decisions inferred from gaze.

## Tactical context is useful, but not optimal tactics

{table(["Task","Distance","Kinematic","Same team","Opponent","Full context"],context_rows)}

For1-2s action forecasting, full context adds{context_delta['mean_delta']:.4f}AP over identical-capacity kinematics ({relative:.1f}%relative); every one of six held matches improves. Descriptive paired match-bootstrap interval:{context_delta['ci95']}. With only six known matches, that is not population significance. Opponent pressure accounts for more of the controlled lift than same-team-only context. Correlated grouped importance is a sensitivity analysis, not a causal explanation.

Separate conditional action-type classifiers also ran: full-context macro F1 is{semantic['lead_1_2s']['macro_match']['full_context']['macro_F1']:.3f}at1-2s, versus{semantic['lead_1_2s']['macro_match']['kinematic']['macro_F1']:.3f}kinematic and{semantic['lead_1_2s']['macro_match']['train_class_prior']['macro_F1']:.3f}class-prior. Dominant categories and absent rare classes limit the stronger-looking accuracy. This is type **given annotated occurrence**, not an unconditional seven-way alarm.

## Seven adaptive branches, actually executed

**A: quantization-aware velocities.** One-second backward smoothing modestly changes geometry2s AP{a['geometry_2s']['macro_match']['native_velocity']['AP']:.4f}to{a['geometry_2s']['macro_match']['smoothed_velocity']['AP']:.4f}; next-BAS1-2s changes{a['next_BAS_1_2s']['macro_match']['native_velocity']['AP']:.4f}to{a['next_BAS_1_2s']['macro_match']['smoothed_velocity']['AP']:.4f}. Helpful but not a major breakthrough.

**B: GPU conditional scene-softmax.** Three matched seeds, six folds and two targets were trained, with epoch selection only on 132877. On 6,609 supported future-BAS scenes, top1 changes {nn['pooled']['hgb']['scene']['top1']:.4f} to {nn['pooled']['listwise_ensemble']['scene']['top1']:.4f}. On 2,928 current-nearest-switch scenes it changes {nn['switch_only']['pooled']['hgb']['scene']['top1']:.4f} to {nn['switch_only']['pooled']['listwise_ensemble']['scene']['top1']:.4f}; conditional NLL nevertheless worsens {nn['conditional_NLL']['hgb']:.4f} to {nn['conditional_NLL']['listwise_ensemble']:.4f}. Geometry 2 s switch top1 improves 0.1549 to 0.2037 over 17,045 scenes. Conditional occurrence is supplied for this diagnostic only.

**C: development calibration/evidence abstention.** Raw next-BAS macro NLL{c['macro_match']['raw']['NLL']:.6f}becomes{c['macro_match']['development_calibrated']['NLL']:.6f}; Brier slightly worsens. ECE changes{c['reliability']['raw']['ECE']:.6f}to{c['reliability']['development_calibrated']['ECE']:.6f}. This is a small annotation-relative reliability change, not operational calibration.

**D: external coordinate-resolution intervention.** Identical native Metrica targets/candidates were kept while player/ball inputs were independently rounded to the source grid. Rounding both lowers Game1 geometry1s AP0.6825to0.6662 and2s0.3993to0.3841. Resolution mismatch matters but does not explain away all useful temporal information.

**E: unconditional factorization.** B's conditional improvement was tested without supplying future occurrence at inference. The same development-calibrated scene probability was allocated by normalized HGB or learned softmax over **all current scenes**. Macro AP changes{e['macro_match']['scene_calibrated_HGB_allocation']['AP']:.6f}to{e['macro_match']['scene_calibrated_listwise_allocation']['AP']:.6f}; delta{e['paired_AP']['mean_delta']:.6f}, descriptive interval{e['paired_AP']['ci95']}. NLL is worse. **The conditional ranking gain did not produce a convincing unconditional forecast gain.**

**F: untouched second external sample replication.** Game2's official file sizes and Git blob SHAs, parser, source features and target hashes were locked before its download/scoring. Exactly the same models/mapping were applied without Metrica fitting. It replicates geometric transfer but does not constitute a new SoccerTrack sealed test.

**G: matched training-objective control.** B's architecture/ensemble confound was addressed by another 36 CUDA fits with identical MLP, input, positive training scenes, seeds, initializations, batches, epochs and conditional-development-NLL selection. Only training loss/link differ: independent BCE versus scene softmax. Next-BAS switch top1 is {g['next_BAS_1_2s']['switch_top1']['binary']:.4f} versus {g['next_BAS_1_2s']['switch_top1']['softmax']:.4f}; overall conditional top1 is {g['next_BAS_1_2s']['pooled_binary']['scene']['top1']:.4f} versus {g['next_BAS_1_2s']['pooled_softmax']['scene']['top1']:.4f}. Conditional representation/objective matters, but E prevents promoting this to an unconditional deployed improvement.

## Independent domain evidence

{table(["Game","Target","Eligible scenes","Nearest AP","Kinematic AP","Projection switch top1","HGB switch top1"],external_rows)}

Game1's initial mapper used an inconsistent ball derivative interval and missed the source pitch-bounds mask. That invalid version is retained; mapping revision2 reuses the **exact same causal producer** as SoccerTrack. Game2 was frozen only after that repair and evaluated once. Neither is used for model selection. Current-ball-available eligibility and domain taxonomy differences are explicit. Source PASS is not silently combined with SoccerTrack's broader contact-like BAS target. No public raw data/video was repackaged.

## Scene workload and adversarial gates

At 10 scenes/hour and top3 actors, early full context reviews {quota['reviewed_scenes']} scenes/{quota['reviewed_actor_candidates']} actor candidates, recovering {quota['unique_events_recalled']}/{quota['unique_supported_events']} distinct supported annotations ({quota['event_recall']:.2%}). This favors selective offline review, not comprehensive alarms. Annotation-relative misses are not physical false positives.

Final audit caught a real scheduling flaw: version1 removed future-label-unsupported queries. It is preserved as invalid. Version2 separately refits frozen decoders to score **every** held-match current query and never consults future validity for scheduling; unknown selected outcomes are unverifiable, not false. Other model comparisons are unchanged.

Within-scene label permutation controls reduce geometry2s AP to0.0463/0.0512/0.0574, versus0.4632 true labels. Tests cover exact clock/prefix causality, future perturbations, identity permutations, duplicate joins, segment transitions, missing/bounded balls, negative raw scores, window boundaries/ties/censoring, distinct-event deduplication, caches, split integrity and ZIP tampering. Browser tests verify blinding/export and actual clip playback without creating labels.

## Decision

Keep the compact interpretable **opponent/full-context HGB** as the prospective baseline. Recent kinematics and source-team pressure contain useful predictive signal; the limitation is not simply absence of useful position information. Better conditional objectives improve who-to-rank, but do not solve when-an-action-occurs. Exact physical touches, representative negatives, human perception/intent, optimal tactics, deployment alarms and sealed SoccerTrack generalization remain **unvalidated**.

The best next question is: **On independently synchronized, event-complete footage, does opponent-aware causal context retain its1-2s benefit when every candidate interval has a verified outcome?** Human review remains paused; no new review is requested in this handoff.
"""
    md("scientific_report.md",main)
    md("data_quality.md",f"""
# Evidence and data-quality boundaries

- Fourteen native halves audited;25Hz clocks, floor event time, unshifted one-based ball labels and zero-based GSR ordinals are distinct conventions.
- Quantized1.05m by0.68m player coordinates and high repeated-position rates make native derivative spikes unsuitable as physical acceleration truth. Ball-speed outliers and out-of-pitch actor points were measured, not overwritten.
- All seven fitting/development BAS local documents match their official local mirror byte-for-byte; no annotation revision discrepancy was found in that scope. Historical source indices/clip references join exactly, with60completed clips located.
- Source physical sync remains unverified. Fourteen private clips and two directly inspected sample frames do not certify exact actor/contact time. Historical human pass-visible judgments support feasibility in some clips, not exact synchronization.
-2D distance does not distinguish an aerial ball from a foot touch. Repeated algorithm candidate frames are not independent contact events. No generic annotation-negative interval is promoted to a physical negative.
- Fixed current rosters are used; future-only target censoring affects measurable denominators and may be informative. It is never a feature or a final scheduling eligibility mask.
- Metrica's source documents synchronization. Its two anonymous matches are a domain shift, not a SoccerTrack sealed test. Game1 mapping repair is disclosed; Game2 mapping and official Git blob SHAs were frozen before scoring.
-{len(source_hashes)}protected file hashes remain unchanged. Raw footage, human labels, event identities, private feature shards and weights stay local and out of the ZIP.
""")
    md("failure_analysis.md","""
# Failed, negative and inconclusive directions

1. Exact physical-touch ground truth remains BLOCKED: no independent actor/time adjudication or representative verified negatives. The two-reviewer tool is ready but new review is paused.
2. Coarse source coordinates generate implausible short-baseline derivatives. Causal smoothing is only a modest gain; do not call the geometry a precision biomechanics measurement.
3. Conditional GPU ordering improves switch top1, yet source NLL can worsen. Matched BCE/softmax controls support an objective effect; the unconditional factorization has effectively zero paired AP advantage and worse NLL. Do not deploy it as a new decision system.
4. Development Platt calibration has a tiny NLL/ECE effect and slightly worse Brier. It does not repair annotation missingness or domain validity.
5. Source-team context is useful, but no possession state, attack direction, optimal receiver or real perception is measured. Full-context comparisons are not claims of tactical optimality.
6. External constant-velocity projection often beats learned switch top1 despite worse AP. Metric choice changes the apparent winner. Two anonymous matches are not population inference.
7. Small review quotas recover well under1%of distinct supported events. Selective retrieval is defensible; operational binary alarms are not validated.
8. Integration failures were executed and repaired: absent inherited module attribute, missing raw event-parser argument, incorrect external loader/target names, an empty-half target loop, and inherited path precedence shadowing the new ZIP verifier. Regression tests now cover the implicated paths.
9. External mapping1 had a native ball-velocity denominator and pitch-validity mismatch. Its results are explicitly retained as invalid; mapping2 uses the shared native causal producer. Game2 was frozen after this repair.
10. Final scheduling audit found version1 consulted future label eligibility. It is retained as invalid; version2 scores all current queries and separates unknown outcomes. Earlier adaptive evidence hashes resolve to the preserved version, not a rewritten lock.
11. The editor did not discover tests in the ignored output directory; native pytest ran the focused suite. Node was unavailable; integrated-browser synthetic core tests and real local-video load tests were executed instead. No labels were fabricated.
12. Official gated README returned401; no bypass attempted. Wide Open Gazes does not have a verified licensed data download here; SoccerNet-GAR task-specific fields remain unverified. No unnecessary large dataset or protected model retraining occurred.
""")
    md("roadmap.md","""
# Revised research roadmap

## Keep
- Prospective causal ball/player kinematics plus verified source-team opponent pressure.
- Matched geometric and annotation targets, switch-only analysis, honest lead bins and current-roster censoring.
- Frozen external replication with file/parser hashes before scoring; preserve simple physics when it wins.
- Selective offline scene review with all-query causal scheduling and unknown outcomes separated.

## Successful narrow findings
Opponent/full context improves1-2s annotation forecasting on all six known held matches. Temporal geometry transfers to two public Metrica samples. Scene-softmax improves conditional switch ranking under a matched objective control.

## Negative / inconclusive
Smoothing and calibration gains are small. Conditional neural ranking does not yield a convincing unconditional AP gain when scene occurrence is held fixed. Richer conditional heads are not the current bottleneck worth further tuning.

## Paused
New human adjudication remains paused by the user. Synthetic gaze calibration and failed panorama crop-orientation work are not restarted. No detector, world model, S4 or historical receiver model is retrained.

## Next evidence gate
Acquire or authorize one event-complete, independently time-verified match with precise touch actors and a representative negative-window protocol. Before training, lock a new forecast definition and a truly untouched evaluation split. Re-test the existing opponent-aware baseline, not a newly tuned neural model.

Only after this gate consider uncertainty-aware actor allocation jointly with a supported event-occurrence model. Do not market geometry, gaze, possession, intent or optimal decisions as ground truth.
""")
    md("README.md",f"""
# Extended evidence-grounded football mission

Status: **PARTIAL_WITH_NEW_FINDINGS**. Read scientific_report.md, failure_analysis.md and data_quality.md before interpreting scores. Original source data/checkpoints/results are preserved; this package contains no footage, private per-event IDs, human label rows, weights or large feature shards.

## Reproduce in the existing repository

Keep this directory under outputs/{RUN.name}. The existing repository, pinned historical scripts and authorized local datasets are required; this is not a standalone data distribution.

```powershell
& '.\\.venv\\Scripts\\python.exe' '.\\outputs\\{RUN.name}\\resume.py' --all
& '.\\.venv\\Scripts\\python.exe' -m pytest '.\\outputs\\{RUN.name}\\test_evidence.py' -q
& '.\\.venv\\Scripts\\python.exe' '.\\outputs\\{RUN.name}\\verify_package.py' '.\\outputs\\return_{RUN.name}.zip'
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
""")
    md("handoff.md",f"""
# Handoff

**PARTIAL_WITH_NEW_FINDINGS**. Actual elapsed time at report generation: {elapsed/3600:.2f} hours. Retained {len(records)} fold fits, {len(gpu)} on CUDA; {execution['CUDA_optimizer_steps']:,} GPU steps. Seven distinct adaptive branches completed. Main physical labels remain unavailable; no fake completion is claimed.

Best practical result: 1-2 s any-BAS full-context macro AP 0.2346 versus 0.1963 kinematic and 0.0739 distance. Opponent-only 0.2262 is stronger than same-team-only 0.2112. All six held matches improve with full context.

Strong negative result: learned conditional actor allocation does not improve unconditional AP convincingly over equal-scene-probability HGB allocation, despite better switch top1. Retain the simple opponent-aware baseline.

Independent evidence: two pinned official Metrica samples evaluated without external fitting; Game2 parser/input hashes and official blob SHAs locked before scoring. Historical60human pass judgments recovered and precisely scoped; zero new labels, review paused.

Validation:{test_result['tests']}pytest tests;7browser core checks and real4sclip playback;{figures['diagnostics']}diagnostics/eight anonymous diagrams;{len(source_hashes)}protected hashes unchanged. Scheduling version1 and external mapping1 are retained as invalid attempts, with verified repairs.

Return archive: outputs/return_{RUN.name}.zip. Final actual SHA-256/CRC/member hashes are in its adjacent .sha256 and .verification.json; verify_package.py independently recomputes them. Exact reproduction commands are in README.md.

Best next question: does the opponent-aware1-2s benefit survive independently synchronized, event-complete evaluation with verified outcomes for all sampled intervals?
""")
    acceptance={"overall_status":"PARTIAL_WITH_NEW_FINDINGS","executed_scientific_stages":True,"protected_hashes":len(source_hashes),"protected_unchanged":True,"physical_sync":"UNVERIFIED","physical_touch_ground_truth":"BLOCKED","new_human_labels":0,"historical_human_judgments":60,"two_reviewer_workflow":"READY_PAUSED","external_domain_matches":2,"soccertrack_sealed_evaluation":"NO_SEALED_EVALUATION","adaptive_branches":len(execution["adaptive_branches_executed"]),"targeted_tests":test_result,"figures":figures,"causal_scheduler_revision":2,"archive_verification_record":f"../return_{RUN.name}.zip.verification.json","archive_policy":"No raw data, private event mappings, human label rows, video/image footage or model weights."}
    write(RUN/"acceptance.json",acceptance)
    status(10,"COMPLETE",overall_status="PARTIAL_WITH_NEW_FINDINGS",tests=test_result["tests"],protected_unchanged=len(source_hashes))
    stages=read(RUN/"stage_status.json")
    reasons={
        0:"Original source hashes, prior archives, causal cohort and protocol pinned.",
        1:"Source clocks, metadata and private clips audited; independent physical synchronization remains unverified.",
        2:"Historical labels audited and two-reviewer tool verified; new human review remains paused with zero new labels.",
        3:"Three contact-candidate algorithms executed; candidate states are not certified physical touches.",
        4:"Two target families executed with six match-excluded folds and target-validity masks.",
        5:"Three action lead windows and fully joint-stale challenge executed.",
        6:"Matched spatial-context ablations executed; exploratory known-match evidence only.",
        7:"Fixed quotas executed using revision2 causal scheduling, without future validity filtering.",
        8:"Pinned Game1 transfer pilot executed with repaired shared mapping; sealed SoccerTrack evaluation unavailable.",
        9:"Seven adaptive branches executed, including second external match and matched CUDA objective control.",
        10:"Tests, figures and protected-source verification passed; physical and operational questions remain partial."
    }
    for k,reason in reasons.items():
        stages[str(k)]["reason"]=reason
        stages[str(k)]["evidence"]="acceptance.json" if k==10 else f"stage_{k}.json"
    write(RUN/"stage_status.json",stages)
    return execution


def package():
    names={
        "protocol.lock.json","research_questions.md","feature_dictionary.md","dataset_comparison.md","archive_reconciliation.json","clock_and_sync_audit.json","clock_and_sync_audit.md","visual_evidence_notes.json","event_label_coverage.json","contact_candidate_audit.json","targets_local_validity.json","long_targets_local_validity.json","development_targets_local_validity.json","next_interactor_results.json","anticipation_challenges.json","tactical_context_results.json","scene_prioritization.json","scene_prioritization_invalid_future_mask.json","scheduling_integrity_correction.json","conditional_action_type.json","context_interpretation.json","external_sources.json","external_sources_game2.json","external_mapping_repair.json","external_results.json","external_results_game2.json","external_results_mapping1_invalid.json","external_contact_association.json","historical_team_association.json","adaptive_decisions.json","adaptive_quantization.json","adaptive_listwise.json","adaptive_calibration.json","adaptive_domain_resolution.json","adaptive_unconditional_factorization.json","adaptive_objective_control.json","objective_control_comparison.json","game2_replication.lock.json","longer_horizon_results.json","adversarial_negative_controls.json","ui_validation.json","figure_inventory.json","scientific_report.md","failure_analysis.md","data_quality.md","roadmap.md","README.md","handoff.md","execution_manifest.json","acceptance.json","source_integrity_after.json","source_hashes_before.json","test_results.json","stage_status.json","decisions.jsonl"
    }
    names.update({"environment.json","dependency_graph.json","context_feature_dictionary.json","target_validity.json","reviewer_instructions.md","adaptive_completion.json","tests.xml"})
    missing=names-{p.name for p in RUN.iterdir() if p.is_file()}
    if missing:raise RuntimeError("Missing package deliverables: "+str(sorted(missing)))
    files={p:RUN.name+"/"+p.name for p in RUN.iterdir() if p.is_file() and (p.name in names or p.suffix in (".py",".html",".js") or (p.name.startswith("stage_") and p.suffix==".json"))}
    for p in (RUN/"figures").glob("*.png"):files[p]=RUN.name+"/figures/"+p.name
    dependencies=[PRIOR/"mission.py",PRIOR/"adaptive.py",ROOT/"outputs"/"ball_centric_action_autonomy_20261102T222620Z"/"run_research.py",ROOT/"outputs"/"action_opportunity_research_20261102_144500"/"build_cohort.py",ROOT/"action_opportunity_followup_20261102"/"possession_features.py",ROOT/"action_opportunity_followup_20261102"/"followup_common.py",ROOT/"src"/"backbone"/"event_time.py"]
    for p in dependencies:
        if not p.exists():raise RuntimeError("Missing source dependency: "+str(p))
        files[p]=RUN.name+"/dependency_snapshots/"+p.relative_to(ROOT).as_posix()
    manifest={name:digest(path) for path,name in files.items()}
    archive_path=ROOT/"outputs"/f"return_{RUN.name}.zip"
    with zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path,name in sorted(files.items(),key=lambda item:item[1]):archive.write(path,name)
        archive.writestr(RUN.name+"/MANIFEST.sha256.json",json.dumps(manifest,indent=2,sort_keys=True))
    sha=digest(archive_path)
    Path(str(archive_path)+".sha256").write_text(sha+"  "+archive_path.name+"\n",encoding="ascii")
    result=verify(archive_path)
    write(Path(str(archive_path)+".verification.json"),result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--package",action="store_true")
    args=parser.parse_args()
    result=report()
    if args.package:print(json.dumps(package(),indent=2))
    else:print(json.dumps({k:v for k,v in result.items() if k!="fit_records"},indent=2))
