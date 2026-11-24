# A researcher's first half hour with Ego–Exo Football

**Start with one player and one instant.** From an external tracking feed, the project constructs the information that player could plausibly have observed, carries forward a time-limited memory, and studies which football actions can be recognized or evaluated from it. The long-term research question is whether this information can support tactical decisions. The public release supports narrower, directly verifiable tasks.

## Follow one observation through the system

1. Open the [frozen configuration](../../configs/backbone_release_baseline.json). Find the source, the player or actor selection and the supplied query times. Those choices determine what the demo is being asked to predict.
2. Read the [release guide](../../results/tracking_backbone_release_baseline.md). An adapter supplies pitch-coordinate observations. The program simulates a player view and maintains causal memory. G1 estimates hidden players when its history contract is met and uses an explicit fallback when needed. S4 reads the observation/memory stream separately and scores seven action classes at a supplied query. G1's inferred coordinates do not feed S4.
3. Read the [configuration and model manifest](../../configs/backbone_release_manifest.json). It pins the release files and identifies prerequisites that live on the lab machine. The [release runner](../../src/backbone/release_baseline.py) and [verification program](../../src/backbone/verify_release_baseline.py) are the supported entry points.
4. Open the [research evidence index](../../results/README.md) and pick a question. For each result, identify the source of the observations, the prediction target and the scope of the evaluation. A correct historical action label tells you which action occurred. Establishing which action would have worked better needs a comparison of outcomes.

## Verify something on day one

With a source checkout and Python 3, open a Python interpreter from the repository root and paste the following. It verifies only the configuration bytes against the published manifest; it needs no dependencies or match data and does not check the models or run the baseline.

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("configs/backbone_release_manifest.json").read_text())
config = Path(manifest["canonical_configuration"])
assert hashlib.sha256(config.read_bytes()).hexdigest() == manifest["canonical_configuration_sha256"]
print("Release configuration matches the frozen manifest.")
```

### First lab run, not a clean-checkout test

The canonical runner needs the pinned G1 and original S4 checkpoints and their
configs, G1 integration manifest, 118577 first-half tracking cache and metadata,
association manifest, training index and indexed retrospective shard files.
The [configuration](../../configs/backbone_release_baseline.json) lists pinned
paths and hashes; the [manifest](../../configs/backbone_release_manifest.json)
and [release guide](../../results/tracking_backbone_release_baseline.md) give
the remaining shard and environment details. These assets are local-only; the
optional external-metre fixture is not needed for this demo. Install
[`requirements-backbone.txt`](../../requirements-backbone.txt) in the lab
environment. The documented Windows x64 environment used Python 3.14.8 and CPU
inference; a fresh installation and asset transfer were not verified.

Once those prerequisites are present, this is the first supported runnable
entry point, from the repository root in PowerShell:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
.\.venv\Scripts\python.exe -m src.backbone.release_baseline `
  --config configs\backbone_release_baseline.json `
  --output "outputs\release_demo_$stamp"
```

Choose a new, empty output directory and open its `viewer.html` locally. At
sample 238 (frame 1720), toggle the evaluation-only annotation. The published
demo recorded a HIGH_PASS prediction beside a PASS annotation: a classification
error, not a comparison of alternative outcomes. The replay contains only the
recorded play.

The [fixture-only video regressions](../../README.md#local-run-and-verification)
need their installed dependencies but not private data or checkpoints; passing
them does not validate real video positions or the release models. Conversely,
the release guide's `verify_release_baseline` command runs regressions **and a
fresh local-data replay**, so it requires the lab assets and a separate fresh
output directory. Historical viewer and verification links in the guide point
to local-only outputs, not files included in a source checkout.

### Where to look in the source

These are functional responsibilities, not assignments to people:

| Area | Public source |
|---|---|
| Configuration, pinned files and query/annotation join | [`release_baseline.run` and `_verified_queries`](../../src/backbone/release_baseline.py) |
| Tracking and external-metre adapters | [`SoccerTrackSource`](../../src/backbone/source.py), [`ExternalObservationSource`](../../src/backbone/external.py) |
| Simulated view and causal memory | [`SyntheticObservationProvider`](../../src/backbone/observation.py), [`CausalMemory`](../../src/backbone/memory.py) |
| Replay and model interfaces | [`ReplayEngine`](../../src/backbone/pipeline.py), G1 [`ModelSession`](../../src/backbone/models.py), S4 [`ActionModelSession`](../../src/backbone/actions.py) |
| Scoped verification | [`verify`](../../src/backbone/verify_release_baseline.py) |

## Roadmap from the published evidence

This written backlog proposes evidence to resolve specific questions, not completed work,
new release gates, or authorization to run experiments.

| Open question | Where the public record stands | Measurable evidence needed |
|---|---|---|
| Can tracking retain a usable player view? | [Tolerance study](../../results/perception_identity_tolerance_20261030.md): focal gaps reduced support even where survivor predictions were unchanged; simulated visibility is not gaze. | On independently annotated matches, report focal identity switches/fragmentation, supported-query fraction and paired S4 macro-F1 and G1 error against the retained tracker, including unsupported queries and occlusions. |
| Can video provide metric positions? | [Cross-match handoff](../../results/video_player_cross_match_20261028_handoff.md): both global calibrations failed; local validation was too sparse. | Meet the published global gate on each source: at least 50 withheld landmarks, median error ≤2 m and p90 ≤5 m, with spatial coverage; also measure footpoint, synchronization and identity errors before accepting video observations. |
| Can actions be recognized without supplied event times? | The [release](../../results/tracking_backbone_release_baseline.md) scores seven classes only at supplied actor/query times. | On independently annotated, previously unseen matches, report event detection precision/recall, timing error, per-class F1 and unsupported-query rate against a simple baseline. |
| Are forecasts useful at tactical horizons? | [Trajectory report](../../results/trajectory_forecasting_report.md) and [audit](../../results/trajectory_leakage_audit.md): 3-second forecasts use quantized tracks; contact timing is unverified. | On independently verified continuous tracks from unseen matches, compare horizon-specific metre errors and coverage to persistence and constant velocity, including missing-track cases; separately measure synchronized ball/contact errors for any interception claim. |
| Does forecasting improve receiver prediction? | [Integration report](../../results/receiver_forecast_integration_report.md): social features improved matched top-1 by 1.773 percentage points, but the verdict was inconclusive and labels are next-actor proxies. | On unseen matches with verified receiver and contact labels, compare paired top-1/MRR and coverage against the matched no-forecast ranker with uncertainty intervals and a pre-event timing protocol. |
| Can the system recommend an action? | The [action contract](../../results/tracking_backbone_release_baseline.md) exposes uncalibrated class scores, not tactical value. | Define feasible alternatives and a simple comparator; measure independently verified outcomes and uncertainty under a credible prospective or causal comparison of alternatives. Recorded action labels alone cannot identify the best action. |

## Keep the handover useful

Use the [evidence index](../../results/README.md) for current entry points. The model and source manifests govern a local replay. Dated reports preserve decisions that changed the defaults or kept a candidate experimental. When adding a new result, name its match roles, metric, comparator, evidence limitations and remaining decision. Separate a source-only check from a replay that requires lab data.

[Project overview](../../README.md) · [Release guide](../../results/tracking_backbone_release_baseline.md) · [Evidence by question](../../results/README.md)
