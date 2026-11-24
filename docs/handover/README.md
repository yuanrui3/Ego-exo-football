# A researcher's first half hour with Ego–Exo Football

**Start with one player and one instant.** From an external tracking feed, the project constructs the information that player could plausibly have observed, carries forward a time-limited memory, and studies which football actions can be recognized or evaluated from it. The long-term research question is whether this information can support tactical decisions. The public release supports narrower, directly verifiable tasks.

## Follow one observation through the system

1. Open the [frozen configuration](../../configs/backbone_release_baseline.json). Find the source, the player or actor selection and the supplied query times. Those choices determine what the demo is being asked to predict.
2. Read the [release guide](../../results/tracking_backbone_release_baseline.md). An adapter supplies pitch-coordinate observations. The program simulates a player view and maintains causal memory. G1 estimates hidden players when its history contract is met and uses an explicit fallback when needed. S4 reads the observation/memory stream separately and scores seven action classes at a supplied query. G1's inferred coordinates do not feed S4.
3. Read the [configuration and model manifest](../../configs/backbone_release_manifest.json). It pins the release files and identifies prerequisites that live on the lab machine. The [release runner](../../src/backbone/release_baseline.py) and [verification program](../../src/backbone/verify_release_baseline.py) are the supported entry points.
4. Open the [research evidence index](../../results/README.md) and pick a question. For each result, identify the source of the observations, the prediction target and the scope of the evaluation. A correct historical action label tells you which action occurred. Establishing which action would have worked better needs a comparison of outcomes.

## Verify something on day one

With a source checkout and Python 3, run the following from the repository root in a Python interpreter. It verifies the configuration bytes against the published manifest and needs no match data.

```python
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("configs/backbone_release_manifest.json").read_text())
config = Path(manifest["canonical_configuration"])
assert hashlib.sha256(config.read_bytes()).hexdigest() == manifest["canonical_configuration_sha256"]
print("Release configuration matches the frozen manifest.")
```

**When you have lab access**, inventory the checkpoints, indexes and tracking sources listed in the manifest. Then follow the Windows commands in the release guide and choose a fresh output directory. Use the generated viewer to inspect a query with the evaluation-only annotation hidden and then displayed. The guide documents a HIGH_PASS prediction beside a recorded PASS. That disagreement is an example of a classification error. The replay contains only the recorded play.

## Roadmap from the published evidence

| Milestone | Where the public record stands | What the next researcher should verify |
|---|---|---|
| Track a player's observations and memory | A frozen tracking-based G1/original S4 release exists. A separate completion study replayed two full tracking halves covering 93.45 source minutes. | Inspect which fields are observed, inferred and unknown at the query. Verify fallback and unsupported-query behavior with the release guide. |
| Connect real video to metric pitch positions | The global validation gate failed on both published cross-match examples. Their median withheld-landmark errors exceeded 8 m. | Trace the coordinate system, landmark support and field coverage in the cross-match handoff before treating an experimental projection as a measured position. |
| Recognize and forecast football events | The release classifies event-conditioned actions. Receiver ranking and trajectory forecasting have separate datasets and evaluation protocols. | Check who supplied the actor and clock, where the labels originate and which matches were previously used. |
| Recommend an action | The repository's action contract exposes supported outputs and unknowns. Class scores from different branches cannot establish the best action. | Define feasible actions, a simple comparator and an independently measured outcome before interpreting a score as tactical benefit. |

## Keep the handover useful

Use the [evidence index](../../results/README.md) for current entry points. The model and source manifests govern a local replay. Dated reports preserve decisions that changed the defaults or kept a candidate experimental. When adding a new result, name its match roles, metric, comparator, evidence limitations and remaining decision. Separate a source-only check from a replay that requires lab data.

[Project overview](../../README.md) · [Release guide](../../results/tracking_backbone_release_baseline.md) · [Evidence by question](../../results/README.md)
