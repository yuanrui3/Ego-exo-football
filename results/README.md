# Research evidence by question

Start with the [researcher tutorial](../docs/handover/README.md) for a first checkout. The entries below lead to reports already distributed with this repository. Paths inside historical reports can point to local-only run artifacts.

| Question | First report to open | What to look for |
|---|---|---|
| What is the supported release? | [Tracking backbone release baseline](tracking_backbone_release_baseline.md) | Frozen interfaces, default checkpoints, prerequisites and a reproducible local viewer demo. |
| Which engineering milestones were reached? | [Backbone completion handoff](backbone_completion_20261022_handoff.md) | Full-half replay, fitting index, comparisons and incomplete acceptance items. |
| How good is a future-position forecast? | [Trajectory forecasting](trajectory_forecasting_report.md) and [input leakage audit](trajectory_leakage_audit.md) | Horizon, error in metres, source quantization and limits on contact timing. |
| Does a forecast help receiver prediction? | [Receiver forecast integration](receiver_forecast_integration_report.md) | Covered rows, factual receiver ranking, event timing and reused evaluation matches. |
| Can a camera observation be placed on the pitch? | [Cross-match video handoff](video_player_cross_match_20261028_handoff.md) | Withheld landmarks, global calibration, regional coverage and experimental coordinates. |
| What happens when the tracker loses or swaps a player? | [Perception and identity research update](perception_identity_tolerance_20261030.md) and [tolerance decision](association_tolerance_20261030/decision.json) | Availability, accuracy on surviving queries, rejected approaches and the retained tracker. |

For model version history, read the [v0.4](backbone_v0_4_handoff.md), [v0.5](backbone_v0_5_handoff.md) and [v0.6](backbone_v0_6_handoff.md) handoffs alongside the release guide. For a fresh checkout, distinguish fixture tests under [tests](../tests) from verification that also requires local assets.

[Back to the project overview](../README.md)
