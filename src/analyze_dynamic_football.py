"""Incremental, explicitly provisional scientific reporting for the GNN queue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.belief_tracking_dataset import load_npz
from src.dynamic_football_data import FIT_MATCHES, assert_partition
from src.dynamic_football_metrics import paired_match_effects
from src.trajectory_dataset import atomic_replace, file_digest, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.md")
    temporary.write_text(text, encoding="utf-8")
    atomic_replace(temporary, path)


def catalog(output):
    records = {"world": [], "risk": [], "receiver": [], "geometry": [], "stress": []}
    for task in ("world", "risk", "receiver"):
        for path in sorted((output / task).glob("*/result.json")):
            result = read(path)
            row = {key: result[key] for key in ("config", "fitting_matches", "parameters", "active_seconds",
                                                "best_epoch", "epochs_completed", "steps", "peak_cuda_allocated_bytes")}
            row["directory"] = str(path.parent.relative_to(output))
            row["evaluation"] = {}
            for evaluation in path.parent.glob("evaluation*.json"):
                metrics = read(evaluation)
                if task == "world":
                    row["evaluation"][evaluation.stem] = {
                        "learned": metrics["learned"], "baselines": metrics["baselines"],
                        **({"common_2s_previously_seen_targets": metrics["common_2s_previously_seen_targets"]}
                           if "common_2s_previously_seen_targets" in metrics else {})}
                elif task == "risk":
                    row["evaluation"][evaluation.stem] = {
                        name: {"overall": value["overall"], "by_match": {
                            match: {"lane_3m_brier": m["lane"]["3"]["brier"],
                                    "hidden_3m_brier": m["hidden_threat"]["3"]["brier"]}
                            for match, m in value["by_match"].items()},
                               "hidden_threat_original_prevalence": value["hidden_threat_original_prevalence"]}
                        for name, value in metrics.items()}
                else:
                    row["evaluation"][evaluation.stem] = metrics
            records[task].append(row)
    for folder in ("geometry", "stress"):
        for path in sorted((output / folder).glob("*/*.json")):
            if folder == "geometry" and "beliefs" in path.stem:
                continue
            data = read(path)
            if "raw" in data:
                records[folder].append({"directory": str(path.parent.relative_to(output)),
                                        "name": path.stem, **{
                                            name: {"overall": value["overall"],
                                                   "hidden_threat_original_prevalence": value["hidden_threat_original_prevalence"]}
                                            for name, value in data.items() if name in ("raw", "calibrated")}})
    return records


def fold_tables(records, task, value, seed=42):
    rows, by_family = [], {}
    for record in records[task]:
        config = record["config"]
        if config["seed"] != seed:
            continue
        excluded = set(FIT_MATCHES) - set(record["fitting_matches"])
        if len(excluded) != 1:
            continue
        family = config["family"]
        if task == "world":
            if config["seconds"] != 2 or config.get("static") or config.get("distance_only"):
                continue
            match = next(iter(excluded))
            metric = record["evaluation"].get(f"evaluation_{match}", {}).get("learned")
        else:
            metric = record["evaluation"].get("evaluation_trained" if task == "risk" else "evaluation")
        if metric is None:
            continue
        number = value(metric)
        by_family.setdefault(family, {}).setdefault(next(iter(excluded)), []).append(number)
    aggregated = {family: {match: float(np.mean(values)) for match, values in matches.items()}
                  for family, matches in by_family.items()}
    for family, matches in aggregated.items():
        rows.append({"family": family, "held_out_matches": len(matches), "mean": float(np.mean(list(matches.values()))),
                     "per_match": matches, "seed_count_per_match": {
                         match: len(by_family[family][match]) for match in matches}})
    baseline = aggregated.get("G1" if task != "receiver" else "P0", {})
    differences = {family: paired_match_effects(matches, baseline) for family, matches in aggregated.items()}
    return sorted(rows, key=lambda row: row["family"]), differences


def seed_robustness(records):
    result = {}
    getters = {
        "world": lambda m: m["localization_m"],
        "risk": lambda m: m["raw"]["overall"]["lane"]["3"]["brier"],
        "receiver": lambda m: m["overall"]["top1"],
    }
    for task, getter in getters.items():
        result[task] = {}
        for seed in (42, 7, 123):
            table, _ = fold_tables(records, task, getter, seed)
            for row in table:
                result[task].setdefault(row["family"], {})[str(seed)] = row
    return result


def validate_artifacts(output):
    checked = []
    for task in ("world", "risk", "receiver"):
        for result in (output / task).glob("*/result.json"):
            directory = result.parent
            spec = read(directory / "config.json")
            assert_partition(spec["fitting_matches"])
            if not (directory / "final.pt").is_file():
                raise FileNotFoundError(directory / "final.pt")
            for path in directory.glob("predictions*.npz"):
                with np.load(path, allow_pickle=False) as data:
                    for key in data.files:
                        value = data[key]
                        if value.dtype.kind == "f" and not np.isfinite(value).all():
                            raise ValueError(f"Non-finite prediction artifact: {path}:{key}")
                    if task == "risk":
                        if data["raw"].shape != (len(data["rows"]), 7):
                            raise ValueError("Risk prediction shape mismatch")
                    if task == "receiver":
                        p = data["probabilities"]
                        if p.shape != (len(data["rows"]), 10) or not np.allclose(p.sum(-1), 1, atol=.02):
                            raise ValueError("Receiver probability shape/normalization mismatch")
            checked.append(str(directory.relative_to(output)))
    for path in (output / "beliefs").glob("*.json"):
        for source in read(path)["sources"]:
            assert_partition(source["fitting_matches"], (source["predicted_match"],), source["outer_excluded"])
            if file_digest(Path(source["checkpoint"])) != source["sha256"]:
                raise ValueError("Nested belief checkpoint hash changed")
    provenance = read(output / "provenance.json")
    root = Path(provenance["repository_root"])
    for relative, expected in provenance["sha256"].items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"Protected/source file changed: {relative}")
    write_json(output / "validation.json", {"status": "PASSED", "completed_jobs_checked": len(checked),
                                          "directories": checked, "nested_teacher_exclusions": "PASSED",
                                          "protected_source_hashes": "UNCHANGED"})


def plots(root, output, records, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    for task, label in (("world", "Hidden-player localization (m; lower is better)"),
                        ("risk", "3m lane Brier (lower is better)"),
                        ("receiver", "Next-event proxy top-1 (higher is better)")):
        values = tables[task]
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([row["family"] for row in values], [row["mean"] for row in values])
        ax.set_ylabel(label)
        ax.set_title("Primary seed 42; held-out match means")
        fig.tight_layout()
        fig.savefig(folder / f"{task}_six_fold.png", dpi=150)
        plt.close(fig)
    for record in reversed(records["risk"]):
        if record["config"]["family"] != "G4":
            continue
        metric = record["evaluation"].get("evaluation_trained")
        if metric is None:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        for kind in ("raw", "calibrated"):
            data = metric[kind]["hidden_threat_original_prevalence"]["3"]
            bins = data.get("calibration_bins", [])
            if bins:
                ax.plot([b["confidence"] for b in bins], [b["frequency"] for b in bins], "o-", label=kind)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlabel="Predicted hidden-threat probability", ylabel="Observed geometric frequency",
               title=f"{record['directory']} (original prevalence)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "hidden_threat_calibration.png", dpi=150)
        plt.close(fig)
        break
    for record in reversed(records["world"]):
        if record["config"]["family"] != "G4" or record["config"]["seconds"] != 2:
            continue
        if not record["evaluation"]:
            continue
        metric = next(iter(record["evaluation"].values()))["learned"]
        age = [(name, value) for name, value in metric["by_age"].items() if value.get("count")]
        if not age:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [name for name, _ in age]
        axes[0].plot(labels, [value["mean_std_m"] for _, value in age], "o-")
        axes[0].set(xlabel="Time since sight (s)", ylabel="Mean predicted std (m)")
        axes[1].plot(labels, [value["coverage_68"] for _, value in age], "o-", label="68% ellipse")
        axes[1].plot(labels, [value["coverage_95"] for _, value in age], "o-", label="95% ellipse")
        axes[1].axhline(.68, ls="--", color="gray")
        axes[1].axhline(.95, ls="--", color="gray")
        axes[1].set(xlabel="Time since sight (s)", ylabel="Empirical coverage")
        axes[1].legend()
        fig.suptitle(record["directory"])
        fig.tight_layout()
        fig.savefig(folder / "uncertainty_by_memory_age.png", dpi=150)
        plt.close(fig)
        break
    graph_cases(root, output)


def graph_cases(root, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import torch
    from src.models.dynamic_football_gnn import graph_geometry
    old = root / "outputs" / "hidden_defender_20261019"
    comparisons = []
    for match in FIT_MATCHES:
        directory = output / "geometry" / f"exclude{match}"
        needed = [directory / f"{name}.npz" for name in ("G0_last_seen", "G1_gaussian", "G4_gaussian", "G4_beliefs")]
        if not all(path.exists() for path in needed):
            continue
        loaded = [load_npz(path) for path in needed]
        if not np.array_equal(loaded[0]["rows"], loaded[2]["rows"]):
            raise ValueError("Paired graph case rows differ")
        for index, row in enumerate(loaded[0]["rows"]):
            comparisons.append((int(row), match, index, loaded))
    if not comparisons:
        return
    data = load_npz(old / "dataset_v2.npz")
    truth = load_npz(old / "target_only_global_v2.npz")
    y = data["target_lane"][:, 1]
    last_gain = np.asarray([(v[3][0]["raw"][v[2], 1] - y[v[0]]) ** 2
                            - (v[3][2]["raw"][v[2], 1] - y[v[0]]) ** 2 for v in comparisons])
    gaussian_gain = np.asarray([(v[3][1]["raw"][v[2], 1] - y[v[0]]) ** 2
                                - (v[3][2]["raw"][v[2], 1] - y[v[0]]) ** 2 for v in comparisons])
    hidden_effect = np.asarray([v[3][2]["raw"][v[2], 4] for v in comparisons])
    choices = [("gnn_better_than_last_seen" if last_gain.max() > 0 else "no_gnn_win_over_last_seen",
                int(last_gain.argmax())),
               ("social_gaussian_better_than_gnn" if gaussian_gain.min() < 0 else "no_social_gaussian_win_over_gnn",
                int(gaussian_gain.argmin())),
               ("hidden_defender_largest_effect" if hidden_effect.max() > 0 else "no_predicted_hidden_effect",
                int(hidden_effect.argmax()))]
    cases = []
    for name, selected in choices:
        row, match, index, arrays = comparisons[selected]
        scene, candidate = int(data["scene"][row]), data["candidate_xy"][row]
        local = int(np.flatnonzero(arrays[3]["scene_rows"] == scene)[0])
        mean, std = arrays[3]["mean"][local], arrays[3]["std"][local]
        current = data["last_xy"][scene].copy()
        hidden = data["hidden_defender"][scene]
        visible = data["visible"][scene, -1]
        current[hidden] = mean[hidden]
        features = np.zeros((1, 22, 10), np.float32)
        features[0, :, :2] = (current - data["actor_xy"][scene]) / [52.5, 34.]
        features[0, :, 4], features[0, :, 7] = visible, np.where(truth["opponents"][scene], -1, 1)
        actor = np.linalg.norm(current - data["actor_xy"][scene], axis=-1).argmin()
        features[0, actor, 9] = 1
        known = data["mask"][scene, -1]
        _, edges = graph_geometry(torch.from_numpy(features), torch.from_numpy(known[None]),
                                   torch.from_numpy(std[None]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlim=(-53, 53), ylim=(-35, 35), aspect="equal", xlabel="Pitch x (m)", ylabel="Pitch y (m)")
        ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], color="green")
        for i, j in np.argwhere(edges[0].numpy()):
            if i < j:
                ax.plot(current[[i, j], 0], current[[i, j], 1], color="lightgray", lw=.5, zorder=0)
        ax.scatter(current[visible, 0], current[visible, 1], c="blue", label="Visible")
        ax.scatter(current[hidden, 0], current[hidden, 1], c="orange", label="GNN hidden mean")
        actual_hidden = truth["opponents"][scene] & ~visible
        ax.scatter(truth["xy"][scene, actual_hidden, 0], truth["xy"][scene, actual_hidden, 1],
                   marker="x", c="red", label="Hidden truth (EVALUATION ONLY)")
        ax.scatter(data["last_xy"][scene, hidden, 0], data["last_xy"][scene, hidden, 1],
                   marker="+", c="black", label="Last seen")
        for slot in np.flatnonzero(hidden):
            ax.add_patch(Ellipse(current[slot], 2 * 1.5096 * std[slot, 0],
                                2 * 1.5096 * std[slot, 1], alpha=.15, color="orange"))
        actor_xy = data["actor_xy"][scene]
        ax.plot([actor_xy[0], candidate[0]], [actor_xy[1], candidate[1]], "k-", lw=2, label="Candidate lane")
        ax.scatter(*actor_xy, marker="*", s=180, c="purple", label="Actor")
        values = {"truth_lane_3m": float(y[row]), "last_seen": float(arrays[0]["raw"][index, 1]),
                  "social_gaussian": float(arrays[1]["raw"][index, 1]),
                  "gnn_gaussian": float(arrays[2]["raw"][index, 1]),
                  "gnn_hidden_probability": float(arrays[2]["raw"][index, 4])}
        ax.set_title(f"{name}\n{data['ids'][scene]} candidate slot {data['slot'][row]} | {values}")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{name}.png", dpi=160)
        plt.close(fig)
        cases.append({"selection": name, "event_id": str(data["ids"][scene]), "candidate_slot": int(data["slot"][row]),
                      "match": match, "values": values, "selection_is_descriptive_not_an_independent_test": True})
    write_json(output / "failure_cases.json", cases)


def generate_report(root, output, final=False):
    progress = read(output / "progress_summary.json")
    records = catalog(output)
    tables, effects = {}, {}
    tables["world"], effects["world"] = fold_tables(records, "world", lambda m: m["localization_m"])
    tables["risk"], effects["risk"] = fold_tables(records, "risk", lambda m: m["raw"]["overall"]["lane"]["3"]["brier"])
    tables["receiver"], effects["receiver"] = fold_tables(records, "receiver", lambda m: m["overall"]["top1"])
    metrics = {"status": progress["status"], "active_hours": progress["active_seconds"] / 3600,
               "completed_jobs": len(progress["completed_jobs"]), "current_job": progress["current_job"],
               "six_fold_match_means": tables, "paired_match_effects": effects, "catalog": records,
               "primary_seed": 42, "seed_robustness": seed_robustness(records),
               "pending_fit_directories": [key for key in read(output / "plan.json")["fits"]
                                            if not (output / key / "result.json").exists()]
                                          if (output / "plan.json").exists() else [],
               "evidence_levels": {"1": "GSR reconstruction under synthetic motion-heading visibility",
                                   "2": "Withheld GSR passing-lane geometry; not real interception outcomes",
                                   "3": "NOT ESTABLISHED: real perception, pass feasibility, intention, coaching benefit"},
               "human_audit": {"YES": 39, "NO": 20, "UNCERTAIN": 1, "receiver_overlap": 17,
                               "independent_receiver_identities": False, "CSV": "read-only; no relabeling"},
               "scope_gaps": progress.get("optional_not_implemented", []),
               "capacity_compute": read(output / "capacity_compute.json") if (output / "capacity_compute.json").exists() else "PENDING",
               "benchmark_population": read(output / "benchmark_population.json") if (output / "benchmark_population.json").exists() else "PENDING",
               "fov_population_diagnostics": read(output / "stress_data" / "manifest.json") if (output / "stress_data" / "manifest.json").exists() else "PENDING",
               "final_validation": read(output / "validation.json") if (output / "validation.json").exists() else "PENDING"}
    if final:
        plots(root, output, records, tables)
    write_json(output / "metrics.json", metrics)
    write_json(root / "results" / "dynamic_gnn_tactical_metrics.json", metrics)
    report = [
        "# Dynamic GNNs for Partially Observable Football", "",
        f"**Status: {progress['status']}**. Cumulative active queue time: {metrics['active_hours']:.3f} h.",
        f"Completed jobs: {metrics['completed_jobs']}. Current operation: {progress['current_job']}.",
        "", "This report is regenerated incrementally. Missing comparisons are NOT completed experiments.",
        "No performance claim is established until the corresponding six held-out matches and artifacts are present.",
        "", "## Protocol", "",
        "Six fitting matches: 117092, 117093, 118576, 118578, 128057, 132831. "
        "132877 is development ONLY. Retrospective 118577/128058 are supplementary, not new prospective evidence.",
        "All world models share identical newly rebuilt first-sight masks, two-second histories, "
        "AdamW and maximum epoch/patience budgets. Six-second models are a separate ablation. "
        "Evaluation uses one perspective per non-overlapping nine-second span. No post-anchor observations enter features.",
        "Six-second localization also reports the exact common two-second previously-seen target identities "
        "(target-only association, never a model input), so changing hidden-target denominators is not mistaken for a history gain.",
        "G1 reuses the existing Social/causal-temporal WorldBlock. G2 uses edge-conditioned GATv2 over a fixed "
        "history summary and an active capacity-matching feed-forward projection (no temporal recurrence). "
        "G3 uses per-step graph updates followed by a temporal GRU. G4 adds uncertainty-aware "
        "edge geometry and a learned Gaussian posterior. Typed team/actor/lane relationships are edge attributes "
        "(not separate relation-specific parameter matrices). No player-ID embeddings are used.",
        "Widths are matched to G1's parameter count by architecture only, not test performance. G4/G5/G6 "
        "risk heads have identical capacity. G6 actively feeds observations through its extra branches, "
        "rather than leaving them as unused parameters.",
        "Risk teachers are nested: fitting-example beliefs exclude both that example's match and the outer test match. "
        "All teachers use development-only selection. Decision seeds share the frozen seed-42 upstream teachers; "
        "world-model seed variability is evaluated separately.",
        "The hidden-threat population includes EVERY original candidate lane with a hidden tracked defender, "
        "including negatives. No test rebalancing. Original-prevalence precision, AP, ROC, calibration and "
        "development-selected operating points are saved. The historical 204-positive-only stratum is descriptive only.",
        "Direct G0 last-seen/CV and G1/GNN point/Gaussian/anonymous geometry are serious controls. "
        "G7 is target-only evaluation. Anonymous occupancy specifically targets NEVER-SEEN DEFENDERS; "
        "it is not a fixed fraction of all-player occupancy. Monte Carlo geometry assumes independent entities/cells.",
        "All receiver models, including fresh P0, share the exact same strictly-as-of three snapshots (0.4s), "
        "ten as-of candidates, exclusions and next-same-team-actor labels. This is a full-observation control, "
        "not evidence of partial-observation receiver improvement. G4/G6 cannot demonstrate a hidden-belief "
        "benefit in that control. Historical post-BAS P0 scores are not directly compared.",
        "", "## Primary six-fold results", "",
        "Primary tables use seed 42 consistently, never a changing mixture of available seeds. "
        "Separate seed-7/123 tables are saved under `seed_robustness`; incomplete seed folds remain explicitly partial.",
    ]
    for task, label in (("world", "Hidden-player localization (m, lower)"),
                        ("risk", "Raw 3m lane Brier (lower)"), ("receiver", "Receiver proxy top-1 (higher)")):
        report += [f"### {label}", "", "| Model | Held-out matches | Match mean | Seeds per match |",
                   "|---|---:|---:|---|"]
        for row in tables[task]:
            report.append(f"| {row['family']} | {row['held_out_matches']}/6 | {row['mean']:.5f} | "
                          f"{row['seed_count_per_match']} |")
        if not tables[task]:
            report.append("| Pending | 0/6 | Not measured | Not measured |")
        report.append("")
    report += ["## Compute and capacity", "", "| Task/model | Fits | Parameters | Training hours |",
               "|---|---:|---|---:|"]
    for task in ("world", "risk", "receiver"):
        for family in sorted({row["config"]["family"] for row in records[task]}):
            values = [row for row in records[task] if row["config"]["family"] == family]
            report.append(f"| {task}/{family} | {len(values)} | {sorted({row['parameters'] for row in values})} | "
                          f"{sum(row['active_seconds'] for row in values)/3600:.3f} |")
    report += ["", "Actual training/evaluation artifacts contain raw and calibrated results separately, thresholds "
               "2/3/5m, pressure MAE, candidate-relative risk ranking, memory-age and visibility strata, "
               "coverage and occupied-cell AP/Brier. Detailed metrics are in [metrics.json](../outputs/"
               "dynamic_gnn_tactical_20261019/metrics.json). Primary paired effects use a consistent seed 42; "
               "additional seeds are summarized separately. The six matches, not seeds, are the sampling units.",
               "", "## Questions and conclusions", ""]
    complete_world = {row["family"]: row for row in tables["world"] if row["held_out_matches"] == 6}
    if "G1" in complete_world and "G3" in complete_world:
        difference = complete_world["G3"]["mean"] - complete_world["G1"]["mean"]
        report.append(f"1. G3 minus G1 hidden localization is {difference:+.5f} m across six match means. "
                      + ("The point estimate favors the GNN; inspect paired uncertainty before claiming generalization."
                         if difference < 0 else "The GNN does NOT improve the primary localization point estimate."))
    else:
        report.append("1. GNN versus Social reconstruction: pending all six held-out matches.")
    report += [
        "2. Explicit relationships: inspect dynamic/static/distance-only matched ablations; attention weights are not explanations.",
        "3. GNN versus geometric baselines: compare direct geometry, not only trained-head scores. "
        "A larger neural network is not a success if it loses to Gaussian geometry.",
        "4. Belief utility: compare separately trained G6 and frozen G4 learned/last/CV/shuffled/empty substitutions. "
        "Frozen substitutions keep the original G4 calibration, avoiding intervention-specific retuning.",
        "5. Generalization: paired six-match effects and exploratory match bootstrap intervals are in metrics; "
        "retrospective matches are supplementary and seeds are not independent matches.",
        "6. Robustness: only completed, explicitly labeled ablation artifacts count as evidence; "
        "do not infer FOV or real tracking robustness from architecture alone.",
        "7. Receiver scores measure the next-event actor proxy ONLY. The human CSV supplies no receiver identities.",
        "8. Publishability/novelty is NOT established by implementing GATv2 or obtaining a proxy gain. "
        "A publishable claim needs reproducible cross-match advantages over strong geometry and independent real-world validation.",
        "9. Real egocentric masks, validated facing/identity/synchronization, actual interceptions and intended receivers remain missing.",
        "", "## Evidence boundaries and prior findings", "",
        "- Level 1: GSR state reconstruction under synthetic visibility. Level 2: GSR geometric labels. Level 3: NOT established.",
        "- Motion heading is not gaze. Ball omitted: no independently reliable causal ball observation.",
        "- Quantized GSR and segmentation limit velocity/position precision. Tracked full state is not guaranteed physical full state.",
        "- Existing audit: 39 YES / 20 NO / 1 UNCERTAIN, only 17 reviewed receiver examples. No labels were changed.",
        "- Prior clearance filter failed all seven unqualified reviewed clearances; it is NOT used.",
        "- Prior neural hidden-defender beliefs did not beat matched controls; this experiment must not assume they will.",
        "- Historical Gaussian lane Brier 0.02127 versus last-seen 0.02787 used an earlier protocol; "
        "these are context, not directly comparable primary six-fold results.",
        "- The orientation follow-up verified software extraction/provenance, not semantic facing or physical synchronization.",
        "- Related reports: [hidden-defender study](./hidden_defender_reasoning_report.md), "
        "[integrated orientation follow-up](./integrated_video_hidden_defender_report.md), "
        "[PASS-label-noise audit](./pass_label_noise_report.md).",
        "", "## Reproduction and artifacts", "",
        "Run `python -m train.run_dynamic_football` from the repository root. The queue resumes optimizer, RNG, "
        "batch offset, best checkpoint and cumulative 30-hour active budget. Successful fits are reused. "
        "Windows advisory locking prevents a second queue, and atomic retrying writes tolerate transient readers.",
        "Use `python -m train.run_dynamic_football --report-only` to regenerate this provisional report.",
        "All source snapshots, protected-file hashes, configuration, training logs, checkpoints, predictions and "
        "nested teacher manifests are under the isolated output directory. No historical outputs are overwritten.",
    ]
    if final:
        report += ["", "## Plots and case analysis", "",
                   "See the output `plots` directory and `failure_cases.json`. Cases are selected descriptively "
                   "from paired held-out predictions, including unfavorable GNN cases; target positions are "
                   "used ONLY to draw evaluation truth, not to construct inference graphs."]
    report += ["", "## Pending or scope-limited work", ""]
    if progress["status"] not in ("COMPLETE", "COMPLETE_WITH_DISCLOSED_SCOPE_GAPS"):
        report.append(f"The queue is {progress['status']}. Not all scheduled fits/ablations/receiver comparisons are complete.")
    for gap in metrics["scope_gaps"]:
        report.append(f"- {gap}")
    text = "\n".join(report) + "\n"
    atomic_text(root / "results" / "dynamic_gnn_tactical_report.md", text)
    output_text = text.replace("../outputs/dynamic_gnn_tactical_20261019/metrics.json", "./metrics.json")
    for name in ("hidden_defender_reasoning_report.md", "integrated_video_hidden_defender_report.md",
                 "pass_label_noise_report.md"):
        output_text = output_text.replace(f"](./{name})", f"](../../results/{name})")
    atomic_text(output / "report.md", output_text)
    return metrics
