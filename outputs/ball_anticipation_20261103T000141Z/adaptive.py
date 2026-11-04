from __future__ import annotations

import hashlib
import time

import numpy as np

from mission import (
    ROOT, RUN, FIT, SEEDS, PARAMS, data, sequence, feature_sets, load_targets,
    load_aligned, groups, subset, summary, contrast, fit_oof, save, write, read,
    old, decision, digest,
)


def temporal_multihorizon():
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for selected multihorizon branch.")
    d = data()
    dev = data("development")
    seq = sequence()
    ds = sequence("development")
    ys, _ = load_targets(d)
    yd, _, _ = old.build_label_targets(dev, "development")
    names = ("lead_0_1s", "lead_1_2s", "lead_2_3s")
    y = np.column_stack([ys[k] for k in names])
    vy = np.column_stack([yd[k] for k in names])
    x = np.column_stack((d["x"], d["ball"])).astype(np.float32)
    vx = np.column_stack((dev["x"], dev["ball"])).astype(np.float32)
    fingerprint = hashlib.sha256()
    fingerprint.update(b"multihorizon_tcn_v1_10epochs_adamw001_4096")
    for array in (x, vx, seq, ds, y, vy):
        fingerprint.update(array.tobytes())
    source_fingerprint = fingerprint.hexdigest()
    result = {"status": "COMPLETE", "arms": {}, "training": [], "target_names": names}
    for arm in ("instant_tcn", "history_tcn"):
        oof = np.zeros((len(y), 3), np.float32)
        per_seed = np.zeros((3, len(y), 3), np.float32)
        for held in FIT:
            tr = d["match"] != held
            te = ~tr
            xm, xs = old.normalize_train(x[tr])
            sm, ss = old.normalize_train(seq[tr].reshape(-1, seq.shape[-1]))
            xn = old.normalize_values(x, xm, xs)
            vn = old.normalize_values(vx, xm, xs)
            sn = old.normalize_values(seq, sm, ss)
            dn = old.normalize_values(ds, sm, ss)
            if arm == "instant_tcn":
                sn = np.repeat(sn[:, -1:], 6, axis=1)
                dn = np.repeat(dn[:, -1:], 6, axis=1)
            for seed_index, seed in enumerate(SEEDS):
                tag = f"multihorizon_{arm}_{held}_{seed}"
                path = RUN / "predictions_local" / f"{tag}.npz"
                metap = path.with_suffix(".json")
                if path.exists() and metap.exists() and read(metap)["source_fingerprint"] == source_fingerprint and read(metap)["sha256"] == digest(path):
                    with np.load(path, allow_pickle=False) as z:
                        per_seed[seed_index, te] = z["p"]
                    result["training"].append(read(metap))
                    continue
                start = time.perf_counter()
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                class Model(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.static = nn.Sequential(nn.Linear(x.shape[1], 96), nn.GELU(), nn.LayerNorm(96), nn.Linear(96, 48), nn.GELU())
                        self.conv1 = nn.Conv1d(8, 32, 3)
                        self.conv2 = nn.Conv1d(32, 32, 3)
                        self.head = nn.Sequential(nn.Linear(80, 48), nn.GELU(), nn.Linear(48, 3))

                    def forward(self, xx, seqx):
                        a = seqx.transpose(1, 2)
                        a = F.gelu(self.conv1(F.pad(a, (2, 0))))
                        a = F.gelu(self.conv2(F.pad(a, (2, 0))))
                        return self.head(torch.cat((self.static(xx), a[:, :, -1]), dim=1))

                net = Model().cuda()
                opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
                xt = torch.from_numpy(xn[tr]).cuda()
                st = torch.from_numpy(sn[tr]).cuda()
                yt = torch.from_numpy((y[tr] > 0).astype(np.float32)).cuda()
                maskt = torch.from_numpy((y[tr] >= 0).astype(np.float32)).cuda()
                vt = torch.from_numpy(vn).cuda()
                vst = torch.from_numpy(dn).cuda()
                vyt = torch.from_numpy((vy > 0).astype(np.float32)).cuda()
                vmask = torch.from_numpy((vy >= 0).astype(np.float32)).cuda()
                torch.cuda.reset_peak_memory_stats()
                best = float("inf")
                best_state = None
                stale = 0
                curve = []
                steps = 0
                for epoch in range(10):
                    net.train()
                    ix = torch.randperm(len(xt), device="cuda")
                    total = 0
                    for a in range(0, len(ix), 4096):
                        idx = ix[a:a+4096]
                        opt.zero_grad(set_to_none=True)
                        logits = net(xt[idx], st[idx])
                        losses = F.binary_cross_entropy_with_logits(logits, yt[idx], reduction="none")
                        loss = (losses * maskt[idx]).sum() / maskt[idx].sum().clamp(min=1)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite multihorizon loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
                        opt.step()
                        total += float(loss.detach())
                        steps += 1
                    net.eval()
                    with torch.inference_mode():
                        vall = []
                        for a in range(0, len(vt), 8192):
                            vall.append(net(vt[a:a+8192], vst[a:a+8192]))
                        logits = torch.cat(vall)
                        val = float((F.binary_cross_entropy_with_logits(logits, vyt, reduction="none") * vmask).sum() / vmask.sum())
                    curve.append({"epoch": epoch + 1, "training_BCE": total / max(1, int(np.ceil(len(xt) / 4096))), "development_masked_BCE": val})
                    if val < best - 1e-6:
                        best = val
                        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                        best_epoch = epoch + 1
                        stale = 0
                    else:
                        stale += 1
                        if stale >= 2:
                            break
                net.load_state_dict(best_state)
                net.eval()
                pp = []
                with torch.inference_mode():
                    for a in range(0, int(te.sum()), 8192):
                        xx = torch.from_numpy(xn[te][a:a+8192]).cuda()
                        qq = torch.from_numpy(sn[te][a:a+8192]).cuda()
                        pp.append(torch.sigmoid(net(xx, qq)).cpu().numpy())
                p = np.concatenate(pp)
                save(path, p=p)
                per_seed[seed_index, te] = p
                weight_path = RUN / "models_local" / (tag + ".pt")
                weight_path.parent.mkdir(exist_ok=True)
                torch.save({"state_dict": {k: v.cpu() for k, v in best_state.items()}, "normalization": {"x_mean": xm, "x_scale": xs, "sequence_mean": sm, "sequence_scale": ss}}, weight_path)
                meta = {"arm": arm, "held_out_match": held, "seed": seed, "best_epoch": best_epoch, "steps": steps, "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()), "seconds": time.perf_counter() - start, "curve": curve, "selection": "132877 development masked BCE only", "training_matches": [m for m in FIT if m != held], "source_fingerprint": source_fingerprint, "sha256": digest(path)}
                write(metap, meta)
                result["training"].append(meta)
                print(tag, "complete", meta["seconds"], flush=True)
                del net, opt, xt, st, yt, maskt, vt, vst, vyt, vmask
                torch.cuda.empty_cache()
            oof[te] = per_seed[:, te].mean(axis=0)
        result["arms"][arm] = {}
        for j, horizon in enumerate(names):
            eligible = y[:, j] >= 0
            dd = subset(d, eligible)
            dd["target"] = y[eligible, j]
            smry = summary(dd, {arm: oof[eligible, j]}, (arm,))
            smry["individual_seeds"] = {str(seed): summary(dd, {"seed": per_seed[k, eligible, j]}, ("seed",))["macro_match"]["seed"] for k, seed in enumerate(SEEDS)}
            result["arms"][arm][horizon] = smry
        save(RUN / f"{arm}_predictions_local.npz", p=oof, per_seed=per_seed)
    result["reference"] = {horizon: read(RUN / "lead_anticipation_study.json")["horizons"][horizon]["macro_match"] for horizon in names}
    result["matched_comparison"] = {}
    with np.load(RUN / "instant_tcn_predictions_local.npz") as a, np.load(RUN / "history_tcn_predictions_local.npz") as b, np.load(RUN / "anticipation_predictions_local.npz") as c:
        for j, horizon in enumerate(names):
            eligible = y[:, j] >= 0
            dd = subset(d, eligible)
            dd["target"] = y[eligible, j]
            pp = {"instant_tcn": a["p"][eligible, j], "history_tcn": b["p"][eligible, j], "interpretable_hgb": c[horizon + "__interpretable"][eligible]}
            comparison = summary(dd, pp, tuple(pp))
            comparison["history_minus_instant"] = contrast(comparison, "history_tcn", "instant_tcn")
            comparison["history_minus_hgb"] = contrast(comparison, "history_tcn", "interpretable_hgb")
            result["matched_comparison"][horizon] = comparison
    write(RUN / "adaptive_multihorizon.json", result)
    decision("9A", "A new same-capacity causal TCN multihorizon comparison was executed with three matched seeds per fold.", {"macro_AP": {arm: {k: v["macro_match"][arm]["AP"] for k, v in by.items()} for arm, by in result["arms"].items()}})
    return result


def future_proximity_targets(d):
    """Labels only: do not call until causal feature construction is finalized."""
    y = np.full(len(d["target"]), -1, np.int16)
    current_nearest = np.zeros(len(y), bool)
    future_dist = np.zeros(len(y))
    for match in FIT:
        for half in (1, 2):
            mask = (d["match"] == match) & (d["half_code"] == half)
            indices = np.flatnonzero(mask)
            name = "1st" if half == 1 else "2nd"
            cp = ROOT / "outputs" / "trajectory_forecasting_20261016" / "cache" / f"{match}_{name}.npz"
            with np.load(cp, allow_pickle=False) as z:
                raw = {k: z[k] for k in ("observed", "xy", "segments")}
            track = old.load_ball(ROOT / "data" / f"{match}_{name}_ball.npz")
            for q in np.unique(d["query_frame"][mask]):
                rows = indices[d["query_frame"][indices] == q]
                if q + 50 >= len(raw["observed"]):
                    continue
                bxy, _, _, good = track.asof(np.array([q + 50]))
                if not good[0]:
                    continue
                active = np.flatnonzero(raw["observed"][q + 50])
                if not len(active):
                    continue
                distances = np.linalg.norm(raw["xy"][q + 50, active] - bxy[0], axis=1)
                winner = int(active[np.argmin(distances)])
                present = (d["actor_slot_local_only"][rows] == winner) & (d["segment_local_only"][rows] == raw["segments"][q + 50, winner])
                if not present.any():
                    continue
                # Require the full currently eligible roster's future continuity to avoid false negatives from disappeared actors.
                slots = d["actor_slot_local_only"][rows].astype(int)
                continuous = raw["observed"][q + 50, slots] & (raw["segments"][q + 50, slots] == d["segment_local_only"][rows])
                if not continuous.all():
                    continue
                y[rows] = present.astype(int)
                future_dist[rows] = float(distances.min())
                valid_current = d["ball"][rows, 0] > .5
                if valid_current.any():
                    order = np.argmin(np.where(valid_current, d["ball"][rows, 4], 10000))
                    current_nearest[rows[order]] = True
    return y, current_nearest, future_dist


def proximity_transition():
    d = data()
    seq = sequence()
    fs = feature_sets(d, seq)
    # Features are finalized before consulting future-state labels.
    frozen_features = fs["interpretable"].copy()
    y, current, touch_distance = future_proximity_targets(d)
    preds = {}
    for name in ("current", "interpretable"):
        preds[name], meta = fit_oof("future_proximity_" + name, fs[name], d, y)
    np.testing.assert_array_equal(frozen_features, fs["interpretable"])
    preds["nearest_now"] = np.where(d["ball"][:, 0] > .5, -d["ball"][:, 4] * 105, -10000)
    rel = seq[:, -1, 2:4] * [105, 68]
    velocity = (seq[:, -1, 2:4] - seq[:, -2, 2:4]) * [105, 68] / .4
    intercept_dist = np.linalg.norm(rel + velocity * 2, axis=1)
    preds["constant_velocity_intercept"] = np.where((seq[:, -1, 0] > .5) & (seq[:, -2, 0] > .5), -intercept_dist, -10000)
    valid = y >= 0
    dd = subset(d, valid)
    dd["target"] = y[valid]
    result = summary(dd, {k: v[valid] for k, v in preds.items()}, ("current", "interpretable"))
    switch_scene = np.zeros(len(y), bool)
    for rows in groups(d):
        if np.all(valid[rows]):
            winner = rows[y[rows] > 0]
            switch_scene[rows] = current[rows].any() and not current[winner].any()
    switched = valid & switch_scene
    switched_d = subset(d, switched)
    switched_d["target"] = y[switched]
    result["switch_only"] = summary(switched_d, {k: v[switched] for k, v in preds.items()}, ("current", "interpretable"))
    result["support"] = {"eligible_rows": int(valid.sum()), "censored_rows": int((~valid).sum()), "switch_rows": int(switched.sum()), "future_nearest_gt_5m_rows": int(np.sum(valid & (touch_distance > 5)))}
    result["target"] = "Nearest source-tracked player to as-of ball at query+2s, provided future roster continuity and winner membership in current causal candidates are valid. This is future geometric proximity, NOT possession, next touch, or a BAS action."
    result["status"] = "COMPLETE"
    result["contrasts"] = {"interpretable_minus_current": contrast(result, "interpretable", "current")}
    save(RUN / "future_proximity_targets_local.npz", y=y, current_nearest=current, future_distance=touch_distance, **preds)
    write(RUN / "adaptive_proximity_transition.json", result)
    decision("9B", "Execute annotation-independent two-second future-proximity and switch-only comparisons.", {"macro_AP": result["macro_match"], "switch_macro_AP": result["switch_only"]["macro_match"], "support": result["support"]})
    return result


def projected_features(d, seq):
    pair = (seq[:, -1, 0] > .5) & (seq[:, -2, 0] > .5)
    rel = seq[:, -1, 2:4] * [105, 68]
    velocity = (seq[:, -1, 2:4] - seq[:, -2, 2:4]) * [105, 68] / .4
    predicted = rel + velocity * 2
    dist = np.linalg.norm(predicted, axis=1)
    features = np.column_stack((
        pair.astype(float), predicted / [105, 68], dist / 105,
        velocity / 40,
        np.linalg.norm(rel + velocity, axis=1) / 105,
        np.linalg.norm(rel + velocity * 3, axis=1) / 105,
        np.zeros(len(pair)), np.zeros(len(pair)),
    )).astype(np.float32)
    for rows in groups(d):
        dd = np.where(pair[rows], dist[rows], 10000)
        order = np.argsort(dd, kind="stable")
        rank = np.empty(len(rows))
        rank[order] = np.arange(len(rows))
        features[rows, -2] = rank / max(len(rows) - 1, 1)
        features[rows, -1] = (dd - dd.min()) / 105
    features[~pair] = 0
    return np.clip(features, -20, 20)


def intercept_fusion():
    reference = read(RUN / "adaptive_proximity_transition.json")
    lock = {
        "hypothesis": "Explicit fixed-two-second extrapolated geometry can combine the rule's switch top1 strength with the learned dynamics model's AP strength.",
        "evidence": {"switch_top1": reference["switch_only"]["macro_scene"], "switch_AP": reference["switch_only"]["macro_match"]},
        "arms": ["projected_geometry_only_HGB", "interpretable_plus_projection_HGB"],
        "selection": "Fixed 2s target and HGB recipe; no held-out hyperparameter selection, all six folds. AP and top1 both reported, including negative trade-offs.",
    }
    lockp = RUN / "intercept_fusion_lock.json"
    if lockp.exists():
        if read(lockp) != lock:
            raise ValueError("Intercept-fusion lock changed after fitting.")
    else:
        write(lockp, lock)
    branch_doc = RUN / "branch_selection.md"
    text = branch_doc.read_text(encoding="utf-8")
    if "## C: Fixed-horizon physics fusion" not in text:
        branch_doc.write_text(text + "\n## C: Fixed-horizon physics fusion\n\nAfter branch B, the simple extrapolation rule had stronger switched-player top1 than the learned dynamics model, despite lower AP. Before fitting, lock a projection-only HGB and a projection-plus-interpretable HGB at the fixed 2s target. Compare both AP and top1 on all six folds, retaining any trade-off. This is a third substantive comparison chosen from executed evidence, not a larger-network search.\n", encoding="utf-8")
    d = data()
    seq = sequence()
    fs = feature_sets(d, seq)
    projected = projected_features(d, seq)
    with np.load(RUN / "future_proximity_targets_local.npz") as z:
        y = z["y"]
        current = z["current_nearest"]
        preds = {k: z[k] for k in ("current", "interpretable", "nearest_now", "constant_velocity_intercept")}
    valid = y >= 0
    for name, x in (
        ("projection_only_hgb", projected),
        ("projection_fused_hgb", np.column_stack((fs["interpretable"], projected))),
    ):
        preds[name], metadata = fit_oof("adaptive_fusion_" + name, x, d, y)
    dd = subset(d, valid)
    dd["target"] = y[valid]
    probabilistic = tuple(k for k in preds if k not in ("nearest_now", "constant_velocity_intercept"))
    result = summary(dd, {k: p[valid] for k, p in preds.items()}, probabilistic)
    switches = np.zeros(len(y), bool)
    for rows in groups(d):
        if valid[rows].all():
            switches[rows] = current[rows].any() and not current[rows[y[rows] > 0]].any()
    sw = valid & switches
    sd = subset(d, sw)
    sd["target"] = y[sw]
    result["switch_only"] = summary(sd, {k: p[sw] for k, p in preds.items()}, probabilistic)
    result["contrasts"] = {
        "fusion_minus_interpretable": contrast(result, "projection_fused_hgb", "interpretable"),
        "switch_fusion_minus_interpretable": contrast(result["switch_only"], "projection_fused_hgb", "interpretable"),
    }
    result["status"] = "COMPLETE"
    result["feature_manifest"] = ["valid_history_pair", "projected_relative_x_2s", "projected_relative_y_2s", "projected_distance_2s", "relative_vx", "relative_vy", "projected_distance_1s", "projected_distance_3s", "scene_projected_distance_rank_2s", "scene_projected_distance_gap_2s"]
    result["causality"] = "All projection inputs end at the query. These are constant-velocity hypothetical positions, not future source positions. Future labels are loaded separately from the already frozen proxy target."
    save(RUN / "intercept_fusion_predictions_local.npz", **preds)
    write(RUN / "adaptive_intercept_fusion.json", result)
    continuation = read(RUN / "autonomous_continuation.json")
    continuation["branch_C"] = {"status": "COMPLETE", "artifact": "adaptive_intercept_fusion.json"}
    write(RUN / "autonomous_continuation.json", continuation)
    decision("9C", "Follow the actual AP/top1 disagreement with compact fixed-horizon physics features rather than a larger neural network.", {"macro_AP": result["macro_match"], "switch_top1": result["switch_only"]["macro_scene"], "contrasts": result["contrasts"]})
    return result
