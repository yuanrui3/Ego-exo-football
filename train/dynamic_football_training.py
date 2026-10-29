"""Resume-safe matched graph training and target-free inference."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from src.dynamic_football_data import WORLD_INPUTS, assert_partition
from src.models.candidate_ranker_receiver import CandidateRankerReceiver, fit_statistics
from src.models.dynamic_football_gnn import GraphDecisionModel, GraphWorldModel
from src.models.entity_transformer_receiver import EntityTransformerReceiver
from src.trajectory_dataset import check_deadline, write_json
from train.run_trajectory_forecasting import restore_rng, rng_state, save_torch, seed_all


def matched_config(family, task="world", seed=42, seconds=2, **overrides):
    base = {"family": family, "width": 128, "layers": 3, "seed": seed,
            "seconds": seconds, "batch_size": 48 if seconds == 2 else 16,
            "lr": .0003, "max_epochs": 25, "patience": 5}
    if family in ("P0", "Entity"):
        return {**base, **overrides}
    factory = GraphWorldModel if task == "world" else lambda c: GraphDecisionModel(c, task)
    reference = sum(p.numel() for p in factory({**base, "family": "G1"}).parameters())
    candidates = []
    for width in range(80, 197, 4):
        candidate = {**base, "width": width}
        count = sum(p.numel() for p in factory(candidate).parameters())
        candidates.append((abs(count / reference - 1), width))
    base["width"] = min(candidates)[1]
    return {**base, **overrides}


def world_loss(output, batch):
    selected = batch["position_mask"].bool()
    logs = output["log_std"].float()
    residual = (output["mean"].float() - batch["position"].float()) * (-logs).exp()
    nll = (.5 * residual.square() + logs + .5 * math.log(2 * math.pi)).sum(-1)
    location = (nll * selected).sum() / selected.sum().clamp_min(1)
    return (F.binary_cross_entropy_with_logits(output["occupancy_logits"], batch["occupancy"])
            + F.binary_cross_entropy_with_logits(output["anonymous_defender_logits"], batch["anonymous_defender"])
            + .04 * location)


def world_forward(model, batch):
    return model(**{key: batch[key] for key in WORLD_INPUTS})


def decision_batch(data, beliefs, rows, kind, task="risk", device="cuda", permutation=None):
    rows = np.asarray(rows, dtype=np.int64)
    scenes = data["scene"][rows] if task == "risk" else rows
    x, mask = data["x"][scenes].copy().astype(np.float32), data["mask"][scenes].copy()
    last = x[:, -1]
    hidden = (mask[:, -1] & ~(last[..., 4] > .5))
    if task == "risk":
        hidden &= last[..., 7] < -.5
    memory = np.zeros((len(rows), 22, 7), np.float32)
    occ = np.zeros((len(rows), 294), np.float32)
    if kind == "observed":
        # Capacity control: all branches process observations, but no hidden belief.
        hidden = mask[:, -1] & (last[..., 4] > .5)
        memory[..., :2] = last[..., :2]
        memory[..., 2:4] = .15
        memory[..., 4:] = last[..., 7:10]
        memory[~hidden] = 0
        observed_flat = (last * hidden[..., None]).reshape(len(rows), -1)
        occ[:, :observed_flat.shape[1]] = observed_flat
    elif kind != "empty":
        center = last[..., :2].copy()
        age = last[..., 6] * 8
        sigma = np.broadcast_to((1.5 + 1.1 * age)[..., None] / 10, (len(rows), 22, 2)).copy()
        if kind == "constant_velocity":
            center += last[..., 2:4] * 12 * age[..., None] / [52.5, 34.]
        elif kind in ("learned", "shuffled", "point"):
            if beliefs is None:
                raise ValueError("Learned decision inputs require explicit out-of-fold beliefs")
            donor = permutation[scenes] if kind == "shuffled" else scenes
            if not beliefs["covered"][donor].all():
                raise ValueError("Missing out-of-fold scene beliefs")
            center = (beliefs["mean"][donor] - data["actor_xy"][donor, None]) / [52.5, 34.]
            sigma = beliefs["std"][donor] / 10
            occ = beliefs["anonymous"][donor].copy()
            if kind == "shuffled":
                donor_x = data["x"][donor, -1]
                hidden = data["mask"][donor, -1] & ~(donor_x[..., 4] > .5) & (donor_x[..., 7] < -.5)
                age = donor_x[..., 6] * 8
            if kind == "point":
                sigma = np.full_like(sigma, .01)
        memory[..., :2], memory[..., 2:4] = center, sigma
        memory[..., 4], memory[..., 5] = age / 8, 1
        memory[..., 6] = np.linalg.norm(center, axis=-1)
        memory[~hidden] = 0
    else:
        hidden[:] = False
    cpu = {"x": x, "mask": mask, "candidate_slot": data["slot"][rows] if task == "risk" else data["candidate_slot"][rows],
           "belief": memory, "belief_mask": hidden, "occupancy": occ}
    return {key: torch.from_numpy(np.asarray(value)).to(device) for key, value in cpu.items()}


def belief_kind(family):
    return {"G4": "learned", "G5": "shuffled", "G6": "observed"}.get(family, "last_seen")


def shuffle_mapping(data, seed):
    result = np.arange(len(data["matches"]))
    hidden_count = (data["mask"][:, -1] & ~(data["x"][:, -1, :, 4] > .5)
                    & (data["x"][:, -1, :, 7] < -.5)).sum(-1)
    for match in np.unique(data["matches"]):
        for count in np.unique(hidden_count[data["matches"] == match]):
            ix = np.flatnonzero((data["matches"] == match) & (hidden_count == count))
            order = np.random.default_rng(seed).permutation(ix)
            if len(order) > 1:
                result[order] = np.roll(order, 1)
    return result


def decision_loss(prediction, target, task):
    if task == "risk":
        return (F.binary_cross_entropy_with_logits(prediction[:, :6], target[:, :6])
                + .2 * F.smooth_l1_loss(prediction[:, 6], target[:, 6] / 10))
    return F.cross_entropy(prediction, target.long())


def array_targets(data, rows, task):
    if task == "risk":
        return np.c_[data["target_lane"][rows], data["target_hidden"][rows], data["target_pressure"][rows]].astype(np.float32)
    return data["target"][rows]


def receiver_forward(model, data, rows, config, device):
    family = config["family"]
    if family in ("P0", "Entity"):
        return model(torch.as_tensor(data["states"][rows], device=device))
    n = len(rows)
    repeated = np.repeat(rows, 10)
    expanded = {**data, "candidate_slot": np.ones(len(data["ids"]), dtype=np.int64)}
    # decision_batch uses example indices for scenes, so candidate slots are set explicitly.
    batch = decision_batch(expanded, None, repeated, "observed", "receiver", device)
    batch["candidate_slot"] = torch.arange(1, 11, device=device).repeat(n)
    prediction = model(**batch).reshape(n, 10)
    valid = torch.as_tensor(data["mask"][rows, -1, 1:11], device=device)
    return prediction.masked_fill(~valid, -10000)


def build_model(config, task, data=None, fitting=None):
    if task == "world":
        return GraphWorldModel(config)
    if config["family"] == "P0":
        if fitting is None:
            raise ValueError("P0 normalization requires explicit fitting-only indices")
        statistics = fit_statistics(torch.from_numpy(data["states"][fitting]))
        return CandidateRankerReceiver(statistics)
    if config["family"] == "Entity":
        return EntityTransformerReceiver()
    return GraphDecisionModel(config, task)


def fit(directory, config, task, training, development, fitting_matches, deadline,
        data=None, beliefs=None, device="cuda", stop_after_steps=None, heartbeat=None):
    """Offset, optimizer and RNG checkpoints resume without repeating completed batches."""
    directory.mkdir(parents=True, exist_ok=True)
    assert_partition(fitting_matches)
    spec = {"config": config, "task": task, "fitting_matches": list(fitting_matches),
            "development_matches": ["132877"], "training_examples": len(training),
            "development_examples": len(development)}
    path = directory / "config.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError(f"Resume configuration mismatch: {directory}")
    write_json(path, spec)
    if (directory / "result.json").exists():
        if not (directory / "final.pt").exists():
            raise FileNotFoundError("Completed checkpoint missing")
        return json.loads((directory / "result.json").read_text())
    seed_all(config["seed"])
    model = build_model(config, task, data, training if task != "world" else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.0001)
    state = {"epoch": 0, "offset": 0, "steps": 0, "best": 1e20, "best_epoch": 0,
             "stale": 0, "history": [], "loss_sum": 0., "loss_count": 0, "active_seconds": 0.}
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        restore_rng(saved["rng"])
    started, last_saved = time.monotonic(), time.monotonic()
    previous_active = state["active_seconds"]
    permutation = shuffle_mapping(data, config["seed"]) if task == "risk" else None
    def save():
        state["active_seconds"] = previous_active + time.monotonic() - started
        save_torch(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "state": state, "rng": rng_state()})
        write_json(directory / "live.json", state)
        if heartbeat is not None:
            heartbeat()
    def get_batches(population, start=0, order=None):
        if task == "world":
            indices = np.arange(len(population)) if order is None else order
            loader = DataLoader(Subset(population, indices[start:].tolist()), batch_size=config["batch_size"],
                                shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(0))
            for cpu in loader:
                yield {key: value.to(device) for key, value in cpu.items()}
        else:
            indices = np.asarray(population) if order is None else order
            for offset in range(start, len(indices), config["batch_size"]):
                yield indices[offset:offset + config["batch_size"]]
    def forward_loss(batch):
        if task == "world":
            return world_loss(world_forward(model, batch), batch)
        target = torch.as_tensor(array_targets(data, batch, task), device=device)
        prediction = (model(**decision_batch(data, beliefs, batch, belief_kind(config["family"]),
                                             device=device, permutation=permutation)) if task == "risk"
                      else receiver_forward(model, data, batch, config, device))
        return decision_loss(prediction, target, task)
    try:
        while state["epoch"] < config["max_epochs"] and state["stale"] < config["patience"]:
            indices = np.arange(len(training)) if task == "world" else np.asarray(training)
            order = np.random.default_rng(config["seed"] + state["epoch"]).permutation(indices)
            model.train()
            for batch in get_batches(training, state["offset"], order):
                check_deadline(deadline)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    loss = forward_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {task} training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                count = len(batch["x"]) if task == "world" else len(batch)
                state["offset"] += count
                state["steps"] += 1
                state["loss_sum"] += float(loss.detach()) * count
                state["loss_count"] += count
                if time.monotonic() - last_saved >= 60:
                    save()
                    last_saved = time.monotonic()
                if stop_after_steps is not None and state["steps"] >= stop_after_steps:
                    save()
                    return {"interrupted_for_test": True, "steps": state["steps"]}
            model.eval()
            losses, counts = [], []
            with torch.no_grad():
                for batch in get_batches(development):
                    check_deadline(deadline)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                        loss = forward_loss(batch)
                    count = len(batch["x"]) if task == "world" else len(batch)
                    losses.append(float(loss) * count)
                    counts.append(count)
            if not counts:
                raise ValueError("Empty development dataset")
            value = sum(losses) / sum(counts)
            state["epoch"] += 1
            state["offset"] = 0
            state["history"].append({"epoch": state["epoch"], "development_loss": value,
                                     "training_loss": state["loss_sum"] / state["loss_count"]})
            state["loss_sum"], state["loss_count"] = 0., 0
            if value < state["best"]:
                state["best"], state["best_epoch"], state["stale"] = value, state["epoch"], 0
                save_torch(directory / "best.pt", {"model": model.state_dict(), "config": config})
            else:
                state["stale"] += 1
            save()
            print(f"{directory.name}: epoch {state['epoch']} dev={value:.5f}", flush=True)
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        save_torch(directory / "final.pt", best)
        result = {**spec, "best_epoch": state["best_epoch"], "development_loss": state["best"],
                  "epochs_completed": state["epoch"], "steps": state["steps"], "history": state["history"],
                  "active_seconds": previous_active + time.monotonic() - started,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0}
        write_json(directory / "result.json", result)
        return result
    except BaseException:
        save()
        raise


@torch.no_grad()
def infer_world(directory, dataset, deadline, device="cuda"):
    spec = json.loads((directory / "config.json").read_text())
    model = build_model(spec["config"], "world").to(device).eval()
    model.load_state_dict(torch.load(directory / "final.pt", weights_only=False, map_location="cpu")["model"])
    output = {"mean": [], "std": [], "occupancy": [], "anonymous": []}
    for cpu in DataLoader(dataset, batch_size=spec["config"]["batch_size"], num_workers=0):
        check_deadline(deadline)
        batch = {key: value.to(device) for key, value in cpu.items()}
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            prediction = world_forward(model, batch)
        for key, value in (("mean", prediction["mean"]), ("std", prediction["log_std"].exp()),
                           ("occupancy", prediction["occupancy_logits"].sigmoid()),
                           ("anonymous", prediction["anonymous_defender_logits"].sigmoid())):
            output[key].append(value.float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in output.items()}


class ArrayWorldDataset(torch.utils.data.Dataset):
    def __init__(self, arrays, indices):
        self.arrays, self.indices = arrays, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        return {key: torch.as_tensor(self.arrays[key][row]) for key in WORLD_INPUTS}


@torch.no_grad()
def infer_decision(directory, data, beliefs, rows, task, deadline, kind=None, device="cuda"):
    spec = json.loads((directory / "config.json").read_text())
    config = spec["config"]
    fitting = np.flatnonzero(np.isin(data["candidate_match"] if task == "risk" else data["matches"],
                                    spec["fitting_matches"]))
    model = build_model(config, task, data, fitting).to(device).eval()
    model.load_state_dict(torch.load(directory / "final.pt", weights_only=False, map_location="cpu")["model"])
    output = []
    permutation = shuffle_mapping(data, config["seed"]) if task == "risk" else None
    for offset in range(0, len(rows), config["batch_size"]):
        check_deadline(deadline)
        local = rows[offset:offset + config["batch_size"]]
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            if task == "risk":
                result = model(**decision_batch(data, beliefs, local, kind or belief_kind(config["family"]),
                                                device=device, permutation=permutation))
                result = torch.cat((result[:, :6].sigmoid(), result[:, 6:] * 10), -1)
            else:
                result = receiver_forward(model, data, local, config, device).softmax(-1)
        output.append(result.float().cpu().numpy())
    return np.concatenate(output)
