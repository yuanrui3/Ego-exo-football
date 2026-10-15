"""Read frozen receiver prompts; metadata is never used as a model feature."""

import json
import math
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.build_pass_receiver_dataset import TRAIN_MATCHES, VAL_MATCHES
from src.evaluate_pass_receiver_model import LABELS

HEADERS = ("STATE_T_MINUS_0.4S", "STATE_T_MINUS_0.2S", "STATE_T")
ENTITIES = ("ACTOR",) + tuple(LABELS) + tuple(
    f"OPPONENT_{i}" for i in range(1, 12)
)
FEATURE_NAMES = (
    "x_normalized", "y_normalized", "observed", "actor", "teammate",
    "opponent", "goalkeeper", "player",
)
ENTITY_INDEX = {label: index for index, label in enumerate(ENTITIES)}
DEVELOPMENT_MATCH = "132877"
ENTITY_LINE = re.compile(
    r"(ACTOR|TEAMMATE_[1-9][0-9]*|OPPONENT_[1-9][0-9]*) "
    r"team=(left|right) role=(player|goalkeeper) "
    r"x=(NA|-?\d+(?:\.\d+)?) y=(NA|-?\d+(?:\.\d+)?)"
)


def target_index(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"Unexpected receiver label: {label!r}")
    return LABELS.index(label)


def parse_receiver_prompt(prompt: str) -> torch.Tensor:
    """Return [time, fixed anonymous slot, feature], independent of line order."""
    if prompt.count("<|user|>\n") != 1 or prompt.count("<|assistant|>") != 1:
        raise ValueError("Expected exactly one user state and assistant boundary")
    state, assistant = prompt.split("<|user|>\n", 1)[1].split("<|assistant|>")
    if assistant.strip():
        raise ValueError("Receiver prompt contains assistant/target text")
    lines = [line for line in state.splitlines() if line]
    if not lines or not re.fullmatch(r"PERIOD [12]", lines[0]):
        raise ValueError("Expected PERIOD 1 or 2 before temporal states")
    blocks: list[dict[str, tuple[str, str, str, str]]] = []
    for line in lines[1:]:
        if line in HEADERS:
            if len(blocks) >= 3 or line != HEADERS[len(blocks)]:
                raise ValueError(f"Unexpected temporal header: {line}")
            blocks.append({})
            continue
        match = ENTITY_LINE.fullmatch(line)
        if not blocks or match is None:
            raise ValueError(f"Unexpected temporal state line: {line!r}")
        label, team, role, x, y = match.groups()
        if label not in ENTITY_INDEX or label in blocks[-1]:
            raise ValueError(f"Unknown or duplicate entity slot: {label}")
        if (x == "NA") != (y == "NA"):
            raise ValueError(f"Partial missing coordinate pair: {label}")
        blocks[-1][label] = (team, role, x, y)
    if len(blocks) != 3:
        raise ValueError("Expected all three temporal states")
    current = blocks[-1]
    if "ACTOR" not in current or current["ACTOR"][2] == "NA":
        raise ValueError("Actor must be observed at time t")
    actor_team = current["ACTOR"][0]
    tensor = torch.zeros(3, 22, len(FEATURE_NAMES), dtype=torch.float32)
    tensor[:, 0, 3] = 1
    tensor[:, 1:11, 4] = 1
    tensor[:, 11:, 5] = 1
    for time_index, block in enumerate(blocks):
        if set(block) != set(current):
            raise ValueError("Anonymous entity set changed across temporal states")
        for label, (team, role, x, y) in block.items():
            if (team, role) != current[label][:2]:
                raise ValueError(f"Entity team/role changed across time: {label}")
            if (team == actor_team) != (not label.startswith("OPPONENT_")):
                raise ValueError(f"Entity relation contradicts actor team: {label}")
            slot = ENTITY_INDEX[label]
            tensor[time_index, slot, 6 if role == "goalkeeper" else 7] = 1
            if x != "NA":
                xy = (float(x) / 52.5, float(y) / 34.0)
                if not all(math.isfinite(value) for value in xy):
                    raise ValueError(f"Non-finite coordinates: {label}")
                tensor[time_index, slot, :2] = torch.tensor(xy)
                tensor[time_index, slot, 2] = 1
    return tensor


def read_receiver_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"Empty receiver dataset: {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate example IDs: {path}")
    return rows


def validate_match_split(train_rows: list[dict], val_rows: list[dict]) -> None:
    if {str(row["match_id"]) for row in train_rows} != set(TRAIN_MATCHES):
        raise ValueError("Training match set differs from the frozen seven-match split")
    if {str(row["match_id"]) for row in val_rows} != set(VAL_MATCHES):
        raise ValueError("Validation match set differs from the frozen two-match split")
    if {row["id"] for row in train_rows} & {row["id"] for row in val_rows}:
        raise ValueError("Train/validation example IDs overlap")


def development_indices(rows: list[dict]) -> tuple[list[int], list[int]]:
    if {str(row["match_id"]) for row in rows} != set(TRAIN_MATCHES):
        raise ValueError("Development selection requires exactly seven training matches")
    training, development = [], []
    for index, row in enumerate(rows):
        destination = development if str(row["match_id"]) == DEVELOPMENT_MATCH else training
        destination.append(index)
    return training, development


class ReceiverTensorDataset(Dataset):
    def __init__(self, rows: list[dict]):
        if not rows:
            raise ValueError("Cannot tensorize an empty receiver dataset")
        self.ids = [row["id"] for row in rows]
        self.features = torch.stack([parse_receiver_prompt(row["prompt"]) for row in rows])
        self.targets = torch.tensor(
            [target_index(row["response"]) for row in rows], dtype=torch.long
        )
        observed = self.features[torch.arange(len(rows)), -1, self.targets + 1, 2]
        if not bool(observed.all()):
            raise ValueError("True receiver is not observed at time t")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def shuffle_coordinates(features: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Shuffle complete pairs among observed entities independently per state."""
    if features.ndim != 4 or tuple(features.shape[1:]) != (3, 22, 8):
        raise ValueError("Expected [examples, 3, 22, 8] features")
    if features.device.type != "cpu":
        raise ValueError("Coordinate shuffling expects CPU features")
    shuffled = features.clone()
    generator = torch.Generator().manual_seed(seed)
    for example in shuffled:
        for state in example:
            observed = torch.where(state[:, 2] == 1)[0]
            permutation = torch.randperm(len(observed), generator=generator)
            state[observed, :2] = state[observed[permutation], :2].clone()
    return shuffled
