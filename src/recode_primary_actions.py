import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_TO_PRIMARY_ACTION = {
    "Pass": "PASS",
    "High Pass": "HIGH_PASS",
    "Drive": "DRIVE",
    "Cross": "CROSS",
    "Shot": "SHOT",
    "Header": "HEADER",
    "Ball Player Block": "DEFENSIVE_INTERVENTION",
    "Player Successful Tackle": "DEFENSIVE_INTERVENTION",
}
EXCLUDED_LABELS = {"Out", "Goal", "Throw In", "Free Kick"}
EXPECTED_LABELS = set(SOURCE_TO_PRIMARY_ACTION) | EXCLUDED_LABELS
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def recode_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    """Map known source labels and omit non-primary event/context labels."""
    unexpected = sorted(
        {record.get("response") for record in records} - EXPECTED_LABELS
    )
    if unexpected:
        raise ValueError(f"Unexpected source action labels: {unexpected}")

    recoded: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    excluded_count = 0
    for record in records:
        source_label = record["response"]
        primary_label = SOURCE_TO_PRIMARY_ACTION.get(source_label)
        if primary_label is None:
            excluded_count += 1
            continue

        output_record = dict(record)
        output_record["source_label"] = source_label
        output_record["response"] = primary_label
        recoded.append(output_record)
        class_counts[primary_label] += 1

    return recoded, class_counts, excluded_count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or not isinstance(
                record.get("response"), str
            ):
                raise ValueError(
                    f"Expected a JSON object with a string response in "
                    f"{path} at line {line_number}"
                )
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_split(source_path: Path, output_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError(f"Refusing to overwrite source dataset: {source_path}")

    source_records = read_jsonl(source_path)
    output_records, class_counts, excluded_count = recode_records(source_records)
    write_jsonl(output_path, output_records)

    return {
        "input_examples": len(source_records),
        "output_examples": len(output_records),
        "excluded_examples": excluded_count,
        "class_counts": dict(sorted(class_counts.items())),
        "match_ids": sorted({record.get("match_id") for record in source_records}),
        "prompts_unchanged": all(
            source["prompt"] == output["prompt"]
            for source, output in zip(
                (record for record in source_records if record["response"] not in EXCLUDED_LABELS),
                output_records,
            )
        ),
    }


def main() -> None:
    splits = (
        ("train", "train_temporal.jsonl", "train_temporal_7class.jsonl"),
        ("validation", "val_temporal.jsonl", "val_temporal_7class.jsonl"),
    )
    for split_name, source_name, output_name in splits:
        summary = process_split(
            REPOSITORY_ROOT / "data" / source_name,
            REPOSITORY_ROOT / "data" / output_name,
        )
        print(f"{split_name}: {json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
