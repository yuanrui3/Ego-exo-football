from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backbone.release_baseline import ROOT, _verified_queries
from src.trajectory_dataset import file_digest


def test_release_config_matches_pinned_manifest_hash():
    manifest = json.loads(
        (ROOT / "configs" / "backbone_release_manifest.json").read_text(encoding="utf-8")
    )
    assert file_digest(ROOT / manifest["canonical_configuration"]) == manifest["canonical_configuration_sha256"]


def test_release_config_has_supported_mixed_action_queries():
    config = json.loads((ROOT / "configs" / "backbone_release_baseline.json").read_text(encoding="utf-8"))
    queries, _ = _verified_queries(config)
    assert len(queries) == 4
    assert {row["mapped_action_label"] for row in queries.values()} == {"DRIVE", "PASS", "HIGH_PASS"}
    assert all(row["conditioning"].startswith("event-conditioned") for row in queries.values())


def test_event_annotation_join_uses_both_stable_keys(monkeypatch):
    config = json.loads((ROOT / "configs" / "backbone_release_baseline.json").read_text(encoding="utf-8"))
    config["event_queries"][0]["annotation_index"] += 1
    with pytest.raises(ValueError, match="does not match current indexed target"):
        _verified_queries(config)


def test_external_query_metadata_does_not_invent_annotation_labels():
    config = {
        "source": {"adapter": "external_jsonl"},
        "event_queries": [{"frame_index": 25, "event_id": "external-event", "annotation_index": 3}],
    }
    queries, provenance = _verified_queries(config)
    assert queries[25]["annotation_status"] == "UNVERIFIED/NOT_SUPPLIED"
    assert "mapped_action_label" not in queries[25]
    assert "No annotation targets" in provenance["labels"]


def test_external_query_rejects_unverified_label():
    config = {
        "source": {"adapter": "external_jsonl"},
        "event_queries": [{
            "frame_index": 25, "event_id": "external-event",
            "annotation_index": 3, "mapped_action_label": "PASS",
        }],
    }
    with pytest.raises(ValueError, match="without a verified label join"):
        _verified_queries(config)
