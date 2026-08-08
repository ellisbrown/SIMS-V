"""Tests for simulator object metadata compatibility."""

from sims.environment import sim_objects
from sims.environment.sim_objects import SimObject


def test_objaverse_annotation_fields_are_mapping_members(monkeypatch):
    annotations = {
        "asset-1": {
            "description": "a wooden chair",
            "most_specific_lemma": "chair",
            "synset": "chair.n.01",
        }
    }
    monkeypatch.setattr(
        sim_objects,
        "_get_objaverse_annotations",
        lambda: annotations,
    )
    obj = SimObject(
        {
            "assetId": "asset-1",
            "objectId": "Chair|1",
            "objectType": "Undefined",
        }
    )

    assert "objectId" in obj
    assert "synset" in obj
    assert "description" in obj
    assert obj["description"] == "a wooden chair"
    obj["cached"] = "value"
    assert "cached" in obj
    assert "asset-1" not in obj
    assert "missing" not in obj
