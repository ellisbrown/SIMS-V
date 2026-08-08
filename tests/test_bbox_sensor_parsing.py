import json

import numpy as np
import pytest

from sims.utils.bounding_box_processing_utils import read_bboxes_from_sensors
from sims.utils.pointing_utils import read_bbox_sensors
from sims.utils.string_utils import convert_string_to_byte


def _sensor_values(serialized_oids):
    encoded = convert_string_to_byte(serialized_oids, max_len=128)
    return {
        "oids_as_bytes": np.stack([encoded]),
        "min_cols": np.array([[0, 10]]),
        "min_rows": np.array([[0, 10]]),
        "max_cols": np.array([[5, 30]]),
        "max_rows": np.array([[5, 30]]),
    }


def test_bbox_readers_decode_json_object_ids():
    sensor_values = _sensor_values(json.dumps(["object-a", "object-b"]))

    assert read_bbox_sensors(sensor_values).shape == (1, 5)
    assert read_bboxes_from_sensors(
        sensor_values,
        {"dest_receptacle_ids": ["object-b"]},
    ).shape == (1, 10)


@pytest.mark.parametrize(
    "reader",
    [
        lambda values: read_bbox_sensors(values),
        lambda values: read_bboxes_from_sensors(values, {}),
    ],
)
def test_bbox_readers_reject_non_json_payloads(reader, tmp_path):
    marker = tmp_path / "eval-executed"
    payload = f"__import__('pathlib').Path({str(marker)!r}).touch()"

    with pytest.raises(json.JSONDecodeError):
        reader(_sensor_values(payload))

    assert not marker.exists()
