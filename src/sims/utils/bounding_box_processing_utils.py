import json
from typing import Dict

import numpy as np

from sims.utils.string_utils import convert_byte_to_string

BBOX_COORDINATE_DUMMY_VALUE = 1000


def read_bboxes_from_sensors(sensor_values: Dict, task_dict: Dict) -> np.ndarray:
    num_boxes = sensor_values["min_cols"].shape[1]

    oids = json.loads(convert_byte_to_string(sensor_values["oids_as_bytes"][0]))

    assert len(oids) == num_boxes, "Number of oids and boxes don't match"

    tgt_1_ids = []
    tgt_2_ids = []

    if "broad_synset_to_object_ids" in task_dict:
        tgt_1_ids = [val for val in task_dict["broad_synset_to_object_ids"].values()]
        tgt_1_ids = sum(tgt_1_ids, [])
    if "dest_receptacle_ids" in task_dict:
        tgt_2_ids = task_dict["dest_receptacle_ids"]

    # if ONLY_GET_CHOSEN_BBOX:
    #     tgt_1_ids = [task_dict["extras"]["chosen_object_id"]]
    #     tgt_2_ids = [task_dict["extras"]["chosen_dest_receptacle_id"]]

    def parse_biggest_bbox(object_indices):
        object_indices = sorted(object_indices)
        if (
            len(object_indices) == 0
        ):  # both bbox_1 and bbox_2 need to have a default value
            res = np.zeros((len(sensor_values["min_cols"]), 5))
            res[:, :4] = BBOX_COORDINATE_DUMMY_VALUE  # res[:, 4] = 0 by default
            return res
        x1 = sensor_values["min_cols"][:, object_indices].astype(int).astype(np.float32)
        y1 = sensor_values["min_rows"][:, object_indices].astype(int).astype(np.float32)
        x2 = sensor_values["max_cols"][:, object_indices].astype(int).astype(np.float32)
        y2 = sensor_values["max_rows"][:, object_indices].astype(int).astype(np.float32)
        if np.any(x1 > x2):
            x1, x2 = x2, x1
        if np.any(y1 > y2):
            y1, y2 = y2, y1
        area = (y2 - y1) * (x2 - x1)
        largest_area_oids = np.argmax(area, axis=1)
        time_ids = np.arange(len(x1))
        bboxes = np.stack(
            [
                x1[time_ids, largest_area_oids],
                y1[time_ids, largest_area_oids],
                x2[time_ids, largest_area_oids],
                y2[time_ids, largest_area_oids],
                area[time_ids, largest_area_oids],
            ],
            axis=1,
        )
        bboxes[bboxes == -1] = 1000
        return bboxes

    bbox_1 = parse_biggest_bbox([oids.index(oid) for oid in tgt_1_ids])
    bbox_2 = parse_biggest_bbox([oids.index(oid) for oid in tgt_2_ids])
    bboxes_combined = np.concatenate([bbox_1, bbox_2], axis=1)
    bbox_to_return = bboxes_combined

    # if NUMBER_OF_BBOXES == 1:  #
    #     bbox_to_return = bbox_1
    # elif NUMBER_OF_BBOXES == 2:
    #     bbox_to_return = bboxes_combined
    # else:
    #     raise NotImplementedError
    return bbox_to_return
