# Useful resource for the math bg of this file: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

import copy
import json
import math
import random

import numpy as np

from typing import (
    List,
    Literal,
    Optional,
    Dict,
    Union,
    Tuple,
    TYPE_CHECKING,
)

from sims.utils.bounding_box_processing_utils import BBOX_COORDINATE_DUMMY_VALUE
from sims.utils.data_generation_utils.exception_utils import TaskSamplerException

if TYPE_CHECKING:
    from sims.environment.stretch_controller import StretchController


from sims.environment.stretch_state import (
    convert_world_to_relative_coordinate_batched,
    wrap_angle_to_pm180,
)
from sims.utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
    INTEL_HEIGHT_CROPPED,
    STRETCH_CAMERA_HORIZONTAL_CROP,
)
from sims.utils.string_utils import convert_byte_to_string


def choose_a_random_camera(controller: "StretchController") -> Literal["nav", "manip"]:
    assert hasattr(controller, "navigation_camera") and hasattr(
        controller, "manipulation_camera"
    )
    return random.choice(["nav", "manip"])


class NoVisiblePointsOnFloorException(TaskSamplerException):
    def __str__(self):
        return "No Visible points were found"


MAXIMUM_DISTANCE_FOR_REACHABLE_POINT = 5


def covert_goal_str_to_action(goal_str: str) -> Union[dict, str]:
    try:
        goal = json.loads(goal_str)
    except Exception:
        goal = goal_str
    return goal


def dict_all_zero(d: dict) -> bool:
    return all(v == 0 for v in d.values())


def convert_goal_to_uniform_version(goal_dict_str: Union[str, dict]) -> dict:
    goal_type = None
    goal_desc = None
    if isinstance(goal_dict_str, dict):
        if dict_all_zero(goal_dict_str["base_position"]):
            goal_type = "arm"
            goal_desc = goal_dict_str["wrist_pose"]
        else:
            goal_type = "base"
            goal_desc = goal_dict_str["base_position"]
        assert not dict_all_zero(goal_desc)
    else:
        goal_type = "atomic"
        goal_desc = goal_dict_str
    return dict(goal_type=goal_type, goal_desc=goal_desc)


def get_camera_kernel(fov_y: dict) -> np.array:
    w, h = INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT

    focal_length = 0.5 * h / math.tan(math.radians(fov_y / 2))
    f_x = f_y = focal_length

    c_x = w / 2
    c_y = h / 2
    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    return K


def convert_3d_point_in_camera_frame_to_2d_in_camera_frame(
    goals_3d_in_camera_coord: np.array, fov_y: float
) -> np.array:
    K = get_camera_kernel(fov_y=fov_y)
    points_3d = goals_3d_in_camera_coord

    projected_point = np.matmul(K, points_3d.T).T
    last_elements = projected_point[:, 2]
    last_elements = last_elements[:, np.newaxis]
    projected_point = projected_point / last_elements
    point_image = projected_point[:, :2]

    return point_image


def get_floor_segmentation_mask(
    controller: "StretchController", which_camera: Literal["nav", "manip"]
) -> np.array:
    visible_objects = controller.get_visible_objects(
        which_camera=which_camera, maximum_distance=10
    )
    all_objects = controller.get_objects()
    all_visible_objects_infos = [
        obj for obj in all_objects if obj["objectId"] in visible_objects
    ]
    if which_camera == "nav":
        segmentation = controller.navigation_camera_segmentation
    elif which_camera == "manip":
        segmentation = controller.manipulation_camera_segmentation
    else:
        raise ValueError(f"Invalid camera: {which_camera}")

    visible_floors = [
        obj["objectId"]
        for obj in all_visible_objects_infos
        if obj["objectId"] in segmentation and obj["objectType"] == "Floor"
    ]
    if len(visible_floors) == 0:
        raise NoVisiblePointsOnFloorException()
    full_floor_segmentation_mask = None
    for floor_id in visible_floors:
        mask = segmentation[floor_id]
        if full_floor_segmentation_mask is None:
            full_floor_segmentation_mask = mask
        else:
            full_floor_segmentation_mask = full_floor_segmentation_mask + mask
    return full_floor_segmentation_mask


def convert_3d_point_in_agent_frame_to_2d_point_in_camera(
    xs: Union[np.array, list],
    ys: Union[np.array, list],
    zs: Union[np.array, list],
    camera_rel_position: dict,
    camera_rel_rotation: dict,
    fov_y: float,
) -> np.array:
    camera_rotation = copy.deepcopy(camera_rel_rotation)
    camera_in_agent_coord = copy.deepcopy(camera_rel_position)
    camera_in_agent_coord["rotation"] = camera_rotation

    world_obj = np.array([xs, ys, zs]).T
    goal_3d_in_camera_coord = convert_world_to_relative_coordinate_batched(
        world_obj, camera_in_agent_coord
    )

    pixels_2d = convert_3d_point_in_camera_frame_to_2d_in_camera_frame(
        goal_3d_in_camera_coord, fov_y=fov_y
    )

    pixels_2d[:, 1] = INTEL_HEIGHT_CROPPED - pixels_2d[:, 1]
    pixels_2d[:, 0] -= STRETCH_CAMERA_HORIZONTAL_CROP
    return pixels_2d


def read_bbox_sensors(sensor_dict: dict) -> np.array:
    num_boxes = sensor_dict["min_cols"].shape[1]

    oids = json.loads(convert_byte_to_string(sensor_dict["oids_as_bytes"][0]))

    assert len(oids) == num_boxes, "Number of oids and boxes don't match"

    def parse_biggest_bbox(object_indices):
        object_indices = sorted(object_indices)
        if (
            len(object_indices) == 0
        ):  # both bbox_1 and bbox_2 need to have a default value
            res = np.zeros((len(sensor_dict["min_cols"]), 5))
            res[:, :4] = 1000  # res[:, 4] = 0 by default
            return res
        x1 = sensor_dict["min_cols"][:, object_indices].astype(int).astype(np.float32)
        y1 = sensor_dict["min_rows"][:, object_indices].astype(int).astype(np.float32)
        x2 = sensor_dict["max_cols"][:, object_indices].astype(int).astype(np.float32)
        y2 = sensor_dict["max_rows"][:, object_indices].astype(int).astype(np.float32)
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

    bbox_1 = parse_biggest_bbox([oids.index(oid) for oid in oids])
    return bbox_1


def read_and_process_last_goal_agent_frame(sensors: dict) -> List[dict]:
    goal_agent_frame = sensors["last_goal_agent_frame"][1:]  # The first one is None
    goal_agent_frame = [convert_byte_to_string(row, None) for row in goal_agent_frame]
    goal_agent_frame = [covert_goal_str_to_action(row) for row in goal_agent_frame]
    uniform_goals = [convert_goal_to_uniform_version(row) for row in goal_agent_frame]
    return uniform_goals


def read_and_process_obj_bbox(sensors: dict) -> List[dict]:
    res = {}
    for k in ["manip_accurate_object_bbox", "nav_accurate_object_bbox"]:
        bboxes = read_bbox_sensors(sensors[k])  # The first one is None
        res[k] = bboxes
    list_res = []
    for i in range(len(res["manip_accurate_object_bbox"])):
        list_res.append(
            dict(
                manip=res["manip_accurate_object_bbox"][i],
                nav=res["nav_accurate_object_bbox"][i],
            )
        )
    return list_res


def read_and_process_all_global_goals(
    sensors: dict, important_only: bool = False
) -> List[dict]:
    all_global_goals = sensors["last_goal_absolute_frame"][1:]  # The first one is None
    all_global_goals = [convert_byte_to_string(row, None) for row in all_global_goals]
    all_global_goals = [covert_goal_str_to_action(row) for row in all_global_goals]

    all_global_base_goals = [
        (
            row["base_position"]
            if isinstance(row, dict) and "base_position" in row
            else {"x": 1000, "y": 1000, "z": 1000, "theta": 0}
        )
        for row in all_global_goals
    ]
    # all_global_base_goals = [row['base_position'] for row in all_global_goals if 'base_position' in row and type(row) == dict ]
    all_agent_poses = sensors["last_agent_location"][:-1]  # The first one is None
    all_agent_poses = [
        {"x": row[0], "y": row[1], "z": row[2], "theta": wrap_angle_to_pm180(row[4])}
        for row in all_agent_poses
    ]

    list_of_relative_goals_per_step = []
    for i in range(len(all_global_base_goals)):
        goal_base_to_look_at = all_global_base_goals[i:]
        filtered_base_only = [
            k
            for k in goal_base_to_look_at
            if k["x"] != 1000 and k["y"] != 1000 and k["z"] != 1000
        ]
        if len(filtered_base_only) > 0:
            relative_goals = convert_world_to_relative_coordinate_batched(
                filtered_base_only, all_agent_poses[i]
            )
            if important_only:
                relative_goals_norm = np.linalg.norm(relative_goals, axis=1)
                filtered_relative_goals = [
                    relative_goals[i]
                    for i in range(len(relative_goals))
                    if relative_goals_norm[i] > 2 and relative_goals_norm[i] < 5
                ]
                if len(filtered_relative_goals) == 0:
                    relative_goals = random.choice(relative_goals)
                else:
                    relative_goals = random.choice(filtered_relative_goals[:3])
                relative_goals = np.array([relative_goals])

            list_of_relative_goals_per_step.append(relative_goals)
        else:
            list_of_relative_goals_per_step.append([])

    return list_of_relative_goals_per_step


def convert_3d_world_coord_to_2d_camera_frame(
    valid_locations: np.array,
    agent_pose: dict,
    camera_rel_position: dict,
    camera_rel_rotation: dict,
    fov_y: float,
) -> np.array:
    agent_position = agent_pose["position"]
    agent_rotation = agent_pose["rotation"]

    valid_locations_agent_frame = convert_world_to_relative_coordinate_batched(
        valid_locations, {**agent_position, "rotation": agent_rotation}
    )

    valid_locations_in_pixel_space = (
        convert_3d_point_in_agent_frame_to_2d_point_in_camera(
            valid_locations_agent_frame[:, 0],
            valid_locations_agent_frame[:, 1],
            valid_locations_agent_frame[:, 2],
            camera_rel_position=camera_rel_position,
            camera_rel_rotation=camera_rel_rotation,
            fov_y=fov_y,
        )
    )

    return valid_locations_in_pixel_space


def generate_point_as_goal(
    task_dict: Dict, seq_len: int, repeat_goal: bool = False
) -> np.array:
    og_goal = generate_goal_as_point_only_first(task_dict, is_it_first_step=True)
    if repeat_goal:
        dummy_goal = og_goal
    else:
        dummy_goal = generate_goal_as_point_only_first(
            task_dict, is_it_first_step=False
        )
    dummy_goals = [dummy_goal] * (seq_len - 1)
    dummy_goals = np.array(dummy_goals)
    return np.concatenate([og_goal[np.newaxis], dummy_goals], axis=0)


def convert_pixels_to_int(pixels: np.array) -> np.array:
    return np.round(pixels).astype(np.int16)


DUMMY_POINT_VALUE = convert_pixels_to_int(
    np.array([BBOX_COORDINATE_DUMMY_VALUE, BBOX_COORDINATE_DUMMY_VALUE])
)


def normalize_point(point: np.array) -> np.array:
    width = INTEL_CAMERA_WIDTH
    height = INTEL_CAMERA_HEIGHT
    assert len(point) == 2 and point[0] < width and point[1] < height
    goal_in_camera_2d = np.array([point[0] / width, point[1] / height]).astype(np.float)
    return goal_in_camera_2d


def unnormalize_point(point: Optional[np.array]) -> Tuple[int, int]:
    return (int(point[0] * INTEL_CAMERA_WIDTH), int(point[1] * INTEL_CAMERA_HEIGHT))


def generate_goal_as_point_only_first(task_dict: Dict, is_it_first_step) -> np.array:
    if is_it_first_step:
        if "goal_in_camera_2d_first_step" in task_dict:
            goal_in_camera_2d_first_step = task_dict["goal_in_camera_2d_first_step"]
        elif "possible_points_on_target_in_first_frame" in task_dict:
            goal_in_camera_2d_first_step = random.choice(
                task_dict["possible_points_on_target_in_first_frame"]
            )
        else:
            return DUMMY_POINT_VALUE

        goal_in_camera_2d = normalize_point(goal_in_camera_2d_first_step)
        return goal_in_camera_2d
    else:
        return DUMMY_POINT_VALUE


def get_filter_for_visible_points_in_camera(
    all_agent_possible_locations_in_pixel_space: np.ndarray,
) -> np.ndarray:
    """
    all_agent_possible_locations_in_pixel_space: np.ndarray of shape (N, 2) where N is the number of possible locations
    """

    all_agent_possible_locations_in_pixel_space = np.round(
        all_agent_possible_locations_in_pixel_space
    )

    # Now that all positions are in pixel space, we need to only keep the ones that are visible in agent's view as their indices
    mask_for_in_frame_locations = (
        (
            all_agent_possible_locations_in_pixel_space
            == all_agent_possible_locations_in_pixel_space
        ).astype(float)
        + (all_agent_possible_locations_in_pixel_space > 0).astype(float)
        + (
            all_agent_possible_locations_in_pixel_space
            < np.array(
                [[INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT]]
                * len(all_agent_possible_locations_in_pixel_space)
            )
        ).astype(float)
    )

    mask_for_in_frame_locations = mask_for_in_frame_locations.sum(axis=1) == 6
    return mask_for_in_frame_locations


def get_most_centered_point_from_possible_points(
    possible_points: np.array, number_of_points: int
) -> np.array:
    center = possible_points.mean(axis=0)
    dist_to_center = np.linalg.norm(possible_points - center, axis=1)
    if number_of_points == 1:
        closest_ind = np.argmin(dist_to_center)
        return possible_points[closest_ind]
    elif number_of_points <= len(possible_points):
        return possible_points
    else:
        indices = np.argpartition(dist_to_center, number_of_points)[:number_of_points]
        return possible_points[indices]


def convert_point_from_3d_world_to_2d_camera(
    point_in_world: np.array,
    controller: "StretchController",
    which_camera: Literal["nav", "manip"],
) -> Tuple[np.array, np.array]:
    camera_rel_position, camera_rel_rotation, fov_y = (
        controller.get_controller_camera_params(which_camera=which_camera)
    )
    point_in_camera = convert_3d_world_coord_to_2d_camera_frame(
        point_in_world,
        controller.get_current_agent_full_pose(),
        camera_rel_position,
        camera_rel_rotation,
        fov_y,
    )
    is_this_valid = get_filter_for_visible_points_in_camera(point_in_camera)
    point_in_camera = convert_pixels_to_int(point_in_camera)

    return point_in_camera, is_this_valid
