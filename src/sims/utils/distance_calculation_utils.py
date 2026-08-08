import math
from typing import Literal, TYPE_CHECKING
import random

from shapely.geometry import Point
import numpy as np

from sims.utils.type_utils import Vector3


# avoid circular import
if TYPE_CHECKING:
    from sims.environment.stretch_controller import StretchController


def position_dist(
    p0: Vector3,
    p1: Vector3,
    ignore_y: bool = False,
    dist_fn: Literal["l1", "l2"] = "l2",
) -> float:
    """Distance between two points of the form {"x": x, "y": y, "z": z}."""
    if dist_fn == "l1":
        return (
            abs(p0["x"] - p1["x"])
            + (0 if ignore_y else abs(p0["y"] - p1["y"]))
            + abs(p0["z"] - p1["z"])
        )
    elif dist_fn == "l2":
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )
    else:
        raise NotImplementedError(
            f'dist_fn must be in {{"l1", "l2"}}. You gave {dist_fn}'
        )


def min_l2_distance_and_target_point(source, target_points):
    return min(
        [(position_dist(target, source), target) for target in target_points],
        key=lambda x: x[0],
    )


def sort_l2_distance_and_target_point(controller: "StretchController", target_points):
    source = controller.get_current_agent_position()

    sorted_dist = sorted(
        [(position_dist(target, source), target) for target in target_points],
        key=lambda x: x[0],
    )

    return sorted_dist


def closest_reachable_point_and_distance(
    controller: "StretchController", reachable_points
):
    if len(reachable_points) > 10:
        best_points_and_l2_distances = sort_l2_distance_and_target_point(
            controller, reachable_points
        )

        # Keep (if available) at least 10 points, or 10% of the initial points
        best_points_and_l2_distances = best_points_and_l2_distances[
            : max(10, int(round(len(best_points_and_l2_distances) / 10)))
        ]

        reachable_points = [point for _, point in best_points_and_l2_distances]

    all_paths = [
        (
            reachable_point,
            controller.get_shortest_path_to_point(
                reachable_point,
                specific_agent_meshes=[controller.agent_ids[-1]],
                attempt_path_improvement=False,
            ),
        )
        for reachable_point in reachable_points
    ]

    min_dist = float("inf")
    closest_reachable_point = None
    for reachable_point, path in all_paths:
        if path is None:
            continue
        dist = sum_dist_path(path)
        if dist < min_dist:
            min_dist = dist
            closest_reachable_point = reachable_point

    return closest_reachable_point, min_dist


def reachable_points_in_room(reachable_points, room_poly, min_points=50):
    to_shuffle = reachable_points[:]
    random.shuffle(to_shuffle)
    reachable_points = to_shuffle

    res = []
    for point in reachable_points:
        if room_poly.contains(Point(point["x"], point["z"])):
            res.append(point)
            if len(res) == min_points:
                break

    return res


def get_approximate_shortest_path_to_room(
    controller: "StretchController",
    room_id,
    room_polymap,
    points_in_room=None,
    reachable_points=None,
):
    if points_in_room is None:
        assert reachable_points is not None
        points_in_room = reachable_points_in_room(
            reachable_points, room_polymap[room_id]
        )

    return closest_reachable_point_and_distance(controller, points_in_room)


def closest_room_id_and_distance(
    controller: "StretchController", room_ids, room_polymap, reachable_points
):
    best_paths = [
        (
            room_id,
            get_approximate_shortest_path_to_room(
                controller, room_id, room_polymap, reachable_points=reachable_points
            ),
        )
        for room_id in room_ids
    ]

    min_dist = float("inf")
    closest_room_id = None
    for room_id, (point, dist_to_point) in best_paths:
        if point is None:
            continue
        if dist_to_point < min_dist:
            min_dist = dist_to_point
            closest_room_id = room_id

    return closest_room_id, min_dist


def reachable_points_per_room_id(room_ids, room_polymap, reachable_points):
    return {
        room_id: reachable_points_in_room(reachable_points, room_polymap[0][room_id])
        for room_id in room_ids
    }


def sum_dist_path(path):
    total_dist = 0
    for i in range(len(path) - 1):
        total_dist += dist(path[i], path[i + 1])
    return total_dist


def dist(loc_1, loc_2):
    return (
        (loc_1["x"] - loc_2["x"]) ** 2
        + (loc_1["y"] - loc_2["y"]) ** 2
        + (loc_1["z"] - loc_2["z"]) ** 2
    ) ** 0.5


def all_distances(source_array, target_array, ignore_y=False):
    """
    :param source_array: source positions array (n x 3)
    :param target_array: target positions array (m x 3)
    :param ignore_y: uses only coordinates 0 and 2 (x and z)
    :return: Euclidean distances array (n x m)
    """

    def norm2(data_array):
        return np.sum(data_array**2, axis=1, keepdims=True)

    if ignore_y:
        source_array = source_array[:, [0, 2]]
        target_array = target_array[:, [0, 2]]

    source_norm2 = norm2(source_array)
    target_norm2 = norm2(target_array)
    sims = source_array @ target_array.T
    return np.sqrt(np.maximum(source_norm2 - 2 * sims + target_norm2.T, 0))
