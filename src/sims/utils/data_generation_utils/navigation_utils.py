import math
import platform
import random
from collections import defaultdict
from typing import Optional, List, Sequence, Dict, Callable, Union
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from shapely.geometry import Point, Polygon, GeometryCollection
from shapely.ops import triangulate
from skimage.morphology import skeletonize

if TYPE_CHECKING:
    from sims.tasks import AbstractSimsTask

from sims.utils.data_generation_utils.loc_grid_conversion import locs2grids, grids2locs
from sims.utils.distance_calculation_utils import (
    position_dist,
    sum_dist_path,
    all_distances,
)

from sims.utils.data_generation_utils.bbox_utils import (
    get_box_from_object,
    get_basis_for_3d_box,
)
from sims.utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from sims.environment.actions import StretchAction
from sims.environment.action_spaces import DONE_ACTION
from sims.environment.stretch_state import (
    StretchState,
    convert_relative_to_world_coordinate,
)

if TYPE_CHECKING:
    from sims.environment.stretch_controller import StretchController


SMALL_TYPES = [
    "Apple",
    "AlarmClock",
    "Mug",
    "Bowl",
    "BaseballBat",
    "SprayBottle",
    "Vase",
    "BasketBall",
]  # this is not ideal
ALIGNMENT_THRESHOLD = 10  # degrees allowed between agent heading and object center
PROP_VISIBLE_THRESHOLD = 0.8

TEMP_VERBOSE = False  # Temporary diagnostic logging gate.


def vector_distance(loc_start, loc_goal):
    cur_x = loc_start["x"]
    cur_z = loc_start["z"]
    goal_x = loc_goal["x"]
    goal_z = loc_goal["z"]
    vector = (goal_x - cur_x, goal_z - cur_z)
    return vector


def default_stop_callback(task: "AbstractSimsTask") -> bool:
    return False


def action_towards_goal_dict(
    task: "AbstractSimsTask", goal_dict: Dict
) -> Optional[StretchAction]:
    current_state = StretchState(controller=task.controller)
    goal_delta_state, goal_abs_state = StretchState.create_delta_from_goal(
        current_state=current_state, goal_dict=goal_dict
    )
    return task.get_action_from_goal(
        absolute_goal=goal_abs_state,
        agent_relative_goal=goal_delta_state,
        current_state=current_state,
    )


# Helper function for simple state transitions
def achieve_goal_dict_with_action_space(
    task: "AbstractSimsTask", goal_dict: Dict
) -> bool:
    # Note that this returns True if the goal has already been achieved and nothing happened.
    # Change that here if you want that to bo otherwise.
    action_success = True
    while action_success:
        action = action_towards_goal_dict(task, goal_dict)
        if action is None:
            break
        task.step_with_random(action, is_random=False)
        action_success = task.last_action_success
    return action_success


def action_space_can_act_on_path(task, path):
    # a bit simplistic - should check every corner
    if path is None or len(path) == 0:
        return False
    return (
        action_towards_goal_dict(task, goal_dict={"base_position": path[-1]})
        is not None
    )


def default_nav_replan_func(task, loc, return_path=True, from_loc=None):
    if return_path:
        return task.controller.get_shortest_path_to_point(loc)


def walk_on_path(
    path,
    task: "AbstractSimsTask",
    max_tries: int,
    stop_for: Union[set, set[str]] = None,
    successful_stop_callback: Callable[
        ["AbstractSimsTask"], bool
    ] = default_stop_callback,
    replan_func=default_nav_replan_func,
    return_success: bool = True,
):
    if stop_for is not None:
        stop_for = set(stop_for)

        def combined_successful_stop_callback(t):
            return (
                successful_stop_callback(t)
                or len(stop_for & set(getattr(t, "seen_object_ids", []))) > 0
            )
    else:
        combined_successful_stop_callback = successful_stop_callback

    total_failures_to_follow_path = 0

    while len(path or []) > 0:
        if total_failures_to_follow_path > max_tries:
            # Look for another target, if any, or call back with the same target...
            break

        if combined_successful_stop_callback(task):
            return True if return_success else []

        action = action_towards_goal_dict(task, goal_dict={"base_position": path[0]})
        if action is None:
            path = path[1:]
            continue

        task.step_with_random(action, is_random=False)

        move_success = task.last_action_success
        if not move_success:
            # print(f"Stuck with {action.long_name}!")
            total_failures_to_follow_path += 1

            planner = replan_func(task, None, False, None)

            def base_step_direction_random_quantity(direction: str):
                if direction == "back":
                    action_quantity = (
                        task.agent_action_space.sample_random_action_quantity(
                            action_string="base_z"
                        )
                    )
                    base_goal_in_agent_frame = {"x": 0, "y": 0, "z": -action_quantity}
                    base_goal_in_world_frame = convert_relative_to_world_coordinate(
                        agent_relative_point=base_goal_in_agent_frame,
                        agent_state=StretchState(task.controller),
                    )
                    goal_dict = {"base_position": base_goal_in_world_frame}

                elif direction in ["left", "right"]:
                    action_quantity = (
                        task.agent_action_space.sample_random_action_quantity(
                            action_string="base_theta"
                        )
                    )
                    # negative if it's left
                    action_quantity = (
                        -action_quantity if direction == "left" else action_quantity
                    )
                    world_theta = (
                        StretchState(task.controller).base_position["theta"]
                        + action_quantity
                    )
                    goal_dict = {"base_position": {"theta": world_theta}}
                else:
                    raise ValueError(f"Invalid direction: {direction}")

                recovery_action = action_towards_goal_dict(task, goal_dict)
                if recovery_action is None:
                    raise ValueError(
                        f"No recovery action with {direction} and sampled action quantity {action_quantity}"
                    )
                return task.step_with_random(recovery_action, is_random=True)

            def undo_failed_action(failed_action: StretchAction):
                # Because this is a first-line recovery strategy, we would never expect a move back
                # to be the failed action
                failed_action_dict = failed_action.to_dict()
                if "z" in failed_action_dict["base"]:
                    undo_success = base_step_direction_random_quantity("back")
                elif "theta" in failed_action_dict["base"]:
                    # check for sign of theta. If negative, failed action was to the left, so turn right
                    # (and vice versa)
                    rotate_success = base_step_direction_random_quantity(
                        "right" if failed_action_dict["base"]["theta"] < 0 else "left"
                    )
                    # also step back after a corrected rotate
                    back_success = base_step_direction_random_quantity("back")
                    undo_success = rotate_success and back_success
                else:
                    raise ValueError(
                        f"I don't know how to recover from the failed action {failed_action}"
                    )

                return undo_success

            lock_state = StretchState(task.controller)

            # First strategy: undo the failed action
            undo_failed_action(action)
            undo_success = task.last_action_success
            move_success = task.controller.sufficient_agent_state_change(
                lock_state, StretchState(task.controller)
            )

            if platform.system() == "Darwin":
                if TEMP_VERBOSE:
                    print("failed action was", action)
                if move_success and not undo_success:
                    if TEMP_VERBOSE:
                        print(
                            "Failed to complete the undo actions but still moved enough to break out"
                        )
                elif not move_success:
                    if TEMP_VERBOSE:
                        print(
                            "Failed to break out of the stuck state with undoing actions. (shouldn't this be rare?) Trying random actions..."
                        )
                else:
                    if TEMP_VERBOSE:
                        print(
                            "Successfully broke out of the stuck state with non-contacting undoing actions"
                        )
            if not move_success:
                # second strategy: try random actions, checking for unsticking every time
                for _ in range(5):
                    # select randomly between back (50%), left(25%), and right(25%)
                    random_direction = random.choices(
                        ["back", "left", "right"], weights=[0.5, 0.25, 0.25]
                    )[0]
                    base_step_direction_random_quantity(random_direction)
                    move_success = task.controller.sufficient_agent_state_change(
                        lock_state, StretchState(task.controller)
                    )
                    if move_success:
                        break

            # TODO: return false if we can't recover from the stuck state?

            # and now replan
            if planner is not None:
                planner.invalidate("all")

            new_path = replan_func(task, path[-1], True, None)

            if new_path is not None:
                path = new_path

    return (path is not None and len(path) == 0) if return_success else path


def rotate_until_visible(
    task: "AbstractSimsTask",
    object_id: str,
    maximum_distance: float = 3,
    use_task_success: bool = False,
    use_object_nav_strict_success: bool = False,
    say_done: bool = False,
    step_callback: Callable[["AbstractSimsTask"], bool] = default_stop_callback,
    use_wall_alignment: bool = False,
    manipulation_camera: bool = False,
):
    assert not (use_task_success and manipulation_camera)
    assert not (use_task_success and use_object_nav_strict_success), (
        "Cannot use both task success and object nav strict success"
    )
    last_alignment: Optional[float] = None

    for i in range(20):  # arbitrary number of rotations. TODO revisit
        step_callback(task)

        if use_task_success:
            cond = task.successful_if_done(
                strict_success=True
            )  # planner should always try for the best
        elif use_object_nav_strict_success:
            cond = is_any_object_sufficiently_visible_and_in_center_frame(
                controller=task.controller,
                object_ids=[object_id],
                object_synset=task.controller.get_object(object_id)["synset"],
                manipulation_camera=manipulation_camera,
            )
        else:
            cond = task.controller.object_is_visible_in_camera(
                object_id,
                which_camera="manip" if manipulation_camera else "nav",
                maximum_distance=maximum_distance,
            )
        if cond:
            if say_done:
                task.step_with_action_str(DONE_ACTION)
            return True

        current_state = StretchState(task.controller)
        # TODO proposal: mod these functions to use StretchState instead of their current situation
        vector = (
            task.controller.get_agent_alignment_to_object(
                object_id, use_arm_orientation=manipulation_camera
            )
            if not use_wall_alignment
            else task.controller.get_agent_alignment_to_wall(
                object_id, use_arm_orientation=manipulation_camera
            )
        )
        if last_alignment is not None:
            # Check if the alignment has changed (otherwise,
            # we're stuck and there's no need to keep rotating)
            if np.isclose(last_alignment, vector):
                return False

        # Save last alignment to compare with the next iteration
        last_alignment = vector

        ## Here is the different part
        goal_delta_state, goal_abs_state = StretchState.create_delta_from_goal(
            current_state=current_state,
            goal_dict={
                "base_position": {
                    "theta": vector + current_state.base_position["theta"]
                }
            },
        )

        action = task.get_action_from_goal(
            absolute_goal=goal_abs_state,
            agent_relative_goal=goal_delta_state,
            current_state=current_state,
        )
        if action is None:
            return False  # Failure. This will also capture the case where the object is too close to the agent
        task.step_with_random(action, is_random=False)

        if not task.last_action_success:
            break
    return False


def get_room_id_from_location(room_polymap, position, verbose=True):
    if isinstance(position, dict) and "x" in position and "z" in position:
        point = Point(position["x"], position["z"])
    else:
        assert len(position) == 3
        point = Point(position[0], position[2])

    room_id_to_dist = {}
    for room_id, poly in room_polymap.items():
        room_id_to_dist[room_id] = poly.distance(point)
        if room_id_to_dist[room_id] == 0:
            return room_id

    on_walls_of_room_ids = [
        room_id for room_id, dist in room_id_to_dist.items() if dist < 1e-3
    ]
    if len(on_walls_of_room_ids) == 0:
        if verbose:
            if TEMP_VERBOSE:
                print(position, "is out of house in get_room_id_from_location")
        return None
    elif len(on_walls_of_room_ids) == 1:
        return on_walls_of_room_ids[0]
    else:
        if verbose:
            if TEMP_VERBOSE:
                print(
                    position,
                    "is on walls of multiple rooms, cannot return unique room id",
                )
        return None


def panoramic_scan(
    task,
    step_callback=default_stop_callback,
):
    AGENT_ROTATION_DEG = task.agent_action_space.base_rot_constant
    for i in range(int(360 / AGENT_ROTATION_DEG)):
        step_callback(task)
        rotation = "rotate_right"
        task.step_with_action_str(action_name=rotation)
    return True


def rotate_until_visible_and_say_done(
    task, object_id, step_callback=default_stop_callback
):
    return rotate_until_visible(
        task,
        object_id,
        use_task_success=True,
        say_done=True,
        step_callback=step_callback,
    )


# TODO: The adjacency logic should maybe go somewhere else, but it's here for now
def get_rooms_polymap_and_type(house):
    room_poly_map = {}
    room_type_dict = {}
    # NOTE: Map the rooms
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )
        room_type_dict[room["id"]] = room["roomType"]
    return room_poly_map, room_type_dict


def get_all_walls(house):
    all_walls = [
        ("room|" + w["id"].split("|")[1], ("|").join(w["id"].split("|")[2:]))
        for w in house["walls"]
        if ("empty" not in w or w["empty"] is False)
    ]
    wall_dict = {}
    for w in all_walls:
        wall_dict.setdefault(w[0], set())
        wall_dict[w[0]].add(w[1])

    walls = {}
    for r1 in wall_dict:
        for r2 in wall_dict:
            if r1 != r2:
                intersection = wall_dict[r1] & wall_dict[r2]
                if len(intersection) > 0:
                    walls[(r1, r2)] = intersection
    return walls


def build_layout_graph_matrix(house, room_polymap):
    rooms = [r["id"] for r in house["rooms"]]
    room_graph = torch.zeros((len(rooms), len(rooms)))
    # Exterior doors have no second room and do not connect rooms in the layout graph.
    doors = [
        sorted([d["room0"], d["room1"]])
        for d in house["doors"]
        if d["room1"] is not None
    ]
    walls = get_all_walls(house)
    for r1_index, r1 in enumerate(rooms):
        for r2_index, r2 in enumerate(rooms):
            connected = 0
            poly_r1 = room_polymap[r1]
            poly_r2 = room_polymap[r2]
            if r1 == r2:
                connected = 1
            elif (
                poly_r1.touches(poly_r2)
                and ((r1, r2) not in walls)
                and (not isinstance(poly_r1.intersection(poly_r2), Point))
            ):
                # Last condition necessary to avoid trivial intersections.
                connected = 1
            elif sorted([r1, r2]) in doors:
                connected = 0.5
            room_graph[r1_index, r2_index] = room_graph[r2_index, r1_index] = connected
    return room_graph, rooms


def get_nearest_positions(task, world_position):
    """Get the n reachable positions that are closest to the world_position."""
    task.reachable_positions.sort(
        key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
    )
    return task.reachable_positions[
        : min(
            len(task.reachable_positions),
            6,  # take the top six candidates
        )
    ]


def filtered_starting_positions(
    locations, margin=1, grid_spacing=0.25
) -> List[Dict[str, float]]:
    im, locs = locs2grids(locations, grid_spacing)

    num_pos = 0
    im2 = None
    while num_pos == 0 and margin > 0:
        im2 = binary_erosion(im, iterations=margin)
        num_pos = np.sum(im2)
        if num_pos == 0:
            margin -= 1

    if num_pos == 0:
        # trying with all locations
        return locations

    return grids2locs(im2, locs)


def thinned_starting_positions(locations, grid_spacing=0.25):
    try:
        im, locs = locs2grids(locations, grid_spacing)

        im2 = skeletonize(im)
        num_pos = np.sum(im2)

        if num_pos == 0:
            # trying with all locations
            return locations

        return grids2locs(im2, locs)
    except Exception:
        import traceback

        print(traceback.format_exc())
        return locations


def room_distances(controller):
    room_polymap = controller.room_poly_map
    y = controller.controller.last_event.metadata["agent"]["position"]["y"]
    navigable_positions = controller.get_reachable_positions()

    in_room = {}
    cent = {}
    res = {}
    for ra in room_polymap:
        cra = dict(x=room_polymap[ra].centroid.x, y=y, z=room_polymap[ra].centroid.y)
        cent[ra] = min(
            [
                (
                    position_dist(cra, pos, ignore_y=True),
                    pos,
                )
                for pos in navigable_positions
            ],
            key=lambda x: x[0],
        )[1]
        in_room[ra] = math.sqrt(2) / 2 * math.sqrt(room_polymap[ra].area)
        res[ra] = {ra: 2 * in_room[ra]}

    room_order = list(room_polymap.keys())

    for ita, ra in enumerate(room_order[:-1]):
        for rb in room_order[ita + 1 :]:
            try:
                res[ra][rb] = (
                    in_room[ra]
                    + in_room[rb]
                    + sum_dist_path(
                        controller.get_shortest_path_to_point(
                            cent[rb], cent[ra], attempt_path_improvement=False
                        )
                    )
                )
            except Exception:
                res[ra][rb] = float("inf")
            res[rb][ra] = res[ra][rb]

    return res


def visit_ids_from_info(
    task,
    nav_step_callback=default_stop_callback,
    rot_step_callback=default_stop_callback,
    max_distance=3,
    maybe_dp=None,
    replan_kwargs=None,
):
    replan_kwargs = replan_kwargs or {}
    use_astar = maybe_dp is not None

    all_targets = set(task.task_info["visit_ids"].keys())
    while len(all_targets):
        current_target_id = task.controller.get_closest_object_from_ids(
            list(all_targets)
        )
        if current_target_id is None:
            current_target_id = random.choice(list(all_targets))

        all_targets.remove(current_target_id)

        left_to_visit = set(task.task_info["visit_ids"][current_target_id])
        failures = set()
        num_attempts = 0
        max_attempts = 2 * len(left_to_visit)
        success = True
        while len(left_to_visit) + len(failures) > 0:
            closest_object_id = None
            if len(failures) > 0 and (len(left_to_visit) == 0 or random.random() > 0.5):
                closest_object_id = task.controller.get_closest_object_from_ids(
                    list(failures)
                )

            if closest_object_id is None:
                closest_object_id = task.controller.get_closest_object_from_ids(
                    list(left_to_visit)
                )

            if closest_object_id is None:
                if len(failures) > 0 and (
                    len(left_to_visit) == 0 or random.random() > 0.5
                ):
                    closest_object_id = random.choice(list(failures))
                elif len(left_to_visit) > 0:
                    closest_object_id = random.choice(list(left_to_visit))

            if use_astar:
                _, path, _ = maybe_dp.nearest_object_id_path_dist([closest_object_id])
            else:
                path = task.controller.get_shortest_path_to_object(
                    object_id=closest_object_id
                )
            if path is not None:
                walk_on_path(
                    path,
                    task,
                    max_tries=20,
                    successful_stop_callback=nav_step_callback,
                    **replan_kwargs,
                )

            vis_succ = rotate_until_visible(
                task=task,
                object_id=closest_object_id,
                use_task_success=False,
                say_done=False,
                maximum_distance=max_distance,
                step_callback=rot_step_callback,
            )

            if not vis_succ:
                if closest_object_id in left_to_visit:
                    left_to_visit.remove(closest_object_id)
                failures.add(closest_object_id)
            elif closest_object_id in left_to_visit:
                left_to_visit.remove(closest_object_id)
            elif closest_object_id in failures:
                failures.remove(closest_object_id)

            num_attempts += 1
            if num_attempts == max_attempts and len(left_to_visit) + len(failures) > 0:
                success = False
                break
        if success:
            return current_target_id

    return None


def get_wall_center_floor_level(wall_id, y):
    wall_xzs = wall_id.split("|")[2:]
    assert len(wall_xzs) == 4

    return dict(
        x=(float(wall_xzs[0]) + float(wall_xzs[2])) / 2,
        y=y,
        z=(float(wall_xzs[1]) + float(wall_xzs[3])) / 2,
    )


def get_nearest_to_wall_center_in_room(
    task, wall_id, room_id, thinned_starting_positions
):
    """Get the reachable position that is closest to the wall center in the current room."""
    world_position = get_wall_center_floor_level(
        wall_id, y=task.controller.get_current_agent_position()["y"]
    )

    positions = thinned_starting_positions

    positions.sort(
        key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
    )

    nearest_reachable = positions[
        : min(
            len(task.reachable_positions),
            50,  # using large number since center of wall might be blocked by e.g. a large bed
        )
    ]

    for attempt, rp in enumerate(nearest_reachable):
        if get_room_id_from_location(task.room_poly_map, rp) == room_id:
            return rp

    return None


def get_object_room_id(obj, room_poly_map, verbose=False):
    room_id = get_room_id_from_location(
        room_poly_map, obj["axisAlignedBoundingBox"]["center"], verbose=False
    )
    if room_id is not None:
        return room_id

    room_ids = defaultdict(int)
    for vertex in get_box_from_object(obj):
        room_ids[
            get_room_id_from_location(
                room_poly_map, {a: x for a, x in zip("xyz", vertex)}, verbose=False
            )
        ] += 1
    room_ids.pop(None, None)

    if len(room_ids) > 1 and verbose:
        print(obj["synset"], dict(room_ids))

    if len(room_ids) == 0:
        if verbose:
            print(obj, "is out of house in get_object_room_id")
        return None

    return max(
        [(count, room_id) for room_id, count in room_ids.items()], key=lambda x: x[0]
    )[1]


def object_ids_in_closed_receptacles(
    controller: "StretchController", closed_openness=0.1
):
    # Get all objects that can't be seen because they're inside closed receptacles
    object_list = controller.get_objects()
    return object_ids_in_closed_receptacles_from_objs(object_list, closed_openness)


def object_ids_in_closed_receptacles_from_objs(object_list, closed_openness=0.1):
    # Get all objects that can't be seen because they're inside closed receptacles
    closed_receptacles = [
        o
        for o in object_list
        if o["receptacle"] and o["openable"] and o["openness"] < closed_openness
    ]

    return {oid for o in closed_receptacles for oid in o["receptacleObjectIds"]}


def front_facing_positions(
    controller: "StretchController",
    object_id,
    reachable_positions,
    max_distance=STRETCH_ENV_ARGS["visibilityDistance"],
    seed_margin_degrees=30,
):
    obj = controller.get_object(object_id)
    box = get_box_from_object(obj)
    center = box.mean(axis=0)
    shift = np.array(
        [0, 0, max_distance]
    )  # Note that the object's surface can be considerably closer
    centrads = np.deg2rad(obj["rotation"]["y"])
    seed_margin_rads_half = np.deg2rad(seed_margin_degrees) / 2

    rads_to_sample = np.linspace(
        start=centrads % (2 * np.pi) - seed_margin_rads_half,
        stop=centrads % (2 * np.pi) + seed_margin_rads_half,
        num=9,
    ) % (2 * np.pi)

    seed_positions = []
    for rads in rads_to_sample:
        cosr = np.cos(rads)
        sinr = np.sin(rads)
        rmat = np.array(
            [
                [cosr, 0, sinr],
                [0, 1, 0],
                [-sinr, 0, cosr],
            ]
        )
        seed_positions.append(center + rmat @ shift)
    seed_positions = np.stack(seed_positions, axis=0)

    # Search for the nearest distance to any of the sample positions
    all_dists = np.min(
        all_distances(reachable_positions, seed_positions, ignore_y=True), axis=1
    )
    cand_pos = reachable_positions[np.nonzero(all_dists < 1.0)]

    angles = np.array(
        [
            np.rad2deg(np.arctan2(center[0] - candp[0], center[2] - candp[2])) % 360
            for candp in cand_pos
        ]
    )

    return cand_pos, angles


def get_room_to_walls(house, min_wall_width: Optional[float] = None):
    res = defaultdict(set)

    from shapely import LineString, MultiLineString, difference as shapely_difference

    def get_wall_to_doors(house):
        res = defaultdict(list)

        for door in house["doors"]:
            if door["room0"] == door["room1"]:
                # We want to use this structure to skip walls dominated by a door,
                # but only if the wall leads to another room (that will be visited anyway)
                continue
            res[door["wall0"]].append(door)
            res[door["wall1"]].append(door)

        return {**res}

    wall_to_doors = get_wall_to_doors(house)

    def get_wall_width_start(wall_id):
        wall_xzs = wall_id.split("|")[2:]
        assert len(wall_xzs) == 4
        difx = abs(float(wall_xzs[2]) - float(wall_xzs[0]))
        difz = abs(float(wall_xzs[3]) - float(wall_xzs[1]))

        if difx >= difz:
            return LineString([(0, 0), (difx, 0)]), min(
                float(wall_xzs[2]), float(wall_xzs[0])
            )
        else:
            return LineString([(0, 0), (difz, 0)]), min(
                float(wall_xzs[3]), float(wall_xzs[1])
            )

    for wall in house["walls"]:
        if "exterior" in wall["id"] or wall["roomId"] == "exterior":
            continue

        if "empty" in wall and wall["empty"]:
            continue

        # if min_wall_width is not None:
        #     points = np.array([[p["x"], p["z"]] for p in wall["polygon"]])
        #     if euclidean_distances(points).max() < min_wall_width:
        #         continue

        if min_wall_width is not None and wall["id"] in wall_to_doors:
            wall_line, bias = get_wall_width_start(wall["id"])
            door_segments = MultiLineString(
                [
                    [(door["holePolygon"][0]["x"], 0), (door["holePolygon"][1]["x"], 0)]
                    for door in wall_to_doors[wall["id"]]
                ]
            )
            free_wall = shapely_difference(wall_line, door_segments)

            if free_wall.length < min_wall_width:
                # print(wall_line.length, door_segments.length, free_wall.length)
                continue

        res[wall["roomId"]].add(wall["id"])

    return {**res}


def is_any_object_sufficiently_visible_and_in_center_frame(
    controller: "StretchController",
    object_ids: List[str],
    object_synset: Optional[str],
    scale: float = 1.5e4,
    manipulation_camera=False,
    absolute_min_pixels: int = 200,
):
    # Note: this code assumes INTEL_VERTICAL_FOV = 59 is used
    # Note: small object of e.g. 0.15 * 0.15 -> 0.0225 m2 (scale ~10,000 for 200 pixels)
    # Also, constraining the maximum threshold to be 1000 (the old one for non-SMALL_TYPES),
    # and the minimum to be 200 (the old one for SMALL_TYPES).
    # Note: Using default scale=15,000 instead of 10,000 as `ProportionOfObjectVisible` resolves it.
    # Note: default image height = 224 -> 224 ^ 2 = 50176 pixels is the area for a 1:1 image shape ratio.
    scale_to_apply = scale * (controller.navigation_camera.shape[0] ** 2) / 50176
    pixel_mass_thresholds = {}
    for oid in object_ids:
        if manipulation_camera:
            pixel_mass_thresholds[oid] = 200
        else:
            # # d1, d2, d3 are the dimensions of the bbox, in meters
            try:
                _, mags = get_basis_for_3d_box(controller.get_object(oid))
            except Exception:
                # This is a hack to deal with the fact that the basis `get_basis_for_3d_box` may
                # fail for some objects. In that case, we just use the default value of 200.
                pixel_mass_thresholds[oid] = 200
                continue
            d1, d2, d3 = tuple(mags)
            largest_bbox_face_area = max(d1 * d2, d2 * d3, d3 * d1)
            pixel_mass_thresholds[oid] = max(
                min(scale_to_apply * largest_bbox_face_area, 1000), absolute_min_pixels
            )

    oid_to_seg = None
    target_visibility_quant = {}
    for oid in object_ids:
        alignment = abs(
            controller.get_agent_alignment_to_object(
                oid, use_arm_orientation=manipulation_camera
            )
        )
        if alignment <= ALIGNMENT_THRESHOLD:
            if oid_to_seg is None:
                # Have to do this here to avoid loading the segmentation masks
                # when we don't need to
                oid_to_seg = (
                    controller.manipulation_camera_segmentation
                    if manipulation_camera
                    else controller.navigation_camera_segmentation
                )
            try:
                pixel_mass = oid_to_seg[oid].sum()
                empty_top = (
                    None  # MANIP CAMERA ONLY: Make sure there are at least 10% empty pixels on top of the image
                    if not manipulation_camera
                    else (
                        oid_to_seg[oid][
                            : int(0.1 * controller.navigation_camera.shape[0])
                        ]
                        == 0
                    ).all()
                )
            except KeyError:
                pixel_mass = 0
                empty_top = False

            target_visibility_quant[oid] = {
                "alignment": alignment,
                "pixel_mass": pixel_mass,
                "empty_top": empty_top,
            }

    sufficient_visibility = False
    for obj_id, obj_data in target_visibility_quant.items():
        if obj_data["alignment"] >= ALIGNMENT_THRESHOLD:
            continue

        if obj_data["pixel_mass"] < absolute_min_pixels:
            continue

        if obj_data["pixel_mass"] <= pixel_mass_thresholds[obj_id]:
            # The pixel mass threshold can be overly strict so we include here an approximate check
            # to see if the object is > prop_visible_threshold visible and, if it is, we say its ok.
            prop_visible = controller.step(
                action="ProportionOfObjectVisible",
                objectId=list(target_visibility_quant.keys())[0],
            ).metadata["actionReturn"]
            if prop_visible < PROP_VISIBLE_THRESHOLD:
                continue

        if obj_data["empty_top"] is not None and (not obj_data["empty_top"]):
            continue
        sufficient_visibility = True
        break

    return sufficient_visibility


def reachable_locs_in_room(
    controller: "StretchController",
    room_id: str,
):
    return [
        pos
        for pos in controller.get_reachable_positions()
        if get_room_id_from_location(controller.room_poly_map, pos) == room_id
    ]


class WallSearch:
    def __init__(
        self,
        task: "AbstractSimsTask",
        max_wall_distance: float = 15.0,
        wall_alignment_threshold: float = 60.0,
        use_all_cameras: bool = False,
        grid_size: float = 0.25,
    ):
        self.task = task
        self.max_wall_distance = max_wall_distance
        self.wall_alignment_threshold = (
            wall_alignment_threshold  # degrees (>180 == disabled)
        )
        self.use_all_cameras = use_all_cameras

        self.house = task.controller.current_scene_json
        self.room2walls = get_room_to_walls(self.house, min_wall_width=1)
        self.seen_walls = set()
        self.thinned_starting_positions = thinned_starting_positions(
            task.controller.get_reachable_positions(grid_size)
        )
        self.targeting_room: Optional[str] = None
        self.targeting_wall: Optional[str] = None
        self.dot_prod_threshold = np.cos(
            np.deg2rad(180 - self.wall_alignment_threshold)
        )

        self._room_triangles = {}

    def room_triangles(self, room_id):
        if room_id not in self._room_triangles:
            self._room_triangles[room_id] = triangulate_room_polygon(
                self.task.room_poly_map[room_id]
            )
        return self._room_triangles[room_id]

    def end_nav_on_seen_wall(self, task: "AbstractSimsTask") -> bool:
        """Callback for navigation utils with success (target wall seen) return"""
        target_seen = False
        if self.targeting_room is not None:
            agent_fwd = agent_forward_vector(task.controller)
            for wall_id in self.room2walls[self.targeting_room] - self.seen_walls:
                wall_norm = wall_normal(
                    wall_id, self.targeting_room, task.room_poly_map
                )
                dot_prod = wall_norm @ agent_fwd
                if dot_prod > self.dot_prod_threshold:
                    continue

                if task.controller.object_is_visible_in_camera(
                    wall_id,
                    which_camera="both" if self.use_all_cameras else "nav",
                    maximum_distance=self.max_wall_distance,
                ):
                    self.seen_walls.add(wall_id)
                    if wall_id == self.targeting_wall:
                        target_seen = True

        return target_seen

    def seen_room_walls_in_step(self, task: "AbstractSimsTask") -> bool:
        """Callback for navigation utils without success (new wall seen) return (always False)"""
        self.end_nav_on_seen_wall(task)
        return False  # default output for callback

    def end_nav_when_entering_room(self, task: "AbstractSimsTask") -> bool:
        """Callback for navigation utils to reach room"""
        self.end_nav_on_seen_wall(task)
        return get_room_id_from_location(
            {self.targeting_room: task.room_poly_map[self.targeting_room]},
            task.controller.get_current_agent_position(),
            verbose=False,
        )

    @property
    def missing_walls(self):
        return self.room2walls[self.targeting_room] - self.seen_walls

    def make_wall_visit_order(self, task):
        alignment_ids = [
            (task.controller.get_agent_alignment_to_wall(wall_id), wall_id)
            for wall_id in self.missing_walls
        ]
        alignment_ids.sort(key=lambda x: x[0] + 180 if x[0] < 0 else x[0])
        return [al_id[1] for al_id in alignment_ids]

    def search_walls(self, task, path_planner=None):
        if len(self.missing_walls) == 0:
            return

        if path_planner.dp is None:
            path_to_room = task.controller.get_shortest_path_to_room(
                room_id=self.targeting_room,
                room_triangles=self.room_triangles(self.targeting_room),
            )
        else:
            path_to_room = path_planner.dp.path_to_room(
                room_id=self.targeting_room, room_triangles=None
            )
        if path_to_room is None:
            return

        walk_on_path(
            path=path_to_room,
            task=task,
            max_tries=20,
            successful_stop_callback=self.end_nav_when_entering_room,
            **path_planner.make_walk_kwargs(),
        )

        # regardless of success: select now wall visitation
        wall_visitation_order = self.make_wall_visit_order(task)

        for wall_id in wall_visitation_order:
            self.targeting_wall = wall_id

            if wall_id not in self.missing_walls:
                continue

            # Note: we avoid aligning to the target wall unless we're in the room (even if the wall could be seen)
            if (
                get_room_id_from_location(
                    task.room_poly_map, task.controller.get_current_agent_position()
                )
                == self.targeting_room
            ):
                rotate_until_visible(
                    task=task,
                    object_id=wall_id,
                    maximum_distance=self.max_wall_distance,
                    step_callback=self.seen_room_walls_in_step,
                    use_wall_alignment=True,
                )

                if wall_id not in self.missing_walls:
                    continue

            loc_near_wall_center = get_nearest_to_wall_center_in_room(
                task=task,
                wall_id=wall_id,
                room_id=self.targeting_room,
                thinned_starting_positions=self.thinned_starting_positions,
            )

            if loc_near_wall_center is not None:
                path = None
                if path_planner.dp is not None:
                    path = path_planner.dp.path_to_loc(loc_near_wall_center)

                if path is None:
                    path = task.controller.get_shortest_path_to_point(
                        target_position=loc_near_wall_center
                    )

                if path is not None:
                    walk_success = walk_on_path(
                        path=path,
                        task=task,
                        max_tries=5,
                        successful_stop_callback=self.end_nav_on_seen_wall,
                        **path_planner.make_walk_kwargs(),
                    )

                    if wall_id not in self.missing_walls:
                        continue

                    if not walk_success:
                        if TEMP_VERBOSE:
                            print("Failed to walk")

                    if walk_success or (
                        get_room_id_from_location(
                            task.room_poly_map,
                            task.controller.get_current_agent_position(),
                        )
                        == self.targeting_room
                    ):
                        rotate_until_visible(
                            task=task,
                            object_id=wall_id,
                            maximum_distance=self.max_wall_distance,
                            step_callback=self.end_nav_on_seen_wall,
                            use_wall_alignment=True,
                        )

        self.targeting_wall = None


def triangulate_room_polygon(room_polygon: Polygon) -> GeometryCollection:
    return triangulate(room_polygon)


def wall_normal(wall_id, room_id, room_polymap, shift=0.1):
    wall_xzs = wall_id.split("|")[2:]
    assert len(wall_xzs) == 4

    along = np.array(
        [
            float(wall_xzs[2]) - float(wall_xzs[0]),
            0,
            float(wall_xzs[3]) - float(wall_xzs[1]),
        ]
    )

    ortho = np.array([along[2], 0, -along[0]])
    ortho = ortho / np.linalg.norm(ortho)

    center = get_wall_center_floor_level(wall_id, y=0)
    center = np.array([center[x] for x in "xyz"])

    front = center + ortho * shift

    if (
        get_room_id_from_location(
            {room_id: room_polymap[room_id]}, front, verbose=False
        )
        == room_id
    ):
        return ortho

    front = center - ortho * shift
    if (
        get_room_id_from_location(
            {room_id: room_polymap[room_id]}, front, verbose=False
        )
        == room_id
    ):
        return -ortho

    if TEMP_VERBOSE:
        print(f"Failed to determine normal for wall {wall_id}")
    return None


def agent_forward_vector(controller):
    a = controller.get_current_agent_full_pose()["rotation"]["y"]
    ca = np.cos(np.deg2rad(a))
    sa = np.sin(np.deg2rad(a))
    r = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    fagent = [0, 0, 1]
    fworld = r @ fagent
    fworld = fworld / np.linalg.norm(fworld)
    return fworld


def snap_to_skeleton(
    controller: "StretchController",
    corners: Sequence[Dict[str, float]],
    thinned_locs: Optional[Sequence[Dict[str, float]]] = None,
    dist_threshold: float = 0.25,
) -> Sequence[Dict[str, float]]:
    """In place modification of corners"""

    if len(corners) > 2:
        # Here we try, one last time, to adjust the path to be further from
        # obstacles on the navmesh. We do this because we might have selected
        # the smallest agent capsule above because there is a single point along
        # the path where the agent barely fits. This results in the agent sometimes
        # walking super close to doors and other obstacles along the ENTIRE path which
        # can cause the agent to get stuck for extended periods of time.
        try:
            if thinned_locs is None:
                thinned_locs = thinned_starting_positions(
                    controller.get_reachable_positions()
                )

            if thinned_locs is not None:
                thinned_pts = np.array([[p["x"], p["z"]] for p in thinned_locs])

                for i, corner in list(enumerate(corners))[1:-1]:
                    p = np.array([[corner["x"], corner["z"]]])
                    dists = np.linalg.norm(p - thinned_pts, axis=1)
                    if dists.min() <= dist_threshold:
                        closest_thinned = thinned_pts[dists.argmin()]
                        corner["x"] = float(closest_thinned[0])
                        corner["z"] = float(closest_thinned[1])
        except Exception:
            if TEMP_VERBOSE:
                print("Failed snap_to_skeleton")
    return corners
