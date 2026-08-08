import math
import platform
import random
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import torch
from scipy.ndimage import binary_erosion, binary_fill_holes

from sims.environment.action_spaces import (
    PICKUP_ACTION,
    DROPOFF_ACTION,
    agent_alignment_to_point,
)
from sims.utils.constants.stretch_initialization_utils import (
    GRID_SIZE,
)
from sims.utils.data_generation_utils.bbox_utils import (
    get_box_from_object,
    compute_bbox_distance,
    BBOX_DIST_THRESHOLD,
    main_box_diagonal,
)
from sims.utils.data_generation_utils.exception_utils import (
    PlannerFailedToFindSpotOnReceptacleException,
)
from sims.utils.data_generation_utils.loc_grid_conversion import locs2grids, grids2locs
from sims.utils.data_generation_utils.navigation_utils import (
    rotate_until_visible,
    walk_on_path,
    achieve_goal_dict_with_action_space,
    action_space_can_act_on_path,
    default_nav_replan_func,
)
from sims.utils.distance_calculation_utils import position_dist
from sims.environment.stretch_state import (
    StretchState,
    convert_world_to_agent_coordinate,
    wrap_angle_to_pm180,
)

if TYPE_CHECKING:
    from sims.tasks import AbstractSimsTask

TEMP_VERBOSE = False  # Temporary diagnostic logging gate.


def get_goal_wrist_pose_for_world_point(
    world_point,
    agent_state: StretchState,
    wrist_height_offset=0.0,
    wrist_extension_offset=0.0,
    error_tolerance=0.0,
):
    arm_point = convert_world_to_agent_coordinate(world_point, agent_state, arm=True)
    # in the arm coordinates, z is the forward direction and x is the side direction (i.e. the arm
    # cannot move directly in x). Negative X is agent forward, Positive Z is agent arm out (i.e. agent right).

    point_to_reach = find_closest_reachable_point_in_arm_xz(
        arm_point["x"], arm_point["z"], error_tolerance=error_tolerance
    )
    if point_to_reach is None:
        return None

    point_to_reach["y"] = arm_point["y"]

    # get the appropriate yaw angle as a function of hand_length to achieve the X distance
    # straight out forward is a zero angle and corresponds to a 0 x value
    # negative angle corresponds to a negative X distance
    # the solution will not be unique but it should be the solution between 0 and -90 or 0 and 75 (forbidden zone)
    # TODO: will this knock the object over? Experiment with wrist_extension_offset.

    wrist_yaw = np.degrees(np.arcsin(point_to_reach["x"] / agent_state.hand_length))

    # this condition should be rare, since objects like this wouldn't be very visible so would fail other checks.
    back_wrist_required = (
        point_to_reach["z"] < agent_state.arm_extreme_values["extend_min"]
    )
    if back_wrist_required:
        # flip the wrist angle to point toward the body, i.e. reflect it around 90 degrees
        # (-45 becomes -135, 30 becomes 150, etc.)
        wrist_yaw = wrap_angle_to_pm180(180 - wrist_yaw)

    if wrist_yaw > agent_state.wrist_rotation_bounds[0]:
        if TEMP_VERBOSE:
            print(
                f"Required wrist angle in or over forbidden zone for object at arm coords {point_to_reach}"
            )
        # This also drops out quadrant 2, but I'm fine with that.
        return None

    # signed depending on whether the wrist needs to go forward or back
    z_offset_from_wrist = (-1) ** back_wrist_required * np.sqrt(
        agent_state.hand_length**2 - point_to_reach["x"] ** 2
    )

    # NOTE: already checked for object in range - just need to trim unachievable offsets
    # can't do offsets if the wrist is rotating backwards, so turn that off
    wrist_extension = np.clip(
        point_to_reach["z"]
        - z_offset_from_wrist
        - (-1) ** back_wrist_required * wrist_extension_offset,
        agent_state.arm_extreme_values["extend_min"],
        agent_state.arm_extreme_values["extend_max"],
    )

    wrist_height = np.clip(
        point_to_reach["y"] + agent_state.hand_height + wrist_height_offset,
        agent_state.arm_extreme_values["lift_min"],
        agent_state.arm_extreme_values["lift_max"],
    )

    wrist_pose = {
        "yaw": wrist_yaw,
        "z": wrist_extension,
        "y": wrist_height,
    }
    return wrist_pose


def find_closest_reachable_point_in_arm_xz(
    arm_x: float,
    arm_z: float,
    error_tolerance: Optional[float] = None,
) -> dict or None:
    dummy_state = StretchState(None)
    max_arm_x = dummy_state.hand_length
    max_arm_z = (
        dummy_state.arm_extreme_values["extend_max"] + dummy_state.hand_length + 0.03
    )
    min_arm_z = dummy_state.arm_extreme_values["extend_min"] - dummy_state.hand_length

    # Check if the point is already reachable
    if (abs(arm_x) < max_arm_x) and (min_arm_z < arm_z < max_arm_z):
        return {"x": arm_x, "z": arm_z, "required_tolerance": 0}

    if platform.system() == "Darwin":
        if TEMP_VERBOSE:
            print(f"Object at arm coords {arm_x}, {arm_z} is not reachable by the arm")

    # If the point is too close to the body, it is not reachable.
    if arm_z < min_arm_z or error_tolerance is None:
        return None

    # This is slightly weird because the wrist will not be pointing directly at the desired point in extreme cases.
    # Good for relatively small values of error_tolerance.

    # Find the closest point on the boundary x = max_arm_x
    closest_x_point_pos = (max_arm_x, min(max_arm_z, arm_z))
    closest_x_distance_pos = np.sqrt(
        (arm_x - max_arm_x) ** 2 + (arm_z - closest_x_point_pos[1]) ** 2
    )

    # Find the closest point on the boundary x = -max_arm_x
    closest_x_point_neg = (-max_arm_x, min(max_arm_z, arm_z))
    closest_x_distance_neg = np.sqrt(
        (arm_x + max_arm_x) ** 2 + (arm_z - closest_x_point_neg[1]) ** 2
    )

    # Find the closest point on the boundary z = max_arm_z
    closest_z_point = (min(max_arm_x, max(arm_x, -max_arm_x)), max_arm_z)
    closest_z_distance = np.sqrt(
        (arm_x - closest_z_point[0]) ** 2 + (arm_z - max_arm_z) ** 2
    )

    # Determine which of the three boundaries is closest
    if (
        closest_x_distance_pos < closest_x_distance_neg
        and closest_x_distance_pos < closest_z_distance
    ):
        closest_point = closest_x_point_pos
    elif (
        closest_x_distance_neg < closest_x_distance_pos
        and closest_x_distance_neg < closest_z_distance
    ):
        closest_point = closest_x_point_neg
    else:
        closest_point = closest_z_point

    # Calculate the distance between the given point and the closest point
    distance_to_closest = np.sqrt(
        (arm_x - closest_point[0]) ** 2 + (arm_z - closest_point[1]) ** 2
    )

    # Check if the distance exceeds the error_tolerance
    if distance_to_closest > error_tolerance:
        return None
    else:
        return {
            "x": closest_point[0],
            "z": closest_point[1],
            "required_tolerance": distance_to_closest,
        }


def is_arm_tucked(task):
    current_state = StretchState(controller=task.controller)
    return (
        abs(
            current_state.arm_extreme_values["extend_min"]
            - current_state.wrist_pose["z"]
        )
        < 0.1
    )


def expert_tuck_arm(task: "AbstractSimsTask", half_way=False) -> bool:
    if is_arm_tucked(task):
        return True

    current_state = StretchState(controller=task.controller)
    goal_arm_dict = {
        "wrist_pose": {
            "y": current_state.arm_extreme_values["lift_max"],
            "z": current_state.arm_extreme_values["extend_min"],
            "yaw": -130,
        }
    }
    arm_success = achieve_goal_dict_with_action_space(task, goal_arm_dict)

    if half_way and arm_success:
        return True

    current_state = StretchState(controller=task.controller)
    goal_arm_dict = {
        "wrist_pose": {
            "yaw": 180,
            "y": current_state.arm_extreme_values["lift_soft_min"],
        }
    }
    arm_success = achieve_goal_dict_with_action_space(task, goal_arm_dict)

    return is_arm_tucked(task)


def plan_positions_near_object(task, object_id):
    # TODO we could also use  GetReachablePositions with maxSeparation (currently in Jordi's local)
    rpos = task.controller.get_reachable_positions(
        grid_size=GRID_SIZE * 0.25
    )  # Use a tiny grid for this one

    obj_id_pos = task.controller.get_obj_pos_from_obj_id(object_id)

    near_rpos = [
        pos for pos in rpos if position_dist(pos, obj_id_pos, ignore_y=True) < 2
    ]

    full_poses = task.controller.get_touching_poses(
        object_id=object_id,
        positions=near_rpos,
        max_poses=100,
        max_distance=0.9,
    )
    if full_poses is None:
        return []
    poses = [pose["position"] for pose in full_poses]

    # go through these poses and only keep sufficiently unique x-z coordinates
    # Visibility imposes an implicit distance limit on these candidate poses.
    unique_poses = []
    for pose in poses:
        # check x and z, ignore y
        if len(unique_poses) == 0:
            unique_poses.append(pose)
        else:
            for up in unique_poses:
                if position_dist(pose, up, ignore_y=True) < 0.05:
                    break
            else:
                unique_poses.append(pose)
    poses = unique_poses

    poses = sorted(
        poses,
        key=lambda x: min(
            [
                position_dist(
                    x,
                    {"x": point["x"], "y": point["y"], "z": point["z"]},
                    ignore_y=True,
                )
                for point in task.controller.step(
                    action="PointOnObjectsCollidersClosestToPoint",
                    objectId=object_id,
                    point={"x": x["x"], "y": x["y"], "z": x["z"]},
                ).metadata["actionReturn"]
            ]
        ),
    )
    return poses


def try_to_get_closer_to_object(
    task, object_id, max_tries=5, replan_func=default_nav_replan_func
):
    poses = plan_positions_near_object(task, object_id)

    path = None
    nav_success = False
    valid_attempts = 0
    while (
        path is None and len(poses) > 0 and valid_attempts < max_tries
    ):  # has to be a nontrivial change
        # choose pose randomly from the top 10
        random_pose = random.choice(poses[:10])
        poses.remove(random_pose)

        # allow for getting closer to things
        path = replan_func(task, random_pose, True, None)
        if path is None:
            continue

        # TODO we might want to allow this.
        #  from a far enough distance we might be able to approach the target with a better angle, or prevent
        #  collisions with nearby objects
        # path_to_pose_distance = position_dist(random_pose, path[-1], ignore_y=True)
        # if path_to_pose_distance > 0.2:
        #     print(f"Path to pose distance is quite large - why?: {path_to_pose_distance}. Skipping")
        #     path = None
        #     continue

        if action_space_can_act_on_path(task, path):
            nav_success = walk_on_path(path, task, max_tries=2, replan_func=replan_func)
            valid_attempts += 1
        else:
            if TEMP_VERBOSE:
                print("invalid attempt", valid_attempts)
            path = None
            nav_success = False
            continue
        if TEMP_VERBOSE:
            print("nav success for getting closer", nav_success)
    return nav_success


def get_nearest_object_point(task, object_id, position=None, return_all=False):
    if position is None:
        agent_state = StretchState(controller=task.controller)
        relevant_agent_pos = agent_state.base_position.copy()
        relevant_agent_pos["y"] = np.min(
            [
                task.env.get_object(object_id)["position"]["y"],
                agent_state.max_interactable_height,
            ]
        )
    else:
        relevant_agent_pos = position
    points_on_obj = task.controller.step(
        action="PointOnObjectsCollidersClosestToPoint",
        objectId=object_id,
        point=relevant_agent_pos,
    ).metadata["actionReturn"]

    if points_on_obj is None or len(points_on_obj) == 0:
        return None, None

    all_distances = [
        position_dist(p, relevant_agent_pos, ignore_y=True) for p in points_on_obj
    ]
    points, distances = zip(
        *sorted(zip(points_on_obj, all_distances), key=lambda x: x[1])
    )
    if return_all:
        return points, distances

    return points[0], distances[0]


def get_arm_interaction_pose_for_current_agent_position(
    task: "AbstractSimsTask",
    object_id: str,
    wrist_height_offset: float = 0.0,
    wrist_extension_offset: float = 0.0,
):
    agent_state = StretchState(controller=task.controller)
    points_on_obj, all_distances = get_nearest_object_point(
        task, object_id, return_all=True
    )
    if points_on_obj is None:
        return False
    # go through sorted all_distances up to 1M away and look for wrist poses
    wrist_pose = None
    for point, dist in sorted(zip(points_on_obj, all_distances), key=lambda x: x[1]):
        if dist > 1.1:
            # don't bother anymore
            break
        else:
            # try for pose
            wrist_pose = get_goal_wrist_pose_for_world_point(
                world_point=point,
                agent_state=agent_state,
                wrist_height_offset=wrist_height_offset,
                wrist_extension_offset=wrist_extension_offset,
            )
            if wrist_pose is not None:
                break

    return wrist_pose


def expert_pick_up_from_current_position(
    task, object_id: str, fake_failure_permitted=False
):
    # ASSUMPTIONS:
    # 1. The agent is close enough to the object to pick it up
    # 2. The arm is tucked or halfway tucked
    # 3. The object is not in the hand
    # 4. The object is visible enough for interaction

    random_wrist_height_offset = 0.0
    random_wrist_extension_offset = 0.0
    random_failure = False
    if fake_failure_permitted and random.random() < 0.3:
        random_failure = True
        random_wrist_height_offset = random.uniform(0, 0.15)
        random_wrist_extension_offset = random.uniform(0, 0.15)

    setup_wrist_pose = get_arm_interaction_pose_for_current_agent_position(
        task,
        object_id,
        wrist_height_offset=0.35 + random_wrist_height_offset,
        wrist_extension_offset=random_wrist_extension_offset,
    )
    if setup_wrist_pose is None:
        return False

    achieve_goal_dict_with_action_space(
        task=task, goal_dict={"wrist_pose": setup_wrist_pose}
    )

    # note: smaller vertical offset
    grasp_wrist_pose = get_arm_interaction_pose_for_current_agent_position(
        task,
        object_id,
        wrist_height_offset=0.03 + random_wrist_height_offset,
        wrist_extension_offset=random_wrist_extension_offset,
    )  # half the sphere radius (6cm)
    if grasp_wrist_pose is None:
        # This one is rare but it can happen
        return False

    grasp_pose_success = achieve_goal_dict_with_action_space(
        task=task, goal_dict={"wrist_pose": grasp_wrist_pose}
    )
    if random_failure:
        task.step_with_random(task.get_action_from_goal(PICKUP_ACTION))
    elif grasp_pose_success:
        task.step_with_random(task.get_action_from_goal(PICKUP_ACTION))

    # get a goal pose that is 10cm above current height
    agent_state = StretchState(controller=task.controller)
    check_pickup_goal_lift = np.clip(
        agent_state.wrist_pose["y"] + 0.1,
        agent_state.arm_extreme_values["lift_min"],
        agent_state.arm_extreme_values["lift_max"],
    )
    check_pickup_goal_extension = np.clip(
        agent_state.wrist_pose["z"] - 0.1,
        agent_state.arm_extreme_values["extend_min"],
        agent_state.arm_extreme_values["extend_max"],
    )
    goal_check_pickup = {
        "wrist_pose": {"y": check_pickup_goal_lift, "z": check_pickup_goal_extension}
    }
    achieve_goal_dict_with_action_space(task=task, goal_dict=goal_check_pickup)

    if object_id in agent_state.held_oids:
        return True
    elif fake_failure_permitted and random_failure:  # redundant, but readable
        return expert_pick_up_from_current_position(
            task, object_id, fake_failure_permitted=False
        )
    else:
        return False


def expert_pick_up_object_with_retries(
    task: "AbstractSimsTask",
    object_id: str,
    num_retries: int = 3,
    replan_func=default_nav_replan_func,
):
    # Assumptions coming into this function: agent has moved to a new location which is close to the object
    pickup_success = False
    if num_retries < 0:
        return pickup_success

    fake_failure_permitted = True
    if num_retries < 3:
        fake_failure_permitted = False

    if not is_arm_tucked(task):
        expert_tuck_arm(task, half_way=True)

    # current pose checks
    rotate_until_visible(
        task=task,
        object_id=object_id,
        use_object_nav_strict_success=True,
        manipulation_camera=True,
    )

    if not pickup_success:
        # do this five times or until the position dist is less than 1 (i.e. theoretically possible)
        closer_success = False
        for _ in range(5):
            # new add: check if this pose is good enough
            wrist_pose = get_arm_interaction_pose_for_current_agent_position(
                task, object_id
            )
            object_sufficiently_visible = (
                task.controller.is_object_visible_enough_for_interaction(
                    object_id, manipulation_camera=True
                )
            )
            if object_sufficiently_visible and wrist_pose is not None:
                pickup_success = expert_pick_up_from_current_position(
                    task=task,
                    object_id=object_id,
                    fake_failure_permitted=fake_failure_permitted,
                )
                if pickup_success:
                    return pickup_success
                else:
                    expert_tuck_arm(task, half_way=True)

            starting_state = StretchState(controller=task.controller)
            closer_success = try_to_get_closer_to_object(
                task, object_id, replan_func=replan_func
            )
            object_nearest_point, object_nearest_distance = get_nearest_object_point(
                task, object_id
            )

            # check if anything actually happened
            state_difference = StretchState.difference(
                final_state=StretchState(controller=task.controller),
                initial_state=starting_state,
            )
            change_too_small, _ = StretchState.state_change_within_tolerance(
                state_difference, task.agent_action_space.action_space_tolerance
            )

            if change_too_small:
                if TEMP_VERBOSE:
                    print(
                        f"Apparently no closer pose could be reached and we are stuck. "
                        f"Breaking out and trying a new approach. {object_nearest_distance}"
                    )
                # TODO: I would like to be able to grab a nearby open floor area here to go to.
                closer_success = False
                break

            # Reaching this distance confirms forward progress.
            if object_nearest_distance < 1.05:
                if TEMP_VERBOSE:
                    print(f"trying to get closer worked {object_nearest_distance}")
                closer_success = True
                break
            elif closer_success:
                if TEMP_VERBOSE:
                    print(
                        f"trying to get closer didn't work, dist: {object_nearest_distance}"
                    )
        if not closer_success:
            # # if it failed or we are stuck, go to a room centroid and try again
            # room_id = task.get_current_room()
            # path_to_room_center = path = task.controller.get_shortest_path_to_room(room_id=room_id)
            # instead of the room center, just go to a reachable point less than a meter away
            current_state = StretchState(controller=task.controller)
            candidate_positions = task.controller.get_reachable_positions(grid_size=0.2)
            # only keep the ones that are less than a meter away
            candidate_positions = [
                pos
                for pos in candidate_positions
                if position_dist(pos, current_state.base_position, ignore_y=True) < 2
            ]
            unstick_path = None
            for pos in candidate_positions:
                unstick_path = replan_func(task, pos, True, None)
                # unstick_path = task.controller.get_shortest_path_to_point(
                #     target_position=pos, attempt_path_improvement=True
                # )
                if (
                    unstick_path is not None
                    and action_space_can_act_on_path(task, unstick_path)
                    and len(unstick_path) < 6
                ):
                    break
            unstick_success = False
            if action_space_can_act_on_path(task, unstick_path):
                unstick_success = walk_on_path(
                    unstick_path, task, max_tries=10, replan_func=replan_func
                )
                if unstick_success:
                    try_to_get_closer_to_object(
                        task, object_id, replan_func=replan_func
                    )
            if not unstick_success:
                # if we can't get closer and we can't get out, we're stuck. end the recursion
                return False
        pickup_success = expert_pick_up_object_with_retries(
            task, object_id, num_retries - 1, replan_func=replan_func
        )

    return pickup_success


def expert_object_placement_from_current_position(
    task: "AbstractSimsTask",
    object_id: str,
    receptacle_id: str,
    ideal_position: Dict[str, float] = None,
    error_tolerance: float = 0.15,
):
    # ASSUMPTIONS:
    # 1. The agent is close enough to the receptacle to place the object and is rotated to arm-face the desired point
    # 2. The arm is halfway tucked
    # 3. The object is in the hand
    # 4. A feasible wrist pose can be found for the receptacle, unless something odd has happened

    agent_state = StretchState(controller=task.controller)
    wrist_pose = get_goal_wrist_pose_for_world_point(
        world_point=ideal_position,
        agent_state=agent_state,
        wrist_height_offset=0.35,
        error_tolerance=error_tolerance,
    )
    if wrist_pose is None:
        return False

    achieve_goal_dict_with_action_space(task=task, goal_dict={"wrist_pose": wrist_pose})

    object_bbox = get_box_from_object(task.controller.get_object(object_id))
    lowest_y = min([p[1] for p in object_bbox])
    min_y_offset_from_wrist = min(
        StretchState(controller=task.controller).hand_position["y"] - lowest_y, 0
    )
    wrist_pose = get_goal_wrist_pose_for_world_point(
        world_point=ideal_position,
        agent_state=agent_state,
        wrist_height_offset=min_y_offset_from_wrist + 0.03,
        error_tolerance=error_tolerance,
    )
    if wrist_pose is None:
        expert_tuck_arm(task, half_way=True)
        return False

    achieve_goal_dict_with_action_space(task=task, goal_dict={"wrist_pose": wrist_pose})
    task.step_with_random(task.get_action_from_goal(DROPOFF_ACTION))

    agent_state = StretchState(controller=task.controller)
    check_dropoff_y = np.clip(
        agent_state.wrist_pose["y"] + 0.1,
        agent_state.arm_extreme_values["lift_min"],
        agent_state.arm_extreme_values["lift_max"],
    )
    check_dropoff_z = np.clip(
        agent_state.wrist_pose["z"] - 0.1,
        agent_state.arm_extreme_values["extend_min"],
        agent_state.arm_extreme_values["extend_max"],
    )
    goal_check_dropoff = {"wrist_pose": {"y": check_dropoff_y, "z": check_dropoff_z}}
    achieve_goal_dict_with_action_space(task=task, goal_dict=goal_check_dropoff)
    receptacles = [
        o["objectId"]
        for o in task.controller.get_objects_that_objects_are_on(
            object_ids=[object_id]
        )[object_id]
    ]
    return receptacle_id in receptacles


def expert_place_object_on_receptacle_with_retries(
    receptacle_id,
    task,
    ideal_position=None,
    object_diagonal=None,
    close_to_center=False,
    num_retries: int = 3,
    replan_func=default_nav_replan_func,
):
    if num_retries < 0:
        return False

    starting_agent_state = StretchState(controller=task.controller)
    starting_agent_position = starting_agent_state.base_position.copy()
    starting_agent_position["y"] = np.min(
        [
            task.env.get_object(receptacle_id)["position"]["y"],
            starting_agent_state.max_interactable_height,
        ]
    )

    held_objects = starting_agent_state.held_oids
    task_relevant_held_object = [
        o for o in held_objects if o in task.task_relevant_oids
    ]
    if len(task_relevant_held_object) == 0:
        return False  # cannot place what is not held
    else:
        task_relevant_held_object = task_relevant_held_object[0]
    object_width = position_dist(
        task.controller.get_object(task_relevant_held_object)["position"],
        starting_agent_state.hand_position,
        ignore_y=True,
    )
    placement_tolerance = 0.1 + object_width

    try:
        if not close_to_center:
            random_spawn_locations_on_receptacle = (
                get_possible_spawn_locations_close_to_agent(
                    task.controller, receptacle_id, object_diagonal=object_diagonal
                )
            )
        else:
            random_spawn_locations_on_receptacle = (
                get_possible_spawn_locations_close_to_center(
                    task.controller, receptacle_id, object_diagonal=object_diagonal
                )
            )
        if ideal_position is None:
            # Order locations by distance to agent center
            target_locations = sorted(
                random_spawn_locations_on_receptacle,
                key=lambda p: position_dist(p, starting_agent_position, ignore_y=True),
            )
        else:
            # Order locations by distance to ideal position
            target_locations = sorted(
                random_spawn_locations_on_receptacle,
                key=lambda p: position_dist(ideal_position, p),
            )

    except Exception:
        raise PlannerFailedToFindSpotOnReceptacleException

    # remove any location whose agent distance is greater than 1.5
    target_locations = [
        loc
        for loc in target_locations
        if position_dist(loc, starting_agent_position, ignore_y=True) < 1.2
    ]

    for _ in range(5):  # try up to five points on the receptacle
        if len(target_locations) == 0:
            # no plausible locations left
            break

        # randomly choose from the top 3 remaining locations
        interim_target_location = random.choice(target_locations[:3])
        target_locations.remove(interim_target_location)

        current_state = StretchState(controller=task.controller)
        angle_to_target_loc = agent_alignment_to_point(
            current_state, interim_target_location, arm=True
        )
        goal_rotation = {
            "base_position": {
                "theta": angle_to_target_loc + current_state.base_position["theta"]
            }
        }
        achieve_goal_dict_with_action_space(task, goal_rotation)
        agent_alignment_to_point(
            StretchState(controller=task.controller), interim_target_location, arm=True
        )

        wrist_pose, target_location = find_laziest_wrist_pose(
            task=task,
            target_locations=target_locations,
            placement_tolerance=placement_tolerance,
        )

        if wrist_pose is None:
            continue
        else:
            target_locations.remove(target_location)
            placement_success = expert_object_placement_from_current_position(
                task=task,
                object_id=task_relevant_held_object,
                receptacle_id=receptacle_id,
                ideal_position=target_location,
                error_tolerance=placement_tolerance,
            )
            if not placement_success:
                break
            else:
                return True

    # check if the object is still held
    if task_relevant_held_object not in task.controller.get_held_objects():
        pick_up_again = expert_pick_up_object_with_retries(
            task=task, object_id=task_relevant_held_object, replan_func=replan_func
        )
        if not pick_up_again:
            return False  # ultimate failure, rip
        expert_tuck_arm(task=task, half_way=True)
    try_to_get_closer_to_object(task, receptacle_id, replan_func=replan_func)
    return expert_place_object_on_receptacle_with_retries(
        receptacle_id=receptacle_id,
        task=task,
        ideal_position=ideal_position,
        object_diagonal=object_diagonal,
        close_to_center=close_to_center,
        num_retries=num_retries - 1,
        replan_func=replan_func,
    )


def find_laziest_wrist_pose(
    task: "AbstractSimsTask",
    target_locations: list,
    placement_tolerance: float,
    wrist_height_offset: float = 0.0,
    wrist_extension_offset: float = 0.0,
):
    """
    we're not lazy we're efficient
    optimal even
    find the one that minimizes (in order) yaw from 0, z from 0, and y from current
    """
    current_agent_state = StretchState(controller=task.controller)
    wrist_poses_with_locations = [
        (
            get_goal_wrist_pose_for_world_point(
                world_point=p,
                agent_state=current_agent_state,
                error_tolerance=placement_tolerance,
                wrist_height_offset=wrist_height_offset,
                wrist_extension_offset=wrist_extension_offset,
            ),
            p,
        )
        for p in target_locations
    ]

    valid_wrist_poses_with_locations = [
        (wp, loc) for wp, loc in wrist_poses_with_locations if wp is not None
    ]

    if not valid_wrist_poses_with_locations:
        return None, None

    wrist_pose_with_location = min(
        valid_wrist_poses_with_locations,
        key=lambda item: (
            abs(item[0]["yaw"]),
            abs(item[0]["z"]),
            abs(item[0]["y"] - current_agent_state.wrist_pose["y"]),
        ),
    )
    return wrist_pose_with_location


def expert_move_object_in_hand_close_to_target(target_id, task):
    current_state = StretchState(controller=task.controller)
    vector = task.controller.get_agent_alignment_to_object(
        target_id, use_arm_orientation=True
    )
    goal_rotation = {
        "base_position": {"theta": vector + current_state.base_position["theta"]}
    }
    achieve_goal_dict_with_action_space(task, goal_rotation)
    task.controller.get_agent_alignment_to_object(target_id, use_arm_orientation=True)
    object_points, point_distances = get_nearest_object_point(
        task, target_id, return_all=True
    )

    starting_agent_state = StretchState(task.controller)
    held_objects = starting_agent_state.held_oids
    task_relevant_held_object = [
        o for o in held_objects if o in task.task_relevant_oids
    ]
    if len(task_relevant_held_object) == 0:
        return False  # cannot place what is not held
    else:
        task_relevant_held_object = task_relevant_held_object[0]

    object_to_hand = position_dist(
        task.controller.get_object(task_relevant_held_object)["position"],
        starting_agent_state.hand_position,
        ignore_y=True,
    )
    diag = main_box_diagonal(
        get_box_from_object(task.controller.get_object(task_relevant_held_object))
    )

    placement_tolerance = object_to_hand + diag / 2

    initial_dist = compute_bbox_distance(
        get_box_from_object(task.controller.get_object(task_relevant_held_object)),
        get_box_from_object(task.controller.get_object(target_id)),
    )

    if TEMP_VERBOSE:
        print("INITIAL", initial_dist)

    if initial_dist < BBOX_DIST_THRESHOLD:
        return True

    # very generous here with proximity
    wrist_pose, target_location = find_laziest_wrist_pose(
        task=task,
        target_locations=[
            object_points[i]
            for i in range(len(object_points))
            if point_distances[i] < 1.1
        ],
        placement_tolerance=BBOX_DIST_THRESHOLD / 2,
        wrist_height_offset=diag / 2 + BBOX_DIST_THRESHOLD / 2,
        wrist_extension_offset=placement_tolerance,
    )
    if wrist_pose is None:
        return False
    else:
        achieve_goal_dict_with_action_space(task, {"wrist_pose": wrist_pose})

    high_dist = compute_bbox_distance(
        get_box_from_object(task.controller.get_object(task_relevant_held_object)),
        get_box_from_object(task.controller.get_object(target_id)),
    )

    if TEMP_VERBOSE:
        print("HIGH", high_dist)

    if high_dist < BBOX_DIST_THRESHOLD:
        return True

    wrist_pose, target_location = find_laziest_wrist_pose(
        task=task,
        target_locations=[
            object_points[i]
            for i in range(len(object_points))
            if point_distances[i] < 3
        ],
        placement_tolerance=BBOX_DIST_THRESHOLD / 2,
        wrist_extension_offset=placement_tolerance,
        wrist_height_offset=diag / 2,
    )
    if wrist_pose is None:
        return False
    else:
        achieve_goal_dict_with_action_space(task, {"wrist_pose": wrist_pose})

    low_dist = compute_bbox_distance(
        get_box_from_object(task.controller.get_object(task_relevant_held_object)),
        get_box_from_object(task.controller.get_object(target_id)),
    )

    if TEMP_VERBOSE:
        print("LOW", low_dist)

    if low_dist < BBOX_DIST_THRESHOLD:
        return True

    return False


def expert_move_object_in_hand_close_to_target_with_retries(
    target_id,
    task,
    num_retries: int = 3,
    replan_func=default_nav_replan_func,
):
    present_success = False
    if num_retries < 0:
        return present_success

    if not is_arm_tucked(task):
        expert_tuck_arm(task, half_way=True)

    present_success = expert_move_object_in_hand_close_to_target(target_id, task)
    if not present_success and num_retries > 0:
        expert_tuck_arm(task, half_way=True)
        # do this five times or until the position dist is less than 1 (i.e. theoretically possible)
        closer_success = False
        for _ in range(5):
            starting_state = StretchState(controller=task.controller)
            closer_success = try_to_get_closer_to_object(
                task, target_id, replan_func=replan_func
            )
            object_nearest_point, object_nearest_distance = get_nearest_object_point(
                task, target_id
            )

            # check if anything actually happened
            state_difference = StretchState.difference(
                final_state=StretchState(controller=task.controller),
                initial_state=starting_state,
            )
            change_too_small, _ = StretchState.state_change_within_tolerance(
                state_difference, task.agent_action_space.action_space_tolerance
            )

            if change_too_small:
                if TEMP_VERBOSE:
                    print(
                        f"Apparently no closer pose could be reached and we are stuck. "
                        f"Breaking out and trying a new approach. {object_nearest_distance}"
                    )
                # TODO: I would like to be able to grab a nearby open floor area here to go to.
                closer_success = False
                break

            # Reaching this distance confirms forward progress.
            if object_nearest_distance < 1.05:
                if TEMP_VERBOSE:
                    print(f"trying to get closer worked {object_nearest_distance}")
                closer_success = True
                break
            elif closer_success:
                if TEMP_VERBOSE:
                    print(
                        f"trying to get closer didn't work, dist: {object_nearest_distance}"
                    )

        if not closer_success:
            # # if it failed or we are stuck, go to a room centroid and try again
            # room_id = task.get_current_room()
            # path_to_room_center = path = task.controller.get_shortest_path_to_room(room_id=room_id)
            # instead of the room center, just go to a reachable point less than a meter away
            current_state = StretchState(controller=task.controller)
            candidate_positions = task.controller.get_reachable_positions(grid_size=0.2)
            # only keep the ones that are less than a meter away
            candidate_positions = [
                pos
                for pos in candidate_positions
                if position_dist(pos, current_state.base_position, ignore_y=True) < 2
            ]
            unstick_path = None
            for pos in candidate_positions:
                unstick_path = replan_func(task, pos, True, None)
                if (
                    unstick_path is not None
                    and action_space_can_act_on_path(task, unstick_path)
                    and len(unstick_path) < 6
                ):
                    break
            unstick_success = False
            if action_space_can_act_on_path(task, unstick_path):
                unstick_success = walk_on_path(
                    unstick_path, task, max_tries=10, replan_func=replan_func
                )
                if unstick_success:
                    try_to_get_closer_to_object(
                        task, target_id, replan_func=replan_func
                    )
            if not unstick_success:
                # if we can't get closer and we can't get out, we're stuck. end the recursion
                return False
        present_success = expert_move_object_in_hand_close_to_target_with_retries(
            target_id, task, num_retries - 1, replan_func=replan_func
        )

    return present_success


def erode_locations(locations, object_diagonal, ignore_holes=False, grid_spacing=0.05):
    im, locs = locs2grids(locations, grid_spacing)

    if ignore_holes:
        im = binary_fill_holes(im)

    half_diag = math.ceil((object_diagonal / 2) / grid_spacing)
    im = binary_erosion(im, iterations=half_diag)

    if np.sum(im) == 0:
        # trying with all locations
        return locations

    return grids2locs(im, locs)


def get_possible_spawn_locations_close_to_center(
    controller, receptacle_id, object_diagonal=None
):
    locations = controller.get_locations_on_receptacle(receptacle_id)
    if object_diagonal is not None:
        locations = erode_locations(locations, object_diagonal)
    locations = convert_xyz_to_torch_tensor(locations)
    center = locations.mean(dim=0, keepdim=True)
    # current_agent_position = convert_xyz_to_torch_tensor(controller.get_current_agent_position())
    distances = (locations - center).norm(dim=-1)
    closest_indices = torch.topk(
        distances, k=min(20, len(distances)), largest=False
    ).indices
    tensor_xyz = locations[closest_indices]
    return convert_torch_tensor_to_xyz(tensor_xyz)


def get_possible_spawn_locations_close_to_agent(
    controller, receptacle_id, object_diagonal=None
):
    locations = controller.get_locations_on_receptacle(receptacle_id)
    if object_diagonal is not None:
        locations = erode_locations(locations, object_diagonal)
    locations = convert_xyz_to_torch_tensor(locations)
    current_agent_position = convert_xyz_to_torch_tensor(
        controller.get_current_agent_position()
    )
    distances = (locations - current_agent_position).norm(dim=-1)
    closest_indices = torch.topk(
        distances, k=min(20, len(distances)), largest=False
    ).indices
    tensor_xyz = locations[closest_indices]
    return convert_torch_tensor_to_xyz(tensor_xyz)


def get_reachable_locations_close_to_object(
    controller, object_id, locations, num_locs=20
):
    locations = convert_xyz_to_torch_tensor(locations)
    current_object_position = convert_xyz_to_torch_tensor(
        controller.get_object_position(object_id)
    )
    distances = (locations - current_object_position).norm(dim=-1)
    closest_indices = torch.topk(
        distances, k=min(num_locs, len(distances)), largest=False
    ).indices
    tensor_xyz = locations[closest_indices]
    return convert_torch_tensor_to_xyz(tensor_xyz)


def calc_arm_movement(arm_1, arm_2):
    total_dist = 0
    for k in ["x", "y", "z"]:
        total_dist += (arm_1[k] - arm_2[k]) ** 2

    return total_dist**0.5


def convert_xyz_to_torch_tensor(list_of_xyz):
    if isinstance(list_of_xyz, dict):
        return torch.Tensor([list_of_xyz["x"], list_of_xyz["y"], list_of_xyz["z"]])
    return torch.Tensor([[x["x"], x["y"], x["z"]] for x in list_of_xyz])


def convert_torch_tensor_to_xyz(tensor_xyz):
    tensor_xyz = tensor_xyz.numpy()
    return [dict(x=x[0], y=x[1], z=x[2]) for x in tensor_xyz]
