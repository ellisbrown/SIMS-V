"""Tests for the walkthrough generator's discrete Stretch action space."""

import math

from sims.environment.action_spaces import DiscreteStretchActionSpace


def test_constructs_discrete_stretch_action_space():
    action_space = DiscreteStretchActionSpace()

    assert action_space.get_action("move_ahead").long_name == "move_ahead"


def test_discrete_action_indices_are_stable():
    action_space = DiscreteStretchActionSpace()

    assert action_space.full_action_list == [
        "move_ahead",
        "rotate_right",
        "rotate_left",
        "move_back",
        "done",
        "sub_done",
        "rotate_left_small",
        "rotate_right_small",
        "pickup",
        "move_arm_in",
        "move_arm_out",
        "move_arm_up",
        "move_arm_down",
        "wrist_open",
        "wrist_close",
        "move_arm_down_small",
        "move_arm_in_small",
        "move_arm_out_small",
        "move_arm_up_small",
        "dropoff",
    ]
    assert action_space.get_action_index_from_string("move_ahead") == 0
    assert action_space.get_action_index_from_string("rotate_right") == 1
    assert action_space.get_action_index_from_string("dropoff") == 19


def test_discrete_action_values_and_tolerances_are_versioned():
    action_space = DiscreteStretchActionSpace()
    tolerance = action_space.action_space_tolerance

    assert action_space.base_move_constant == 0.2
    assert action_space.base_rot_constant == 30
    assert action_space.get_action("rotate_left_small").to_dict()["base"]["theta"] == -6
    assert (
        math.hypot(tolerance.base_position["x"], tolerance.base_position["z"]) == 0.19
    )
    assert tolerance.base_position["theta"] == 5
    assert tolerance.wrist_pose["y"] == tolerance.wrist_pose["z"] == 0.05
