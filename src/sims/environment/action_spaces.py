import abc
import json
import math
import random
from typing import Dict, Optional, Union, List

from sims.environment.actions import (
    AbstractAction,
    StretchAction,
    StretchGraspAction,
    StretchDropOffAction,
)
from sims.environment.stretch_state import (
    StretchState,
    angle_point_to_point,
    convert_world_to_agent_coordinate,
    wrap_angle_to_pm180,
)

# Keep this module dependency-light to avoid circular environment imports.

# Special action names shared by the supported action spaces.
DONE_ACTION = "done"
SUB_DONE_ACTION = "sub_done"
PICKUP_ACTION = "pickup"
DROPOFF_ACTION = "dropoff"


# Helpers for state-difference actions live here to keep action-space
# construction independent of higher-level project modules.


def relative_base_degrees_toward_delta_state(
    difference_state: StretchState, current_state: StretchState = None
):
    if current_state is None:
        # using agent-relative coordinates, so the agent is always facing positive Z
        current_heading = 0
    else:
        current_heading = current_state.base_position["theta"]

    delta_x = difference_state.base_position["x"]
    delta_z = difference_state.base_position["z"]

    if delta_x == 0 and delta_z == 0:
        return 0  # No rotation needed if there is no movement

    result = math.degrees(math.atan2(delta_x, delta_z))
    result -= current_heading

    return wrap_angle_to_pm180(result)


class AbstractActionSpace(abc.ABC):
    action_type = AbstractAction
    random_min_translation = 0.05
    random_max_translation = 0.2
    random_min_rotation = 5
    random_max_rotation = 30

    @abc.abstractmethod
    def _get_singleton_action(
        self, action_string: str = "", action_quantity: float = None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(
        self,
        action_strings: Union[str, List[str]],
        action_quantities: Union[float, List[float]] = None,
    ) -> AbstractAction:
        raise NotImplementedError

    # The meaningful label depends on the actions supported by the space.
    # most general case for AbstractAction provided here
    def get_meaningful_action_string(self, action_call: AbstractAction):
        # make a dict of the action string and quantity for each component
        return action_call.to_string()

    @abc.abstractmethod
    def state_difference_to_action(
        self,
        current_state: StretchState,
        state_difference: StretchState = None,
        final_state: StretchState = None,
    ) -> Optional[StretchAction]:
        raise NotImplementedError

    @staticmethod
    def _set_action_space_tolerance(
        base_meters=0.05, arm_meters=0.02, degrees=3
    ) -> StretchState:
        # class property StretchState that is a difference representing the tolerance of the action space
        # i.e. if current_state.difference(final_state) is within the tolerance, then the action space can't do anything
        # and get_action will return None
        xzbreakout = base_meters / math.sqrt(2)
        return StretchState._create_difference_state(
            diff_base={"x": xzbreakout, "z": xzbreakout, "theta": degrees},
            diff_wrist={"y": arm_meters, "z": arm_meters, "yaw": degrees},
            diff_hand={
                "x": arm_meters,
                "y": arm_meters,
                "z": arm_meters,
            },  # this one is a little weird
            diff_gripper=0,
            diff_held_oids=set(),
        )

    def get_action_from_meaningful_action_string(self, action_string) -> AbstractAction:
        if action_string in [
            PICKUP_ACTION,
            DROPOFF_ACTION,
            DONE_ACTION,
            SUB_DONE_ACTION,
        ]:
            return self.get_action(action_string)

        else:
            # get the json string and parse it
            action_dict = json.loads(action_string)
            if "action_values" in action_dict:
                action_dict = action_dict["action_values"]
            action_types, action_values = self.flatten_stretch_action_dict(action_dict)
            return self.get_action(action_types, action_values)

    @staticmethod
    def flatten_stretch_action_dict(action_dict):
        keys = []
        values = []
        for outer_key, inner_dict in action_dict.items():
            for inner_key, value in inner_dict.items():
                # Create the tuple and add it to the list
                keys.append(f"{outer_key}_{inner_key}")
                values.append(value)
        return keys, values

    def sample_random_action_quantity(self, action_string: str):
        if action_string in ["base_z", "arm_y", "arm_z"]:
            return random.uniform(
                self.random_min_translation, self.random_max_translation
            )
        elif action_string in ["base_theta", "wrist_yaw"]:
            return random.uniform(self.random_min_rotation, self.random_max_rotation)
        else:
            raise NotImplementedError(f"Action string {action_string} not recognized")


class DiscreteStretchActionSpace(AbstractActionSpace):
    def __init__(self):
        self.is_tokenized = False
        self.action_type = StretchAction
        self.grasp_action = StretchGraspAction
        self.dropoff_action = StretchDropOffAction

        self.arm_move_constant = 0.1
        self.base_move_constant = 0.2
        self.base_rot_constant = 30
        self.wrist_rotation_constant = 10

        self.random_translation_eps = 1e-5
        self.random_min_translation = (
            0.2 + self.random_translation_eps
        )  # larger than tolerance
        self.random_max_translation = 0.2 + self.random_translation_eps
        self.random_min_rotation = 30
        self.random_max_rotation = 30

        self.base_actions = [
            self.action_type(
                base=dict(z=self.base_move_constant),
                long_name="move_ahead",
                short_name="m",
            ),
            self.action_type(
                base=dict(z=-self.base_move_constant),
                long_name="move_back",
                short_name="b",
            ),
            self.action_type(
                base=dict(theta=-self.base_rot_constant),
                long_name="rotate_left",
                short_name="l",
            ),
            self.action_type(
                base=dict(theta=self.base_rot_constant),
                long_name="rotate_right",
                short_name="r",
            ),
            self.action_type(
                base=dict(theta=-self.base_rot_constant / 5),
                long_name="rotate_left_small",
                short_name="ls",
            ),
            self.action_type(
                base=dict(theta=self.base_rot_constant / 5),
                long_name="rotate_right_small",
                short_name="rs",
            ),
        ]
        self.arm_actions = [
            self.action_type(
                arm=dict(y=self.arm_move_constant),
                long_name="move_arm_up",
                short_name="yp",
            ),
            self.action_type(
                arm=dict(y=-self.arm_move_constant),
                long_name="move_arm_down",
                short_name="ym",
            ),
            self.action_type(
                arm=dict(z=self.arm_move_constant),
                long_name="move_arm_out",
                short_name="zp",
            ),
            self.action_type(
                arm=dict(z=-self.arm_move_constant),
                long_name="move_arm_in",
                short_name="zm",
            ),
            self.action_type(
                arm=dict(y=self.arm_move_constant / 5),
                long_name="move_arm_up_small",
                short_name="yps",
            ),
            self.action_type(
                arm=dict(y=-self.arm_move_constant / 5),
                long_name="move_arm_down_small",
                short_name="yms",
            ),
            self.action_type(
                arm=dict(z=self.arm_move_constant / 5),
                long_name="move_arm_out_small",
                short_name="zps",
            ),
            self.action_type(
                arm=dict(z=-self.arm_move_constant / 5),
                long_name="move_arm_in_small",
                short_name="zms",
            ),
        ]
        self.wrist_actions = [
            self.action_type(
                wrist=dict(yaw=-self.wrist_rotation_constant),
                long_name="wrist_open",
                short_name="wp",
            ),
            self.action_type(
                wrist=dict(yaw=self.wrist_rotation_constant),
                long_name="wrist_close",
                short_name="wm",
            ),
        ]
        self.special_actions = [
            self.action_type(long_name=DONE_ACTION, short_name="end"),
            self.action_type(long_name=SUB_DONE_ACTION, short_name=SUB_DONE_ACTION),
            self.grasp_action(long_name=PICKUP_ACTION, short_name="p"),
            self.dropoff_action(long_name=DROPOFF_ACTION, short_name="d"),
        ]
        self.spatial_actions = [
            *self.base_actions,
            *self.arm_actions,
            *self.wrist_actions,
        ]

        self.action_space_tolerance = self._set_action_space_tolerance(
            base_meters=0.19, arm_meters=0.05, degrees=5
        )
        action_list_order = [
            "m",
            "r",
            "l",
            "b",
            "end",
            "sub_done",
            "ls",
            "rs",
            "p",
            "zm",
            "zp",
            "yp",
            "ym",
            "wp",
            "wm",
            "yms",
            "zms",
            "zps",
            "yps",
            "d",
        ]

        self.full_action_list = [
            self.get_action(short_name).long_name for short_name in action_list_order
        ]

    def get_action_index_from_string(self, action_string: str):
        if action_string == "":
            return (
                len(self.full_action_list) + 1
            )  # 0-19 are actions, 20 is for padding, and 21 is for start token ("")
        return self.full_action_list.index(action_string)

    @property
    def long_action_names(self):
        return [action for action in self.full_action_list]

    def get_num_actions(self):
        return len(self.full_action_list)

    def get_action_list(self):
        return self.full_action_list

    # This is a property of the action space because "meaningful" is going to depend on the possibilities
    # simple for the discrete Stretch but requires more thought otherwise
    def get_meaningful_action_string(self, action_call: StretchAction):
        return action_call.long_name

    def _get_singleton_action(
        self, action_string: str = "", action_quantity: float = None
    ):
        # Singleton lookup covers spatial and special discrete actions.
        for action in [*self.spatial_actions, *self.special_actions]:
            if action_string in [action.long_name, action.short_name]:
                return action
        raise ValueError(f"Action string {action_string} not recognized")

    def get_action(
        self,
        action_strings: Union[str, List[str]],
        action_quantities: Union[float, List[float]] = None,
    ) -> StretchAction:
        # This action space doesn't support multiple actions at once so this is 1:1.
        return self._get_singleton_action(action_strings, action_quantities)

    def get_action_from_meaningful_action_string(self, action_string) -> AbstractAction:
        return self._get_singleton_action(action_string)

    def calculate_time_step_needed_to_complete(self, action):
        # Tokenized action spaces count one step per emitted action token.
        if self.is_tokenized is False:
            return 1
        else:
            return len(self.tokenizer.get_action_list_from_translated_token(action))

    @staticmethod
    def _get_base_rotation_string_from_delta(delta_theta: float, include_small: bool):
        if delta_theta > 0:
            if delta_theta < 15:
                return "rotate_right_small" if include_small else None
            else:
                return "rotate_right"
        else:
            if delta_theta > -15:
                return "rotate_left_small" if include_small else None
            else:
                return "rotate_left"

    def state_difference_to_action(
        self,
        current_state: StretchState,
        state_difference: StretchState = None,
        final_state: StretchState = None,
    ) -> Optional[StretchAction]:
        # verify that either state_difference or final_state is not None and the other is None
        if state_difference is None and final_state is None:
            raise ValueError("Either state_difference or final_state must be provided")
        if state_difference is not None and final_state is not None:
            raise ValueError(
                "Only one of state_difference or final_state can be provided"
            )

        # create a state difference if final_state is provided
        if final_state is not None:
            state_difference = StretchState.difference(
                final_state=final_state, initial_state=current_state
            )

        change_too_small, relevant_params = StretchState.state_change_within_tolerance(
            state_difference, self.action_space_tolerance
        )

        if change_too_small:
            return None

        # Prioritize base changes
        if len(relevant_params["base_position"]) > 0:
            # get relative degrees toward delta state
            base_rotation = relative_base_degrees_toward_delta_state(
                difference_state=state_difference
            )
            # if the desired position is close and directly behind the agent, move back. Never do anything else.
            if (
                abs(base_rotation) > 178
                and abs(state_difference.base_position["z"]) < 0.25
                and (
                    "x" in relevant_params["base_position"]
                    or "z" in relevant_params["base_position"]
                )
            ):
                return self._get_singleton_action("move_back")
            if (
                abs(base_rotation)
                > self.action_space_tolerance.base_position["theta"] * 2
            ):
                # Possibility: only include small actions if the state difference x and z are together less than 0.5
                # observationally I don't like this heuristic as much
                rotation_string = self._get_base_rotation_string_from_delta(
                    base_rotation,
                    include_small=True,
                    # abs(state_difference.base_position["x"])
                    # + abs(state_difference.base_position["z"])
                    # < 0.5,
                )

                # This condition fails if a small turn would be appropriate, but we are too far away.
                # In that case, move ahead until closer or more off track.
                if rotation_string is not None:
                    return self._get_singleton_action(rotation_string)

            # if x or z difference are actionable
            if (
                "x" in relevant_params["base_position"]
                or "z" in relevant_params["base_position"]
            ):
                # at this point we can assume that the agent is sufficiently aligned with its goal
                return self._get_singleton_action("move_ahead")

            if "theta" in relevant_params["base_position"]:
                return self._get_singleton_action(
                    self._get_base_rotation_string_from_delta(
                        state_difference.base_position["theta"], include_small=True
                    )
                )

        # key assumption: the arm is aligned with the object in question. If not, base motions.
        # if y is positive, that resolves first (lifting arm is always a priority)
        # if y is negative, that resolves last  (descending arm is last in a setup)
        # z resolves inside of that (i.e. extend arm before deciding wrist rotation)
        # So overall priority: [y+, z+, wrist both directions, z-, y-]
        if (
            "y" in relevant_params["wrist_pose"]
            and state_difference.wrist_pose["y"] > 0
        ):
            return (
                self._get_singleton_action("move_arm_up")
                if state_difference.wrist_pose["y"] > 0.1
                else self._get_singleton_action("move_arm_up_small")
            )
        if (
            "z" in relevant_params["wrist_pose"]
            and state_difference.wrist_pose["z"] > 0
        ):
            return (
                self._get_singleton_action("move_arm_out")
                if state_difference.wrist_pose["z"] > 0.1
                else self._get_singleton_action("move_arm_out_small")
            )
        if "yaw" in relevant_params["wrist_pose"]:
            return (
                self._get_singleton_action("wrist_close")
                if state_difference.wrist_pose["yaw"] > 0
                else self._get_singleton_action("wrist_open")
            )

        if (
            "z" in relevant_params["wrist_pose"]
            and state_difference.wrist_pose["z"] < 0
        ):
            return (
                self._get_singleton_action("move_arm_in")
                if state_difference.wrist_pose["z"] < -0.1
                else self._get_singleton_action("move_arm_in_small")
            )
        if (
            "y" in relevant_params["wrist_pose"]
            and state_difference.wrist_pose["y"] < 0
        ):
            return (
                self._get_singleton_action("move_arm_down")
                if state_difference.wrist_pose["y"] < -0.1
                else self._get_singleton_action("move_arm_down_small")
            )
        if len(relevant_params["held_oids"]):
            if (
                len(
                    [
                        oid
                        for delta_sign, oid in state_difference.held_oids
                        if not delta_sign
                    ]
                )
                > 0
            ):
                return self._get_singleton_action(action_string=DROPOFF_ACTION)
            if (
                len(
                    [
                        oid
                        for delta_sign, oid in state_difference.held_oids
                        if delta_sign
                    ]
                )
                > 0
            ):
                return self._get_singleton_action(action_string=PICKUP_ACTION)

        # Absolute hand-pose deltas are not representable in this action space.
        raise Exception(
            "A spatial action was feasible, but the action space could not resolve it."
        )


def agent_alignment_to_point(
    current_state: StretchState, target_location: Dict[str, float], arm: bool = False
):
    transformed_target = convert_world_to_agent_coordinate(
        world_point=target_location, agent_state=current_state, arm=arm
    )
    alignment = angle_point_to_point(
        loc_start={"x": 0, "z": 0}, loc_goal=transformed_target
    )
    return alignment
