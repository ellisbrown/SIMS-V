import json
from typing import Optional, Dict, Union, Any, List


class AbstractAction:
    def __init__(
        self,
        short_name: Optional[str] = None,
        long_name: Optional[str] = None,
    ):
        self.short_name = short_name
        self.long_name = long_name
        self.unit_info = {}

    def enact(self):
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError


class StretchAction(AbstractAction):
    def __init__(
        self,
        base: Optional[Dict[str, Union[int, float]]] = None,
        arm: Optional[Dict[str, Union[int, float]]] = None,
        wrist: Optional[Dict[str, Union[int, float]]] = None,
        gripper: Optional[Dict[str, Union[int, float]]] = None,
        absolute_arm_and_wrist_motion: bool = False,
        long_name: Optional[str] = None,
        short_name: Optional[str] = None,
    ):
        super().__init__(long_name=long_name, short_name=short_name)
        base_defaults = {"z": None, "theta": None}
        arm_defaults = {"y": None, "z": None}
        wrist_defaults = {"yaw": None}
        gripper_defaults = {"gripper_openness": None}

        self._base = {**base_defaults, **(base or {})}
        self._arm = {**arm_defaults, **(arm or {})}
        self._wrist = {**wrist_defaults, **(wrist or {})}
        self._gripper = {**gripper_defaults, **(gripper or {})}
        self.absolute_arm_and_wrist_motion = absolute_arm_and_wrist_motion

        self.unit_info = {
            "base": {
                "z": "meters",
                "theta": "degrees",
            },
            "arm": {
                "y": "meters",
                "z": "meters",
            },
            "wrist": {
                "yaw": "degrees",
            },
            "gripper": {
                "gripper_openness": "percentage",
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        def filter_none_values(d):
            if isinstance(d, dict):
                return {k: filter_none_values(v) for k, v in d.items() if v is not None}
            else:
                return d

        full_dict = {
            "base": self._base,
            "arm": self._arm,
            "wrist": self._wrist,
            "gripper": self._gripper,
        }
        final_filter = {k: v for k, v in filter_none_values(full_dict).items() if v}
        return final_filter

    def to_string(self) -> str:
        if self.long_name:
            return self.long_name
        return json.dumps(self.to_dict())

    def __str__(self) -> str:
        return self.to_string()

    def time_estimate(self) -> float:
        base_mps_estimate = 0.25  # NB: quite slow, but TODO check
        base_dps_estimate = 45
        arm_mps_estimate = 0.25
        wrist_dps_estimate = 45
        gripper_percent_per_second = 15

        base_time = (
            abs(self._base["z"] or 0) / base_mps_estimate
            + abs(self._base["theta"] or 0) / base_dps_estimate
        )
        arm_time = (
            abs(self._arm["y"] or 0) / arm_mps_estimate
            + abs(self._arm["z"] or 0) / arm_mps_estimate
        )
        wrist_time = abs(self._wrist["yaw"] or 0) / wrist_dps_estimate
        gripper_time = (
            abs(self._gripper["gripper_openness"] or 0) / gripper_percent_per_second
        )

        # Return the maximum estimated time
        return max(base_time, arm_time, wrist_time, gripper_time)

    @classmethod
    def init_from_singletons(
        cls, singleton_actions: List["StretchAction"]
    ) -> "StretchAction":
        # Initialize empty dictionaries for each component
        base = {"z": None, "theta": None}
        arm = {"y": None, "z": None}
        wrist = {"yaw": None}
        gripper = {"gripper_openness": None}

        for action in singleton_actions:
            if action._base["z"] is not None:
                assert base["z"] is None, "Duplicate base z action"
                base["z"] = action._base["z"]
            if action._base["theta"] is not None:
                assert base["theta"] is None, "Duplicate base theta action"
                base["theta"] = action._base["theta"]
            if action._arm["y"] is not None:
                assert arm["y"] is None, "Duplicate arm y action"
                arm["y"] = action._arm["y"]
            if action._arm["z"] is not None:
                assert arm["z"] is None, "Duplicate arm z action"
                arm["z"] = action._arm["z"]
            if action._wrist["yaw"] is not None:
                assert wrist["yaw"] is None, "Duplicate wrist yaw action"
                wrist["yaw"] = action._wrist["yaw"]
            if action._gripper["gripper_openness"] is not None:
                assert gripper["gripper_openness"] is None, "Duplicate gripper action"
                gripper["gripper_openness"] = action._gripper["gripper_openness"]

        # Create a new instance of the StretchAction class with the combined actions
        return cls(base=base, arm=arm, wrist=wrist, gripper=gripper)

    def enact(self):
        if self.absolute_arm_and_wrist_motion:
            raise NotImplementedError(
                "Absolute arm and wrist motion not yet implemented."
            )

        flattened_action_components = []
        if self._base["z"] is not None:
            flattened_action_components.append(
                {"action": "MoveAgent", "ahead": self._base["z"]}
            )
        if self._base["theta"] is not None:
            flattened_action_components.append(
                {"action": "RotateAgent", "degrees": self._base["theta"]}
            )
        if self._arm["y"] or self._arm["z"]:
            flattened_action_components.append(
                {
                    "action": "MoveArmRelative",
                    "offset": {
                        "x": 0,
                        "y": self._arm["y"] or 0,
                        "z": self._arm["z"] or 0,
                    },
                }
            )
        if self._wrist["yaw"]:
            flattened_action_components.append(
                {"action": "RotateWristRelative", "yaw": self._wrist["yaw"]}
            )
        if self._gripper["gripper_openness"]:
            raise NotImplementedError("Gripper openness not yet implemented.")
            # flattened_action_components.append(
            #     {"action": "SetGripper", "gripper_openness": self._gripper["gripper_openness"]}
            # )
        return flattened_action_components


class StretchGraspAction(StretchAction):
    def enact(self):
        return [{"action": "PickupObject"}]


class StretchDropOffAction(StretchAction):
    def enact(self):
        return [{"action": "ReleaseObject"}]


class StretchActionReal(StretchAction):
    def enact(self):
        if self.absolute_arm_and_wrist_motion:
            raise NotImplementedError(
                "Absolute arm and wrist motion not yet implemented."
            )

        flattened_action_components = []
        if self._base["z"] is not None:
            flattened_action_components.append(
                {"action": "MoveAgent", "ahead": self._base["z"]}
            )
        if self._base["theta"] is not None:
            flattened_action_components.append(
                {"action": "RotateAgent", "degrees": self._base["theta"]}
            )
        if self._arm["y"]:
            flattened_action_components.append(
                {
                    "action": "MoveArmBase",
                    "args": {"move_scalar": self._arm["y"]},
                }
            )
        if self._arm["z"]:
            flattened_action_components.append(
                {
                    "action": "MoveArmExtension",
                    "args": {"move_scalar": self._arm["z"]},
                }
            )
        if self._wrist["yaw"]:
            flattened_action_components.append(
                {"action": "RotateWrist", "args": {"move_scalar": self._wrist["yaw"]}}
            )
        return flattened_action_components


class StretchGraspActionReal:
    # TODO
    pass


class StretchDropOffActionReal:
    # TODO
    pass
