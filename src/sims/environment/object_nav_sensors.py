import copy
import json
import random
from typing import TYPE_CHECKING, Any, Dict, Literal, Union

import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super

from sims.environment.stretch_controller import StretchController
from sims.utils.pointing_utils import (
    DUMMY_POINT_VALUE,
    convert_point_from_3d_world_to_2d_camera,
    generate_goal_as_point_only_first,
    get_most_centered_point_from_possible_points,
)
from sims.utils.string_utils import (
    convert_string_to_byte,
    json_templated_task_string,
)
from sims.utils.constants.stretch_initialization_utils import (
    STRETCH_CAMERA_HORIZONTAL_CROP,
    cropped_stretch_camera_width,
)
from sims.utils.type_utils import get_task_relevant_synsets

if TYPE_CHECKING:
    from sims.tasks.abstract_task import AbstractSimsTask
else:
    from typing import TypeVar

    AbstractSimsTask = TypeVar("AbstractSimsTask")


class LastActionSuccessSensor(Sensor):
    def __init__(self, uuid: str = "last_action_success") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array([task.last_action_success], dtype=np.int64)


class LastActionIsRandomSensor(Sensor):
    def __init__(self, uuid: str = "last_action_is_random") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array([task.last_action_random], dtype=np.int64)


class LastAgentLocationSensor(Sensor):
    def __init__(self, uuid: str = "last_agent_location") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        agent_position_rotation = env.get_current_agent_full_pose()
        agent_position = agent_position_rotation["position"]
        agent_rotation = agent_position_rotation["rotation"]

        return np.array(
            [
                agent_position["x"],
                agent_position["y"],
                agent_position["z"],
                agent_rotation["x"],
                agent_rotation["y"],
                agent_rotation["z"],
            ],
            dtype=np.float64,
        )


class TaskTemplatedTextSpecSensor(Sensor):
    def __init__(
        self,
        uuid: str = "templated_task_spec",
        str_max_len: Union[str, int] = "adaptive",
        is_constant_across_episode: bool = True,
    ) -> None:
        assert isinstance(str_max_len, int) or str_max_len == "adaptive"
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        self.is_constant_across_episode = is_constant_across_episode
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.MultiDiscrete:
        if self.str_max_len == "adaptive":
            return gym.spaces.MultiDiscrete([256] * 1)
        else:
            return gym.spaces.MultiDiscrete([256] * self.str_max_len)

    @staticmethod
    def encode_observation(task_info, str_max_len: Union[str, int]):
        task_string = json_templated_task_string(task_info)
        if str_max_len == "adaptive":
            bytes = convert_string_to_byte(task_string, 2 * len(task_string))
            final_index = len(bytes) + 1
            for ind in reversed(range(len(bytes))):
                if bytes[ind] == 0:
                    final_index = ind
            return bytes[:final_index]
        elif isinstance(str_max_len, int):
            return convert_string_to_byte(task_string, str_max_len)
        else:
            raise NotImplementedError

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return self.encode_observation(task.task_info, self.str_max_len)


def round_floats_in_dict(target_dict, decimal_digits=4):
    if target_dict is None or not isinstance(target_dict, dict):
        return target_dict
    target_dict = copy.deepcopy(target_dict)
    result_dict = {}
    for key, value in target_dict.items():
        if isinstance(value, float):
            formatted_float = round(value, decimal_digits)
            result_dict[key] = formatted_float
        elif isinstance(value, dict):
            result_dict[key] = round_floats_in_dict(value, decimal_digits)
        else:
            result_dict[key] = value
    return result_dict


class AgentsCameraParametersSensor(Sensor):
    def __init__(
        self,
        uuid: str = "agent_camera_params",
        str_max_len: Union[str, int] = 1000,
    ) -> None:
        assert isinstance(str_max_len, int)
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.MultiDiscrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        agent_parameter_sensors = task.controller.get_all_camera_parameters()
        agent_parameter_sensors = round_floats_in_dict(agent_parameter_sensors)
        param_string = json.dumps(agent_parameter_sensors)
        return convert_string_to_byte(param_string, self.str_max_len)


class HypotheticalTaskSuccessSensor(Sensor):
    def __init__(self, uuid: str = "hypothetical_task_success") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array([task.successful_if_done(strict_success=True)], dtype=np.int64)


class MinimumTargetAlignmentSensor(Sensor):
    def __init__(self, uuid: str = "minimum_visible_target_alignment") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if "synsets" not in task.task_info:
            return np.array([-1], dtype=np.float64)
        object_type = task.task_info["synsets"][0]
        visible_target_objects = [
            obj
            for obj in task.task_info["synset_to_object_ids"][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera="nav", maximum_distance=2
            )
        ]
        visible_target_alignments = [
            abs(task.controller.get_agent_alignment_to_object(obj))
            for obj in visible_target_objects
        ]
        if len(visible_target_alignments) == 0:
            return np.array([-1], dtype=np.float64)
        else:
            return np.array([min(visible_target_alignments)], dtype=np.float64)


class Visible4mTargetCountSensor(Sensor):
    def __init__(self, uuid: str = "visible_target_4m_count") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if "synsets" not in task.task_info:
            return np.array([0], dtype=np.float64)
        object_type = task.task_info["synsets"][0]
        visible_target_objects = [
            obj
            for obj in task.task_info["synset_to_object_ids"][object_type]
            if task.controller.object_is_visible_in_camera(
                obj, which_camera="nav", maximum_distance=4
            )
        ]
        return np.array([len(visible_target_objects)], dtype=np.int64)


class TaskRelevantObjectBBoxSensor(Sensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal["nav", "manip"] = "nav",
        uuid: str = "task_relevant_object_bbox",
    ) -> None:
        self.convert_to_pixel_coords = convert_to_pixel_coords
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def _get_observation_space(self) -> gym.spaces.Dict:
        # The `10` number below is a lie, this will be the length of the number of target ids.
        if self.convert_to_pixel_coords:
            return gym.spaces.Dict(
                spaces={
                    "oids_as_bytes": gym.spaces.MultiDiscrete([256] * 10),
                    "synset_to_oids_as_bytes": gym.spaces.MultiDiscrete([256] * 10),
                    "min_rows": gym.spaces.Box(
                        low=-1, high=np.inf, shape=(10,), dtype=np.float32
                    ),
                    "max_rows": gym.spaces.Box(
                        low=-1, high=np.inf, shape=(10,), dtype=np.float32
                    ),
                    "min_cols": gym.spaces.Box(
                        low=-1, high=np.inf, shape=(10,), dtype=np.float32
                    ),
                    "max_cols": gym.spaces.Box(
                        low=-1, high=np.inf, shape=(10,), dtype=np.float32
                    ),
                },
            )
        else:
            return gym.spaces.Dict(
                spaces={
                    "oids_as_bytes": gym.spaces.MultiDiscrete([256] * 10),
                    "synset_to_oids_as_bytes": gym.spaces.MultiDiscrete([256] * 10),
                    "min_xs": gym.spaces.Box(
                        low=-1, high=1, shape=(10,), dtype=np.float32
                    ),
                    "max_xs": gym.spaces.Box(
                        low=-1, high=1, shape=(10,), dtype=np.float32
                    ),
                    "min_ys": gym.spaces.Box(
                        low=-1, high=1, shape=(10,), dtype=np.float32
                    ),
                    "max_ys": gym.spaces.Box(
                        low=-1, high=1, shape=(10,), dtype=np.float32
                    ),
                },
            )

    @staticmethod
    def encode_json(json_serializable: Any):
        oids_json = json.dumps(json_serializable)

        bytes = convert_string_to_byte(oids_json, 2 * len(oids_json))
        final_index = len(bytes) + 1
        for ind in reversed(range(len(bytes))):
            if bytes[ind] == 0:
                final_index = ind
        return bytes[:final_index]

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if task.num_steps_taken() == 0:
            if hasattr(task, "task_relevant_synset_to_oids"):
                self.task_relevant_synset_to_objects = task.task_relevant_synset_to_oids
                self.task_relevant_oids = task.task_relevant_oids
                self.oids_as_bytes = task.oids_as_bytes
                self.synset_to_oids_as_bytes = task.synset_to_oids_as_bytes
            else:
                task_relevant_synsets = get_task_relevant_synsets(
                    task_spec=task.task_info
                )
                all_objects = env.get_objects()

                self.task_relevant_synset_to_objects = {}
                for synset in task_relevant_synsets:
                    self.task_relevant_synset_to_objects[synset] = (
                        env.get_all_objects_of_synset(
                            synset=synset, include_hyponyms=True, all_objs=all_objects
                        )
                    )

                self.task_relevant_oids = list(
                    sorted(
                        set(
                            o["objectId"]
                            for objs in self.task_relevant_synset_to_objects.values()
                            for o in objs
                        )
                    )
                )

                task_relevant_synset_to_oids = {
                    synset: [o["objectId"] for o in objs]
                    for synset, objs in self.task_relevant_synset_to_objects.items()
                }

                self.oids_as_bytes = self.encode_json(self.task_relevant_oids)
                self.synset_to_oids_as_bytes = self.encode_json(
                    task_relevant_synset_to_oids
                )

                task.task_relevant_synset_to_oids = self.task_relevant_synset_to_objects
                task.task_relevant_oids = self.task_relevant_oids
                task.oids_as_bytes = self.oids_as_bytes
                task.synset_to_oids_as_bytes = self.synset_to_oids_as_bytes

        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        for oid in self.task_relevant_oids:
            min_x, min_y, max_x, max_y = -1, -1, -1, -1

            if task.controller.object_is_visible_in_camera(
                oid, which_camera=self.which_camera, maximum_distance=4
            ):
                points = env.get_approx_object_mask(
                    object_id=oid, which_camera=self.which_camera, divisions=7
                )

                if points is not None and len(points) != 0:
                    xs = [max(min(p["x"], 1), 0) for p in points]
                    ys = [max(min(p["y"], 1), 0) for p in points]
                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)

            min_xs.append(min_x)
            max_xs.append(max_x)
            min_ys.append(min_y)
            max_ys.append(max_y)

        if self.convert_to_pixel_coords:
            # Useful function for displaying bounding boxes on an image using below output
            # def display_bbox_from_minmax_rows_cols():
            #     import copy, cv2
            #
            #     frame = copy.deepcopy(env.navigation_camera if self.which_camera == "nav" else env.manipulation_camera)
            #     for col_min, row_min, col_max, row_max, oid in zip(min_cols, min_rows, max_cols, max_rows, self.task_relevant_oids):
            #         if col_min == -1:
            #             continue
            #         h, w = frame.shape[:2]
            #
            #         start_point = (int(col_min), int(row_min))
            #         end_point = (int(col_max), int(row_max))
            #         print(start_point, end_point)
            #         cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=1)
            #
            #         cv2.putText(
            #             frame,
            #             oid,
            #             (int(col_min), int(row_min - 10)),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.6,
            #             color=(255, 255, 255),
            #             thickness=2
            #         )
            #
            #     cv2.imwrite(f"example_with_bounding_boxes_{self.which_camera}.jpg", frame)

            h, w = env.controller.last_event.frame.shape[:2]
            doesnt_have_value = ~np.array([min_x != -1 for min_x in min_xs], dtype=bool)
            min_cols = w * np.array(min_xs, dtype=np.float32)
            max_cols = w * np.array(max_xs, dtype=np.float32)
            max_rows = h * (1 - np.array(min_ys, dtype=np.float32))
            min_rows = h * (1 - np.array(max_ys, dtype=np.float32))

            hs, ws = env.navigation_camera.shape[:2]
            # print(f"h, w = {h}, {w}")
            # print(f"hs, ws = {hs}, {ws}")

            expected_width = cropped_stretch_camera_width(w)
            assert ws == expected_width and hs == h, (
                f"Expected ws={expected_width}, hs={h}, but got ws={ws}, hs={hs}"
            )

            min_cols = np.clip(
                min_cols - STRETCH_CAMERA_HORIZONTAL_CROP,
                a_min=0,
                a_max=ws - 1,
            )
            max_cols = np.clip(
                max_cols - STRETCH_CAMERA_HORIZONTAL_CROP,
                a_min=0,
                a_max=ws - 1,
            )

            min_cols[doesnt_have_value] = -1
            max_cols[doesnt_have_value] = -1
            max_rows[doesnt_have_value] = -1
            min_rows[doesnt_have_value] = -1
            return {
                "oids_as_bytes": self.oids_as_bytes,
                "synset_to_oids_as_bytes": self.synset_to_oids_as_bytes,
                "min_cols": min_cols.astype(int).astype(np.float32),
                "max_cols": max_cols.astype(int).astype(np.float32),
                "max_rows": max_rows.astype(int).astype(np.float32),
                "min_rows": min_rows.astype(int).astype(np.float32),
            }

        else:
            # Useful function for displaying bounding boxes on an image using below output
            # def display_bbox_from_minmax_xy():
            #     import copy
            #
            #     import cv2
            #
            #     frame = copy.deepcopy(env.navigation_camera if self.which_camera == "nav" else env.manipulation_camera)
            #     for x_min, y_min, x_max, y_max, oid in zip(min_xs, min_ys, max_xs, max_ys, self.task_relevant_oids):
            #         h, w = frame.shape[:2]
            #
            #         x_min = w * x_min
            #         y_min = h * (1 - y_min)
            #         x_max = w * x_max
            #         y_max = h * (1 - y_max)
            #
            #         start_point = (int(x_min), int(y_min))
            #         end_point = (int(x_max), int(y_max))
            #         cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=1)
            #
            #         cv2.putText(
            #             frame,
            #             oid,
            #             (int(x_min), int(y_min) - 10),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale=0.6,
            #             color=(255, 255, 255),
            #             thickness=2
            #         )
            #
            #     cv2.imwrite(f"example_with_bounding_boxes_{self.which_camera}.jpg", frame[:,:,::-1])
            return {
                "oids_as_bytes": self.oids_as_bytes,
                "min_xs": np.array(min_xs, dtype=np.float32)
                .astype(int)
                .astype(np.float32),
                "max_xs": np.array(max_xs, dtype=np.float32)
                .astype(int)
                .astype(np.float32),
                "min_ys": np.array(min_ys, dtype=np.float32)
                .astype(int)
                .astype(np.float32),
                "max_ys": np.array(max_ys, dtype=np.float32)
                .astype(int)
                .astype(np.float32),
            }


class SlowAccurateObjectBBoxSensor(TaskRelevantObjectBBoxSensor):
    def __init__(
        self,
        convert_to_pixel_coords: bool = True,
        which_camera: Literal["nav", "manip"] = "nav",
        uuid: str = "accurate_object_bbox",
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.convert_to_pixel_coords = convert_to_pixel_coords
        observation_space = self._get_observation_space()
        assert convert_to_pixel_coords

        self.which_camera = which_camera

        self.task_relevant_oids = []
        self.task_relevant_synset_to_objects = {}
        self.oids_as_bytes = None
        self.synset_to_oids_as_bytes = None

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if task.num_steps_taken() == 0:
            if hasattr(task, "task_relevant_synset_to_oids"):
                self.task_relevant_synset_to_objects = task.task_relevant_synset_to_oids
                self.task_relevant_oids = task.task_relevant_oids
                self.oids_as_bytes = task.oids_as_bytes
                self.synset_to_oids_as_bytes = task.synset_to_oids_as_bytes
            else:
                task_relevant_synsets = get_task_relevant_synsets(
                    task_spec=task.task_info
                )
                all_objects = env.get_objects()

                self.task_relevant_synset_to_objects = {}
                for synset in task_relevant_synsets:
                    self.task_relevant_synset_to_objects[synset] = (
                        env.get_all_objects_of_synset(
                            synset=synset, include_hyponyms=True, all_objs=all_objects
                        )
                    )

                self.task_relevant_oids = list(
                    sorted(
                        set(
                            o["objectId"]
                            for objs in self.task_relevant_synset_to_objects.values()
                            for o in objs
                        )
                    )
                )

                task_relevant_synset_to_oids = {
                    synset: [o["objectId"] for o in objs]
                    for synset, objs in self.task_relevant_synset_to_objects.items()
                }

                self.oids_as_bytes = self.encode_json(self.task_relevant_oids)
                self.synset_to_oids_as_bytes = self.encode_json(
                    task_relevant_synset_to_oids
                )

                task.task_relevant_synset_to_oids = self.task_relevant_synset_to_objects
                task.task_relevant_oids = self.task_relevant_oids
                task.oids_as_bytes = self.oids_as_bytes
                task.synset_to_oids_as_bytes = self.synset_to_oids_as_bytes

        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        for oid in self.task_relevant_oids:
            min_x, min_y, max_x, max_y = -1, -1, -1, -1
            segm_mask = task.controller.get_segmentation_mask_of_object(
                oid, which_camera=self.which_camera
            )
            if np.any(segm_mask):
                min_x = np.min(np.where(segm_mask)[1])
                min_y = np.min(np.where(segm_mask)[0])
                max_x = np.max(np.where(segm_mask)[1])
                max_y = np.max(np.where(segm_mask)[0])
            min_xs.append(min_x)
            max_xs.append(max_x)
            min_ys.append(min_y)
            max_ys.append(max_y)

        min_cols = np.array(min_xs, dtype=np.float32).astype(int).astype(np.float32)
        max_cols = np.array(max_xs, dtype=np.float32).astype(int).astype(np.float32)
        max_rows = np.array(max_ys, dtype=np.float32).astype(int).astype(np.float32)
        min_rows = np.array(min_ys, dtype=np.float32).astype(int).astype(np.float32)

        return {
            "oids_as_bytes": self.oids_as_bytes,
            "synset_to_oids_as_bytes": self.synset_to_oids_as_bytes,
            "min_cols": min_cols,
            "max_cols": max_cols,
            "max_rows": max_rows,
            "min_rows": min_rows,
        }


class MinL2TargetDistanceSensor(Sensor):
    def __init__(self, uuid: str = "minimum_l2_target_distance") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if not hasattr(task, "min_l2_distance_to_target"):
            return np.array([-1], dtype=np.float64)
        return np.array([task.min_l2_distance_to_target()], dtype=np.float64)


class LastActionStrSensor(Sensor):
    def __init__(self, uuid: str = "last_action_str", str_max_len=200) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return convert_string_to_byte(task.last_taken_action_str, self.str_max_len)


class LastGoalAgentReferenceSensor(Sensor):
    def __init__(self, uuid: str = "last_goal_agent_frame", str_max_len=500) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        relative_goal = round_floats_in_dict(task.agent_relative_goal)
        return convert_string_to_byte(relative_goal, self.str_max_len)


class LastGoalAbsoluteReferenceSensor(Sensor):
    def __init__(self, uuid: str = "last_goal_absolute_frame", str_max_len=500) -> None:
        self.str_max_len = str_max_len
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.str_max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        absolute_goal = round_floats_in_dict(task.absolute_goal)
        return convert_string_to_byte(absolute_goal, self.str_max_len)


class LastActionTimeEstimateSensor(Sensor):
    def __init__(self, uuid: str = "last_action_time_estimate") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(task.last_action_time_estimate, dtype=np.float64)


class HouseNumberSensor(Sensor):
    def __init__(self, uuid: str = "house_index") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(task.task_info["house_index"]))


class RoomsSeenSensor(Sensor):
    def __init__(self, uuid: str = "rooms_seen") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(int(len(task.visited_and_left_rooms)))


class GoalAsPointInFirstFrame(Sensor):
    def __init__(self, uuid: str = "goal_in_camera_2d_first_step") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if task.num_steps_taken() == 0:
            return generate_goal_as_point_only_first(
                task.task_info, is_it_first_step=True
            )
        else:
            return generate_goal_as_point_only_first(
                task.task_info, is_it_first_step=False
            )


class ReapetGoalAsPointInFirstFrameEval(Sensor):
    def __init__(self, uuid: str = "repeat_goal_in_camera_2d") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return generate_goal_as_point_only_first(task.task_info, is_it_first_step=True)


class TaskRelevantPointSensor(Sensor):
    def __init__(
        self,
        uuid: str = "oracle_goal_point",
        which_camera: Literal["nav", "manip"] = "nav",
        maximum_num_points=1,
    ) -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.uuid = f"{self.uuid}_{which_camera}"
        self.which_camera = which_camera
        self.maximum_num_points = maximum_num_points

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1000)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        #  the only tasks that will have this sensor are  GoToPointOnFloor, gonearobject and pickupobject, it should be dummy value for the rest
        if "goal_in_world_3d" in task.task_info:  # GoToPointOnFloor
            goal_point = task.task_info["goal_in_world_3d"]
            goal_point = np.array(
                [goal_point["x"], env.get_floor_level(), goal_point["z"]]
            )
            # Add a dimension to the goal point similar to batching
            goal_point = goal_point[np.newaxis, :]

            assert goal_point.shape == (1, 3)

            point_in_camera, is_this_valid = convert_point_from_3d_world_to_2d_camera(
                goal_point, env, which_camera=self.which_camera
            )

            if is_this_valid.item():
                return point_in_camera[0]
            else:
                return DUMMY_POINT_VALUE

        elif (
            "target_obj_in_3d" in task.task_info
        ):  # GoNearPointOnObject and PickupPointOnObject
            goal_obj = task.task_info["object_id"]
            mask = env.get_segmentation_mask_of_object(
                object_id=goal_obj, which_camera=self.which_camera
            )
            if mask.sum() == 0:
                return DUMMY_POINT_VALUE
            all_possible_points = np.nonzero(mask)
            possible_point_array = np.stack(
                [all_possible_points[1], all_possible_points[0]], axis=0
            ).T
            assert (
                possible_point_array.shape[1] == 2
                and len(possible_point_array.shape) == 2
            )
            # This will be a list of points at most self.maximum_num_points
            center_point = get_most_centered_point_from_possible_points(
                possible_point_array, self.maximum_num_points
            )

            return center_point

        else:
            return DUMMY_POINT_VALUE


class RoomCurrentSeenSensor(Sensor):
    def __init__(self, uuid: str = "room_current_seen") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        return np.array(bool(task.get_current_room() in task.visited_and_left_rooms))


class TimeStepSensor(Sensor):
    def __init__(self, uuid: str = "time_ids", max_time_for_random_shift=0) -> None:
        observation_space = self._get_observation_space()
        self.max_time_for_random_shift = max_time_for_random_shift
        self.random_start = 0
        super().__init__(**prepare_locals_for_super(locals()))
        self._update = False

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def sample_random_start(self):
        self.random_start = random.randint(0, max(self.max_time_for_random_shift, 0))

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        steps = task.num_steps_taken()
        if self._update:
            steps += 1
        else:
            self._update = True
        if task.is_done():  # not increment at next episode start
            self._update = False
            self.sample_random_start()
        return np.array(self.random_start + int(steps), dtype=np.int64)


class TrajectorySensor(Sensor):
    def __init__(self, uuid: str = "traj_index", max_idx: int = 4) -> None:
        observation_space = self._get_observation_space()
        self.curr_idx = 0
        self.max_idx = max_idx
        self._update = False
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if self._update:
            self.curr_idx += 1
            if self.curr_idx >= self.max_idx:
                self.curr_idx = 0
            self._update = False
        if task.is_done():  # update at next episode start
            self._update = True
        return np.array(self.curr_idx, dtype=np.int64)
