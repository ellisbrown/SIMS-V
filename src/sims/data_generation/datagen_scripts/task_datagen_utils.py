import math
import os
import sys
from typing import Literal, Dict, Optional, Union, Sequence

import canonicaljson
import prior
from allenact.base_abstractions.sensor import Sensor
from ezcolorlog import root_logger as logger

from sims.data_generation.sensors import (
    RawNavigationStretchRGBSensor,
    RawManipulationStretchRGBSensor,
)
from sims.environment.action_spaces import AbstractActionSpace
from sims.environment.manipulation_sensors import (
    AnObjectIsInHand,
    RelativeArmLocationMetadata,
)
from sims.environment.object_nav_sensors import (
    AgentsCameraParametersSensor,
    GoalAsPointInFirstFrame,
    LastActionSuccessSensor,
    LastActionIsRandomSensor,
    LastAgentLocationSensor,
    LastActionStrSensor,
    HouseNumberSensor,
    SlowAccurateObjectBBoxSensor,
    TaskRelevantPointSensor,
    TaskTemplatedTextSpecSensor,
    HypotheticalTaskSuccessSensor,
    MinimumTargetAlignmentSensor,
    Visible4mTargetCountSensor,
    MinL2TargetDistanceSensor,
    RoomsSeenSensor,
    RoomCurrentSeenSensor,
    TaskRelevantObjectBBoxSensor,
    LastGoalAgentReferenceSensor,
    LastGoalAbsoluteReferenceSensor,
    LastActionTimeEstimateSensor,
    TimeStepSensor,
    TrajectorySensor,
    ReapetGoalAsPointInFirstFrameEval,
)
from sims.video_paths import RGB_VIDEO_STEM
from sims.data_generation.paths import configured_objaverse_data_dir
from sims.utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_WIDTH,
    INTEL_CAMERA_HEIGHT,
    cropped_stretch_camera_width,
)

from sims.utils.type_utils import AbstractTaskArgs

# Memory-based worker calculation
_MEM_PER_WORKER = 15 * 1024**3


def default_workers_per_device(environ=None, platform_name=None):
    """Infer a conservative worker count without import-time side effects."""
    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name

    configured = environ.get("NUM_WORKERS_ON_SINGLE_DEVICE")
    if configured is not None:
        return max(1, int(configured))
    if platform_name != "linux":
        return 1

    assigned_memory = environ.get("ASSIGNED_MEMORY_BYTES")
    if assigned_memory is None:
        return 3
    return max(1, math.floor(int(assigned_memory) / _MEM_PER_WORKER))


_DATASET_CACHE = {}


def get_house_dataset(
    house_dataset: Literal["objaverse", "procthor"],
    max_houses_per_split: Optional[Union[int, Dict[str, int]]] = None,
):
    key = (
        house_dataset,
        canonicaljson.encode_canonical_json(max_houses_per_split).decode("utf-8"),
    )
    if key not in _DATASET_CACHE:
        if house_dataset == "objaverse":
            houses_dir = configured_objaverse_data_dir() / "houses"
            _DATASET_CACHE[key] = prior.load_dataset(
                entity="ellisbrown",
                dataset="procthor-objaverse",
                revision="251d104d900f5694ffdf2e3e868f3ed22c291a99",
                path_to_splits=None,
                split_to_path={
                    k: str(houses_dir / f"{k}.jsonl.gz")
                    for k in ["train", "val", "test"]
                },
                max_houses_per_split=max_houses_per_split,
                offline=os.environ.get("PRIOR_OFFLINE", "0") == "1",
            )
        elif house_dataset == "procthor":
            logger.info(
                f"Loading Procthor dataset ellisbrown/procthor-100k. max_houses_per_split: {max_houses_per_split}"
            )
            _DATASET_CACHE[key] = prior.load_dataset(
                entity="ellisbrown",
                dataset="procthor-100k",
                revision="dae36fb48906fdbeecfaf4360cdb4f1b2cc4cf16",
                max_houses_per_split=max_houses_per_split,
                offline=os.environ.get("PRIOR_OFFLINE", "0") == "1",
            )
        else:
            raise NotImplementedError(f"Unknown dataset {house_dataset}")
    return _DATASET_CACHE[key]


def get_core_sensors(
    include_manipulation_sensor=True,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
):
    cropped_width = cropped_stretch_camera_width(width)
    sensors = []
    sensors.append(
        RawNavigationStretchRGBSensor(
            uuid=RGB_VIDEO_STEM,
            width=cropped_width,
            height=height,
        )
    )
    if include_manipulation_sensor:
        sensors.append(
            RawManipulationStretchRGBSensor(
                uuid="raw_manipulation_camera",
                width=cropped_width,
                height=height,
            )
        )
    sensors.extend(
        [
            LastActionSuccessSensor(),
            LastActionIsRandomSensor(),
            LastAgentLocationSensor(),
            LastActionStrSensor(),
            LastGoalAgentReferenceSensor(),
            LastGoalAbsoluteReferenceSensor(),
            LastActionTimeEstimateSensor(),
            HouseNumberSensor(),
            TaskTemplatedTextSpecSensor(),
            HypotheticalTaskSuccessSensor(),
            MinimumTargetAlignmentSensor(),
            Visible4mTargetCountSensor(),
            TaskRelevantObjectBBoxSensor(
                which_camera="nav", uuid="nav_task_relevant_object_bbox"
            ),
            TaskRelevantObjectBBoxSensor(
                which_camera="manip", uuid="manip_task_relevant_object_bbox"
            ),
            SlowAccurateObjectBBoxSensor(
                which_camera="nav", uuid="nav_accurate_object_bbox"
            ),
            SlowAccurateObjectBBoxSensor(
                which_camera="manip", uuid="manip_accurate_object_bbox"
            ),
            MinL2TargetDistanceSensor(),
            RoomCurrentSeenSensor(),
            RoomsSeenSensor(),
            AnObjectIsInHand(),
            RelativeArmLocationMetadata(),
            AgentsCameraParametersSensor(),
            GoalAsPointInFirstFrame(),
            TaskRelevantPointSensor(which_camera="nav", uuid="oracle_goal_point"),
            TaskRelevantPointSensor(which_camera="manip", uuid="oracle_goal_point"),
            TimeStepSensor(uuid="time_ids", max_time_for_random_shift=0),
            TrajectorySensor(uuid="traj_index", max_idx=2048),
            ReapetGoalAsPointInFirstFrameEval(uuid="repeat_goal_in_camera_2d"),
        ]
    )
    return sensors


def get_core_task_args(
    max_steps: int,
    action_space: AbstractActionSpace,
    include_manipulation_sensor=True,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
) -> AbstractTaskArgs:
    return AbstractTaskArgs(
        sensors=get_core_sensors(
            include_manipulation_sensor=include_manipulation_sensor,
            width=width,
            height=height,
        ),
        action_space=action_space,
        max_steps=max_steps,
        reward_config=None,
    )


def add_extra_sensors_to_task_args(
    task_args: AbstractTaskArgs, extra_sensors: Optional[Sequence[Sensor]]
):
    if extra_sensors is None or len(extra_sensors) == 0:
        return

    core_sensor_dict = {x.uuid: x for x in task_args["sensors"]}

    for sensor in extra_sensors:
        if sensor.uuid in core_sensor_dict:
            print(
                f"WARNING: WE ARE REPLACING {core_sensor_dict[sensor.uuid]} WITH {sensor}"
            )
            del core_sensor_dict[sensor.uuid]

    task_args["sensors"] = list(core_sensor_dict.values()) + list(extra_sensors)


def get_walkthrough_task_args(
    max_steps: int,
    action_space: AbstractActionSpace,
    extra_sensors: Sequence[Sensor] = tuple(),
    include_manipulation_sensor=True,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
):
    task_args = get_core_task_args(
        max_steps=max_steps,
        action_space=action_space,
        include_manipulation_sensor=include_manipulation_sensor,
        width=width,
        height=height,
    )

    add_extra_sensors_to_task_args(task_args, extra_sensors)

    return task_args
