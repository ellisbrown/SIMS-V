import copy
import random
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union, Type, Any

import numpy as np
from shapely import Point

from allenact.utils.system import get_logger
from sims.tasks.abstract_task import AbstractSimsTask
from sims.tasks.abstract_task_sampler import AbstractSimsTaskSampler
from sims.utils.data_generation_utils.exception_utils import TaskSamplerException
from sims.utils.data_generation_utils.navigation_utils import (
    filtered_starting_positions,
)
from sims.utils.type_utils import (
    Vector3,
    AgentPose,
    REGISTERED_TASK_PARAMS,
    AbstractTaskArgs,
)


class BaseTaskSampler(AbstractSimsTaskSampler):
    task_type_str: Optional[str] = None
    TELEPORT_THEN_SAMPLE: bool = True
    """
    Needs to be unique for each task type, and identical to the class name specifying
    the task type specific params (in utils.type_utils' REGISTERED_TASK_PARAMS)
    """

    def __init__(
        self,
        task_args: AbstractTaskArgs,
        houses: List[Dict],
        house_inds: List[int],
        controller_args: Dict,
        controller_type: Type,
        max_tasks: Union[int, float],
        sample_per_house: int,
        prob_randomize_materials: float = 0,
        task_type: Type = AbstractSimsTask,
        device: Type = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task_args=task_args,
            houses=houses,
            house_inds=house_inds,
            controller_args=controller_args,
            controller_type=controller_type,
            max_tasks=max_tasks,
            sample_per_house=sample_per_house,
            prob_randomize_materials=prob_randomize_materials,
            task_type=task_type,
            device=device,
            **kwargs,
        )
        # get the total number of tasks assigned to this process
        self.max_tasks = max_tasks
        self.sample_per_house = sample_per_house

        self.valid_rotations = np.arange(start=0, stop=360, step=30).tolist()

        self.reset()

    @abstractmethod
    def sample_task_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self._current_tasks_left

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self._current_tasks_left

    @property
    def last_sampled_task(self) -> Optional[AbstractSimsTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.
        # Returns
        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    @property
    def current_house_index(self) -> int:
        return self.house_inds[self.house_iterator_index]

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.current_house_index]

    def increment_task_and_reset_house(
        self, force_advance_scene: bool, house_index: Optional[int] = None
    ):
        if house_index is not None:
            assert self.max_tasks == float("inf")
            self.house_iterator_index = self.house_inds.index(house_index)
            self.samples_per_current_house = 1
        else:
            self.increment_task(force_advance_scene=force_advance_scene)

        # The above code ensure that self.current_house will now be the next house
        self.reset_controller_in_current_house_and_cache_house_data(
            retain_agent_pose=False
        )

    def select_and_teleport_to_start(self, room_id_to_fsps, max_tries=1, params=None):
        # Returns starting pose if successful and if the task should be resampled (latter only used in subclasses)
        for _ in range(max_tries):
            starting_pose = AgentPose(
                position=random.choice(
                    room_id_to_fsps[random.choice(list(room_id_to_fsps.keys()))]
                ),
                rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            )
            event = self.controller.teleport_agent(**starting_pose)

            if not event:
                get_logger().warning(
                    f"Teleport failing in {self.current_house_index} at {starting_pose}"
                )
                # get_logger().warning(event)
            else:
                return starting_pose, False
        return None, False

    def increment_task(self, force_advance_scene: bool):
        if (
            not force_advance_scene
        ) and self.samples_per_current_house < self.sample_per_house:
            self.samples_per_current_house += 1
        else:
            self.house_iterator_index = (self.house_iterator_index + 1) % len(
                self.house_index_to_house
            )
            self.samples_per_current_house = 1

    def next_task(
        self,
        force_advance_scene: bool = False,
        house_index: Optional[int] = None,
    ) -> Optional[AbstractSimsTask]:
        # NOTE: Stopping condition
        if self._current_tasks_left <= 0:
            return None

        self.increment_task_and_reset_house(
            force_advance_scene=force_advance_scene, house_index=house_index
        )
        assert house_index is None or self.current_house_index == house_index

        # Figure out which reachable positions are in each room so that we can first randomly
        # sample a room and then randomly sample a spawn location within that room (makes things more uniform)
        fsps = filtered_starting_positions(self.reachable_positions)
        room_poly_map = self.controller.room_poly_map
        room_id_to_fsps = defaultdict(lambda: [])
        for fsp in fsps:
            for room_id, poly in room_poly_map.items():
                if poly.contains(Point((fsp["x"], fsp["z"]))):
                    room_id_to_fsps[room_id].append(fsp)
                    break

        starting_pose, resample_params = None, False

        if self.TELEPORT_THEN_SAMPLE:
            starting_pose, resample_params = self.select_and_teleport_to_start(
                room_id_to_fsps, params=None
            )
            if starting_pose is None:
                raise TaskSamplerException("Teleportation failed")

        # Try teleporting to up to 5 positions (possibly near objects of interest) before rejecting the house
        for resample_try in range(5):
            if resample_try == 0 or resample_params:
                sampled_parameters = self.sample_task_parameters()
                sampled_parameters["extras"] = {}

            if self.TELEPORT_THEN_SAMPLE:
                break

            starting_pose, resample_params = self.select_and_teleport_to_start(
                room_id_to_fsps, params=sampled_parameters
            )
            if starting_pose is not None:
                break

        if starting_pose is None:
            raise TaskSamplerException("Teleportation failed")

        self._current_tasks_left -= 1

        for param in REGISTERED_TASK_PARAMS[self.task_type_str]:
            assert param in sampled_parameters, (
                f"Missing {param} from {self.task_type_str} (generated {list(sampled_parameters.keys())})"
            )

        sampled_parameters.update(
            {
                "task_type": self.task_type_str,
                "num_rooms": len(
                    self.house_index_to_house[self.current_house_index]["rooms"]
                ),
                "house_index": str(self.current_house_index),
                "starting_pose": starting_pose,
            }
        )

        task_kwargs = dict(
            controller=self.controller,
            task_sampler=self,
            task_info=sampled_parameters,
        )
        self._last_sampled_task = self.task_type(
            **task_kwargs,
            **self.task_args,
            house=copy.deepcopy(self.house_index_to_house[self.current_house_index]),
        )
        return self._last_sampled_task

    def reset(self):
        self.house_iterator_index = -1  # -1 so that it doesn't skip the first task
        self.samples_per_current_house = 0
        self._current_tasks_left = self.max_tasks
        self.object_synset_counter = Counter()
