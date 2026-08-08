import copy
import time
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, final, TYPE_CHECKING, Type

from sims.environment.stretch_state import StretchState
from sims.utils.distance_calculation_utils import position_dist
from sims.utils.sel_utils import sc_metric

if TYPE_CHECKING:
    from sims.environment.stretch_controller import StretchController
    from sims.tasks.abstract_task_sampler import AbstractSimsTaskSampler

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from sims.environment.actions import StretchAction
from sims.environment.action_spaces import (
    AbstractActionSpace,
    DiscreteStretchActionSpace,
    SUB_DONE_ACTION,
    DONE_ACTION,
    PICKUP_ACTION,
    DROPOFF_ACTION,
)

from sims.utils.type_utils import RewardConfig
from sims.utils.string_utils import (
    get_natural_language_spec,
    json_templated_task_string,
)
from sims.utils.data_generation_utils.navigation_utils import get_room_id_from_location


class AbstractSimsTask(Task["StretchController"]):
    task_type_str: Optional[str] = None

    def __init__(
        self,
        controller: "StretchController",
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_space: Type[AbstractActionSpace] = None,  # DiscreteStretchActionSpace(),
        reward_config: Optional[RewardConfig] = None,
        house: Optional[Dict[str, Any]] = None,
        collect_observations: bool = True,
        task_sampler: Optional["AbstractSimsTaskSampler"] = None,
        **kwargs,
    ) -> None:
        self.collect_observations = collect_observations
        self.task_sampler = task_sampler

        super().__init__(
            env=controller,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.agent_action_space = action_space
        self.last_action_success: Union[bool, int] = -1
        self.last_action_random: Union[bool, int] = -1
        self.last_taken_action_str = ""
        self.agent_relative_goal_temp = None
        self.agent_relative_goal = None
        self.absolute_goal_temp = None
        self.absolute_goal = None
        self.last_action_time_estimate_temp = -1.0
        self.last_action_time_estimate = -1.0
        self.path: List = []
        self.travelled_distance = 0.0

        self._metrics = None
        self.observation_history = []
        self._observation_cache = None

        self.task_info["followed_path"] = [self.controller.get_current_agent_position()]
        self.task_info["agent_poses"] = [self.controller.get_current_agent_full_pose()]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []

        self.task_info["id"] = (
            self.task_info["task_type"]
            + "_"
            + str(self.task_info["house_index"])
            + "_"
            + str(int(time.time()))
            # + "_"
            # + self.task_info["natural_language_spec"].replace(" ", "")  ths gives error
        )
        if "natural_language_spec" in self.task_info:
            self.task_info["id"] += "_" + self.task_info[
                "natural_language_spec"
            ].replace(" ", "")

        assert task_info["extras"] == {}, (
            "Extra information must exist and is reserved for information collected during task"
        )

        # Set the object filter to be empty, NO OBJECTS RETURN BY DEFAULT.
        # This is all handled intuitively if you use self.controller.get_objects() when you want objects, don't do
        # controller.controller.last_event.metadata["objects"] !
        self.controller.set_object_filter([])

        self.room_poly_map = controller.room_poly_map
        self.room_type_dict = controller.room_type_dict

        self.visited_and_left_rooms = set()
        self.previous_room = None

    def is_successful(self):
        return self.successful_if_done() and self._took_end_action

    @final
    def record_observations(self):
        # This function should be called:
        # 1. Once before any step is taken and
        # 2. Once per step AFTER the step has been taken.
        # This is implemented in the `def step` function of this class

        assert (
            len(self.observation_history) == 0 and self.num_steps_taken() == 0
        ) or len(self.observation_history) == self.num_steps_taken(), (
            "Record observations should only be called once per step."
        )
        self.observation_history.append(self.get_observations())

    @property
    def action_space(self):
        return self.agent_action_space

    def reached_terminal_state(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def step_with_action_str(self, action_name: str, is_random=False):
        assert isinstance(self.agent_action_space, DiscreteStretchActionSpace), (
            "Only discrete Stretch action space is supported for raw strings."
        )
        assert action_name in self.agent_action_space.long_action_names, (
            f"Action {action_name} not in action space."
        )
        self.last_action_random = is_random

        return self.step(action_name)

    def step_with_random(self, action_call: StretchAction, is_random=False):
        """Step while recording whether the planner selected a random action."""
        self.last_action_random = is_random
        return self.step(action_call)

    def get_observation_history(self):
        return self.observation_history

    def get_current_room(self):
        agent_position = self.controller.get_current_agent_position()
        if self.room_poly_map is None:
            return "0"
        else:
            return get_room_id_from_location(self.room_poly_map, agent_position)

    def get_action_from_goal(
        self,
        absolute_goal: Union[str, StretchState],
        agent_relative_goal: StretchState = None,
        current_state: StretchState = None,
    ):
        if absolute_goal in [
            DONE_ACTION,
            SUB_DONE_ACTION,
            PICKUP_ACTION,
            DROPOFF_ACTION,
        ]:
            action = self.agent_action_space.get_action(
                action_strings=absolute_goal, action_quantities=None
            )
            self.last_action_time_estimate_temp = 1.0
            agent_relative_goal = absolute_goal
        else:
            assert agent_relative_goal is not None, (
                "Agent relative goal must be provided for spatial action."
            )
            assert current_state is not None, (
                "Current state must be provided for spatial action."
            )
            action = self.agent_action_space.state_difference_to_action(
                current_state=current_state, state_difference=agent_relative_goal
            )
            if action is not None:
                self.last_action_time_estimate_temp = action.time_estimate()
            else:
                self.last_action_time_estimate_temp = -1.0

        self.agent_relative_goal_temp = str(agent_relative_goal)
        self.absolute_goal_temp = str(absolute_goal)

        return action

    @final
    def step(self, action: Any) -> RLStepResult:
        if self.num_steps_taken() == 0:
            self.record_observations()

        if isinstance(action, int):
            # Integer actions index the discrete Stretch action list.
            action = self.agent_action_space.get_action_list()[action]

        if not isinstance(action, StretchAction):
            action = self.agent_action_space.get_action(
                action_strings=action, action_quantities=None
            )

        # put it back even if it's already a string, so it's enforced in a standard form defined by the action space
        action_str = self.agent_action_space.get_meaningful_action_string(action)

        current_room = self.get_current_room()
        if current_room != self.previous_room and current_room is not None:
            if self.previous_room is not None:
                self.visited_and_left_rooms.add(self.previous_room)
            self.previous_room = current_room

        self.controller.reset_visibility_cache()
        self._observation_cache = None

        step_result = super().step(action=action)
        self.record_observations()

        position = self.controller.get_current_agent_position()
        self.task_info["taken_actions"].append(action_str)
        self.task_info["followed_path"].append(position)
        self.task_info["agent_poses"].append(
            self.controller.get_current_agent_full_pose()
        )
        self.task_info["action_successes"].append(self.last_action_success)

        return step_result

    def _step(self, action: StretchAction) -> RLStepResult:
        action_str = self.agent_action_space.get_meaningful_action_string(action)
        self.last_taken_action_str = action_str
        self.last_action_time_estimate = copy.deepcopy(
            self.last_action_time_estimate_temp
        )
        self.agent_relative_goal = copy.deepcopy(self.agent_relative_goal_temp)
        self.absolute_goal = copy.deepcopy(self.absolute_goal_temp)

        if action.long_name == DONE_ACTION:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success

        elif action.long_name == SUB_DONE_ACTION:
            self.last_action_success = False
        else:
            event = self.controller.agent_step(action=action)
            self.last_action_success = bool(event)

            position = self.controller.get_current_agent_position()
            self.path.append(position)

            if len(self.path) > 1:
                self.travelled_distance += position_dist(
                    p0=self.path[-1], p1=self.path[-2], ignore_y=True
                )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def render(
        self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError(f"Mode '{mode}' is not supported.")

    @abstractmethod
    def successful_if_done(self, strict_success=False) -> bool:
        raise NotImplementedError

    def get_observations(self, **kwargs) -> Any:
        if self.collect_observations:
            if self._observation_cache is None:
                obs = super().get_observations()
                self._observation_cache = obs
            else:
                obs = self._observation_cache
            return obs
        return None

    def metrics(self) -> Dict[str, Any]:
        # raise NotImplementedError
        if not self.is_done():
            return {}

        metrics = super().metrics()

        metrics["success"] = self._success
        metrics["task_info"] = self.task_info

        # Note that we need to exclude the last action from the collision count, if it is the end action
        if self._took_end_action:
            action_successes = self.task_info["action_successes"][:-1]
        else:
            action_successes = self.task_info["action_successes"]
        num_collisions = action_successes.count(False)
        metrics["has_collision"] = num_collisions > 0
        metrics["collision_rate"] = (
            num_collisions / len(action_successes) if len(action_successes) > 0 else 0
        )
        metrics["sc"] = sc_metric(success=self._success, num_collisions=num_collisions)

        self._metrics = metrics

        return metrics

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        raise NotImplementedError

    def judge(self) -> float:
        """Judge the last event."""
        if self.reward_config is None:
            return 0
        raise NotImplementedError

    def to_dict(self):
        return self.task_info

    def to_string(self):
        return get_natural_language_spec(self.task_info["task_type"], self.task_info)

    def to_string_templated(self):
        return json_templated_task_string(self.task_info)

    def add_extra_task_information(self, key, value):
        assert key not in self.task_info["extras"], (
            "Key already exists in task_info['extras'], overwriting is not permitted. Addition only"
        )
        self.task_info["extras"][key] = value
