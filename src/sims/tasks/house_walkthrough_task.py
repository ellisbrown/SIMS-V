import copy
from typing import Any, Dict, List, Optional

import numpy as np
from typing_extensions import Literal
from shapely.geometry import Point

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from sims.environment.stretch_controller import StretchController
from sims.tasks.abstract_task import AbstractSimsTask
from sims.utils.distance_calculation_utils import position_dist
from sims.utils.type_utils import RewardConfig
from sims.environment.actions import StretchAction
from sims.environment.action_spaces import (
    DONE_ACTION,
    SUB_DONE_ACTION,
    AbstractActionSpace,
)


class HouseWalkthroughTask(AbstractSimsTask):
    """Visit every room and capture a panoramic scan at each room center."""

    task_type_str = "HouseWalkthrough"

    def __init__(
        self,
        controller: StretchController,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        action_space: AbstractActionSpace,
        reward_config: Optional[RewardConfig] = None,
        distance_type: Literal["l2"] = "l2",
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self.path: List = []  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0
        self.last_taken_action_str = ""
        self.last_action_success = -1
        self.last_action_random = -1

        self.reachable_positions = controller.get_reachable_positions()
        self.seen_rooms = []

        self.last_num_seen_rooms = len(self.seen_rooms)

        self.distance_type = distance_type
        self.dist_to_target_func = self.min_l2_distance_to_target

        last_distance = self.dist_to_target_func()
        self.closest_distance = last_distance
        self.optimal_distance = (
            last_distance
            if self.dist_to_target_func == self.min_geodesic_distance_to_target
            else self.min_geodesic_distance_to_target()
        )

        self.visualize = visualize

        self.num_sub_done = 0
        self.num_successful_sub_done = 0
        self._took_sub_done_action = False
        self.visited_rooms = set()
        self.visited_loc = set()

    def min_l2_distance_to_target(self):
        # Walkthroughs use the nearest unseen room as their progress signal.
        distances = self.get_room_distances()
        if len(distances) > 0:
            return min(distances)
        else:
            return 0

    def min_geodesic_distance_to_target(self):
        # Walkthroughs do not have a single target with a geodesic distance.
        return -1

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def get_agent_loc(self):
        agent_position = self.controller.get_current_agent_position()
        return round(agent_position["x"], 1), round(agent_position["z"], 1)

    def get_room_distances(self):
        agent_position = self.controller.get_current_agent_position()
        p = Point(agent_position["x"], agent_position["z"])
        distances = []
        for r, m in self.room_poly_map.items():
            if r not in self.seen_rooms:
                dis = m.distance(p)
                if dis > 0:
                    distances.append(dis)
        return distances

    def _step(self, action: StretchAction) -> RLStepResult:
        action_str = self.agent_action_space.get_meaningful_action_string(action)
        self.last_taken_action_str = action_str
        self.last_action_time_estimate = copy.deepcopy(
            self.last_action_time_estimate_temp
        )
        self.agent_relative_goal = copy.deepcopy(self.agent_relative_goal_temp)
        self.absolute_goal = copy.deepcopy(self.absolute_goal_temp)

        self._took_sub_done_action = False

        if action_str == DONE_ACTION:
            self._took_end_action = True
            self._success = self.successful_if_done()
            self.last_action_success = self._success
        elif action_str == SUB_DONE_ACTION:
            self.num_sub_done += 1
            self._took_sub_done_action = True
            if self.previous_room not in self.seen_rooms:
                self.num_successful_sub_done += 1
                self.last_action_success = True
                self.seen_rooms.append(self.previous_room)
                # refresh the closest distance for reward shaping: update it to other unexplored rooms
                self.closest_distance = self.dist_to_target_func()
            else:
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

    def successful_if_done(self, percentage_seen=None, strict_success=False) -> bool:
        return len(self.seen_rooms) == len(self.house["rooms"])

    def shaping(self) -> float:
        if self.reward_config is None:
            return 0
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if self.distance_type == "l2":
            # reward = max(self.closest_distance - cur_distance, 0)
            self.closest_distance = min(self.closest_distance, cur_distance)

        if len(self.seen_rooms) > self.last_num_seen_rooms:
            # reward += 1
            self.last_num_seen_rooms = len(self.seen_rooms)

        if self.get_agent_loc() not in self.visited_loc:
            reward -= self.reward_config.step_penalty * 2
            self.visited_loc.add(self.get_agent_loc())

        if self.get_current_room() not in self.visited_rooms:
            reward += 0.5
            self.visited_rooms.add(self.get_current_room())

        if self._took_sub_done_action:
            if self.last_action_success:
                reward += 0.5
            else:
                reward -= 0.5

        return reward * self.reward_config.shaping_weight

    def judge(self) -> float:
        """Judge the last event."""
        if self.reward_config is None:
            return 0
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = dict(
            coverage=len(self.seen_rooms) / len(self.house["rooms"]),
            distance=self.travelled_distance,
            ep_length=self.num_steps_taken(),
            total_reward=np.sum(self._rewards),
            num_seen_rooms=len(self.seen_rooms),
            num_visited_rooms=len(self.visited_rooms),
            num_visited_locations=len(self.visited_loc),
            success=self._success,
            num_sub_done=self.num_sub_done,
            sub_done_acc=(
                self.num_successful_sub_done / self.num_sub_done
                if self.num_sub_done > 0
                else 0.0
            ),
        )
        self._metrics = metrics
        return metrics
