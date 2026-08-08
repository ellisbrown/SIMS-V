import abc
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import networkx as nx
import numpy as np
from shapely import Point

from sims.environment.action_spaces import DONE_ACTION, SUB_DONE_ACTION
from sims.environment.stretch_state import StretchState
from sims.planning.discrete_planner import DiscretePlanner
from sims.tasks.abstract_task import AbstractSimsTask
from sims.tasks.house_walkthrough_task import HouseWalkthroughTask
from sims.utils.data_generation_utils.exception_utils import (
    ExplorationCoverageException,
    PlannerFailedToNavigateException,
    PlannerFailedToSeeException,
)
from sims.utils.data_generation_utils.navigation_utils import (
    WallSearch,
    build_layout_graph_matrix,
    get_nearest_positions,
    get_room_id_from_location,
    panoramic_scan,
    walk_on_path,
)

if TYPE_CHECKING:
    from sims.environment.stretch_controller import StretchController


class PathPlanner(abc.ABC):
    task_type_str: Optional[List[str]] = None

    @abc.abstractmethod
    def plan(self, task: AbstractSimsTask) -> Optional[List[Dict[str, Any]]]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_planner_guaranteed_to_fail(self, task: AbstractSimsTask) -> bool:
        return False


class HouseWalkthroughPlanner(PathPlanner):
    """Plan the room-center tour used to generate the SIMS videos."""

    task_type_str = ["HouseWalkthrough"]
    task: HouseWalkthroughTask
    controller: "StretchController"
    observations: List[Any]
    wall_search: Optional[WallSearch]
    dp: Optional[DiscretePlanner]

    def __init__(self):
        self.wall_search = None
        self.dp = None
        self.use_astar = True

    def reset(self, task: HouseWalkthroughTask):
        self.task = task
        self.controller = task.controller
        self.observations = []

        self.wall_search = WallSearch(
            task=task,
            max_wall_distance=15.0,
            wall_alignment_threshold=85,
            use_all_cameras=False,
            grid_size=0.25,
        )

        if self.use_astar:
            self.dp = DiscretePlanner(task)

    def make_walk_kwargs(self):
        if self.use_astar:
            return {
                "replan_func": lambda task,
                *args,
                **kwargs: self.dp.blacklist_or_replan_to_loc(*args, **kwargs)
            }
        return {}

    def is_planner_guaranteed_to_fail(self, task: AbstractSimsTask) -> bool:
        return False

    def get_shortest_path_to_room(
        self, task: HouseWalkthroughTask, room_id: str
    ) -> Optional[Sequence[Dict[str, float]]]:
        room_centroid = {
            "x": task.room_poly_map[room_id].centroid.x,
            "y": 0,
            "z": task.room_poly_map[room_id].centroid.y,
        }
        nearest_reachable = [
            position
            for position in get_nearest_positions(task, room_centroid)
            if get_room_id_from_location(task.room_poly_map, position) == room_id
        ]

        for room_center_ish in nearest_reachable:
            path_to_room = task.controller.get_shortest_path_to_point(
                target_position=room_center_ish
            )
            if path_to_room is not None:
                return path_to_room
        return None

    def navigate_to_room(
        self, task: HouseWalkthroughTask, room_id: str, stop_for: Optional[List] = None
    ) -> bool:
        agent_current_room = get_room_id_from_location(
            task.room_poly_map, StretchState(self.controller).base_position
        )
        if agent_current_room == room_id:
            return True

        if self.use_astar:
            path_to_room = self.dp.path_to_room(room_id)
        else:
            path_to_room = self.get_shortest_path_to_room(task=task, room_id=room_id)

        nav_success = False
        if path_to_room is not None:
            nav_success = walk_on_path(
                path_to_room,
                task,
                max_tries=5,
                stop_for=stop_for,
                successful_stop_callback=self.wall_search.seen_room_walls_in_step,
                **self.make_walk_kwargs(),
            )
            if not nav_success:
                agent_position = task.controller.get_current_agent_position()
                nav_success = task.room_poly_map[room_id].contains(
                    Point(agent_position["x"], agent_position["z"])
                )
        return nav_success

    def get_dfs_visition_order(
        self,
        task: HouseWalkthroughTask,
        room_id_list: List[str],
        layout_graph_matrix,
    ) -> List[str]:
        source_id = task.get_current_room()
        if len(room_id_list) == 1:
            return room_id_list
        adjacency_matrix = np.ceil(layout_graph_matrix.numpy())
        graph = nx.from_numpy_array(adjacency_matrix)
        preorder = list(
            nx.dfs_preorder_nodes(graph, source=room_id_list.index(source_id))
        )
        return [source_id] + [
            room_id_list[index]
            for index in preorder
            if room_id_list[index] != source_id
        ]

    def plan(self, task: HouseWalkthroughTask):
        self.reset(task)
        layout_graph_matrix, room_id_list = build_layout_graph_matrix(
            task.house, task.room_poly_map
        )
        room_visit_order = self.get_dfs_visition_order(
            task, room_id_list, layout_graph_matrix
        )

        for room_id in room_visit_order:
            if room_id in task.seen_rooms:
                continue

            if "num_rooms_in_house" not in task.task_info:
                self.wall_search.targeting_room = room_id
                self.wall_search.search_walls(task, self)

            self.navigate_to_room(task=task, room_id=room_id)

            if task.get_current_room() == room_id:
                panoramic_scan(task)
                task.step_with_random(task.get_action_from_goal(SUB_DONE_ACTION))
            else:
                raise PlannerFailedToNavigateException(
                    f"Could not navigate to room {room_id}"
                )

            if (
                "num_rooms_in_house" not in task.task_info
                and len(self.wall_search.missing_walls) > 0
            ):
                print(
                    f"{len(self.wall_search.missing_walls)} wall(s) not visited in room {room_id}"
                )

        task.add_extra_task_information(
            "natural_language_description", task.to_string()
        )
        task.step_with_random(task.get_action_from_goal(DONE_ACTION))

        if not task.is_successful():
            raise PlannerFailedToSeeException("Task unsuccessful after issuing done.")

        if task.successful_if_done():
            return task.get_observation_history()

        error = f"Failure with {len(task.seen_rooms)} of {len(task.house['rooms'])} rooms seen"
        raise ExplorationCoverageException(error)
