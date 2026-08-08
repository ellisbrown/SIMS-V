import time
from contextlib import contextmanager
from typing import Dict, Optional, TYPE_CHECKING
from collections import Counter

import networkx as nx
import numpy as np

from sims.utils.data_generation_utils.loc_grid_conversion import (
    grids2locs,
    upscale_grids,
    locs2grids,
)
from sims.utils.data_generation_utils.distance_transform import (
    make_distance_transform,
    make_grid_graph,
    most_central_discrete_location_pose_and_dist,
    nearest_discrete_location_loc_and_dist,
    make_discrete_path,
)
from sims.utils.data_generation_utils.bbox_utils import (
    main_box_diagonal,
    get_box_from_object,
)
from sims.utils.distance_calculation_utils import position_dist
from sims.utils.data_generation_utils.manipulation_utils import (
    plan_positions_near_object,
)

from sims.utils.data_generation_utils.exception_utils import (
    PlannerFailedToNavigateException,
)

if TYPE_CHECKING:
    from sims.tasks.abstract_task import AbstractSimsTask

PRINT_PLANNING_TIMES = False


@contextmanager
def time_block(msg):
    stime = 0
    try:
        if PRINT_PLANNING_TIMES:
            stime = time.time()
        yield None
    finally:
        if PRINT_PLANNING_TIMES:
            print(f"{time.time() - stime:.4f} s to {msg}")


class DiscretePlanner:
    def __init__(
        self,
        task: "AbstractSimsTask",
        raw_grid_size=0.1,
        grid_upscaling_factor=3,
        graph_weight_exponent=3,
        max_poses=100,
        max_distance=1.0,
        low_res_spacing_scaling=3.0,
        distance_to_obstacle_clip=1.0,
        allowable_distance_to_obstacle=0.6,
        black_list_blob_width: Optional[int] = None,
    ):
        self.task = task
        self.raw_grid_size = raw_grid_size
        self.grid_upscaling_factor = grid_upscaling_factor
        self.graph_weight_exponent = graph_weight_exponent
        self.max_poses = max_poses
        self.max_distance = max_distance
        self.low_res_spacing_scaling = low_res_spacing_scaling
        self.distance_to_obstacle_clip = distance_to_obstacle_clip
        self.allowable_distance_to_obstacle = allowable_distance_to_obstacle

        self.black_list_blob_width = black_list_blob_width or grid_upscaling_factor
        assert (
            self.black_list_blob_width % 2 != 0
            and self.black_list_blob_width <= self.grid_upscaling_factor
        ), (
            "black_list_blob_width must be an odd integer and leq to the grid_upscaling_factor"
        )

        self.black_list = []

        self._raw_locs = {}
        self.invalidate()

    def invalidate(self, *modes: str):
        if len(modes) == 0:
            modes = ["all"]

        modes = set(modes)

        if "grid" in modes or "all" in modes:
            self._valid_grid = None
            self._grid_locs = None
            self._digitizer = None
            self._low_res_locs_list = None
            self._low_res_digitizer = None
            modes.add("dist_transform")

        if "dist_transform" in modes or "all" in modes:
            self._dist_transform = None
            modes.add("graph")

        if "graph" in modes or "all" in modes:
            self._graph = None

        if "target_poses" in modes or "all" in modes:
            self.target_poses = {}

    def apply_black_list(self):
        locs = Counter(self.digitizer.row_col(loc) for loc in self.black_list)
        if len(locs) == 0:
            return

        width = self.black_list_blob_width
        hsize = (width - 1) // 2
        for loc in locs:
            raw_rmin = loc[0] - hsize
            rmin = max(raw_rmin, 0)
            rmax = min(raw_rmin + width, self.dist_transform.shape[0] - 1)

            raw_cmin = loc[1] - hsize
            cmin = max(raw_cmin, 0)
            cmax = min(raw_cmin + width, self.dist_transform.shape[1] - 1)

            # Make it lower with overlapping contributions, why not?
            self.dist_transform[rmin:rmax, cmin:cmax] *= np.power(0.5, locs[loc])

        # print(f"Black-listed locations near {locs}")

    def _set_grid_locs_digitizer(self):
        try:
            with time_block("get reachable positions"):
                orientation = (
                    int(
                        round(
                            self.task.controller.get_current_agent_full_pose()[
                                "rotation"
                            ]["y"]
                            % 360
                        )
                    )
                    // 3
                )
                # TODO take into account currently held object - also invalidate caches accordingly
                if orientation not in self._raw_locs:
                    self._raw_locs[orientation] = (
                        self.task.controller.get_reachable_positions(
                            grid_size=self.raw_grid_size
                        )
                    )
                raw_pos = self._raw_locs[orientation]

            with time_block("create raw grids"):
                raw_grid, raw_locs, raw_digitizer = locs2grids(
                    raw_pos, grid_spacing=self.raw_grid_size, return_digitizer=True
                )

                low_res_grid, low_res_locs, low_res_digitizer = locs2grids(
                    raw_pos,
                    grid_spacing=self.raw_grid_size * self.low_res_spacing_scaling,
                    return_digitizer=True,
                )
                self._low_res_locs_list = grids2locs(low_res_grid, low_res_locs)
                self._low_res_digitizer = low_res_digitizer

            with time_block("create upscaled grids"):
                self._valid_grid, self._grid_locs, self._digitizer = upscale_grids(
                    raw_grid,
                    raw_locs,
                    raw_digitizer,
                    factor=self.grid_upscaling_factor,
                    optimistic=True,
                )
        except Exception:
            raise PlannerFailedToNavigateException("Can't generate navigation grids")

    @property
    def valid_grid(self):
        if self._valid_grid is None:
            self._set_grid_locs_digitizer()
        return self._valid_grid

    @property
    def grid_locs(self):
        if self._grid_locs is None:
            self._set_grid_locs_digitizer()
        return self._grid_locs

    @property
    def digitizer(self):
        if self._digitizer is None:
            self._set_grid_locs_digitizer()
        return self._digitizer

    @property
    def dist_transform(self):
        if self._dist_transform is None:
            self._dist_transform = make_distance_transform(
                self.valid_grid,
                self.digitizer.grid_spacing,
                self.distance_to_obstacle_clip,
            )
            self.apply_black_list()
        return self._dist_transform

    @property
    def graph(self):
        if self._graph is None:
            with time_block("create grid graph"):
                self._graph = make_grid_graph(
                    self.valid_grid,
                    self.dist_transform,
                    weight_exp=self.graph_weight_exponent,
                )
        return self._graph

    @property
    def low_res_locs_list(self):
        if self._low_res_locs_list is None:
            self._set_grid_locs_digitizer()
        return self._low_res_locs_list

    @property
    def low_res_digitizer(self):
        if self._low_res_digitizer is None:
            self._set_grid_locs_digitizer()
        return self._low_res_digitizer

    def safest_loc_and_pose_for_oid(
        self, dist_transform, digitizer, object_id, self_ref
    ):
        if object_id in self.target_poses:
            pose, dist = self.target_poses[object_id]

            if pose is None:
                return None, None, None

            return self.digitizer.row_col(pose), pose, dist

        with time_block("filter poses"):
            object_loc = self.task.controller.get_obj_pos_from_obj_id(object_id)

            half_diag = (
                main_box_diagonal(
                    get_box_from_object(self.task.controller.get_object(object_id))
                )
                / 2
            )
            threshold = (
                half_diag
                + self.max_distance
                + self.low_res_spacing_scaling * self.raw_grid_size
            )

            near_positions = []
            for loc in self.low_res_locs_list:
                if position_dist(object_loc, loc, ignore_y=True) <= threshold:
                    near_positions.append(loc)

        with time_block("get touching poses"):
            poses = self.task.controller.get_touching_poses(
                object_id=object_id,
                positions=near_positions,
                max_poses=self.max_poses,
                max_distance=self.max_distance,
            )

        # TODO better for pickup? It gets us closer to the object, in a place with relatively high density of good
        #  grasping locations.
        safest_row_col, pose, pose_cost = most_central_discrete_location_pose_and_dist(
            poses, self.low_res_digitizer
        )

        # Cache the pose and cost to avoid recomputing
        self.target_poses[object_id] = (pose, pose_cost)

        if pose is None:
            return None, None, None

        return self.digitizer.row_col(pose), pose, pose_cost

    def current_row_col(self):
        current_loc = self.task.controller.get_current_agent_position()
        return self.digitizer.row_col(current_loc)

    def discrete_path_to_loc(self, oid, row, col, source_row=None, source_col=None):
        source_row_col = (
            (source_row, source_col)
            if source_row is not None
            else self.current_row_col()
        )

        if source_row_col not in self.graph or (row, col) not in self.graph:
            return None, np.inf

        with time_block("create path to goal location"):
            try:
                path, locs, path_cost = make_discrete_path(
                    self.graph,
                    *source_row_col,
                    target_row=row,
                    target_col=col,
                    distance_transform=self.dist_transform,
                    weight_exp=self.graph_weight_exponent,
                    grid_spacing=self.digitizer.grid_spacing,
                    max_distance_to_obstacle=self.allowable_distance_to_obstacle,
                )
            except nx.NetworkXNoPath:
                return None, np.inf

        # viz_path(self.dist_transform, locs, path, self, oid)

        return path, path_cost

    def discrete_to_continuous_path(self, discrete_path):
        continuous_path = []
        for p in discrete_path:
            loc = self.grid_locs[int(round(p[0])), int(round(p[1]))]
            continuous_path.append(dict(x=loc[0], y=loc[1], z=loc[2]))

        return continuous_path

    def plan_for_pickup(self, dist_transform, digitizer, object_id, self_ref):
        poses = plan_positions_near_object(self.task, object_id)
        if len(poses) == 0:
            return None, None, None
        return self.digitizer.row_col(poses[0]), poses[0], None

    def _nearest_object_id_path_dist_impl(self, object_ids, target_func):
        if len(object_ids) > 1:
            best_oid = self.task.controller.get_closest_object_from_ids(object_ids)
        else:
            best_oid = object_ids[0]

        safest_row_col, best_pose, _ = target_func(
            self.dist_transform, self.digitizer, best_oid, self
        )
        if safest_row_col is None:
            return (
                best_oid,
                self.task.controller.get_shortest_path_to_object(best_oid),
                np.inf,
            )

        # Try to get a path
        discrete_path, path_cost = self.discrete_path_to_loc(best_oid, *safest_row_col)
        if discrete_path is None:
            return (
                best_oid,
                self.task.controller.get_shortest_path_to_object(best_oid),
                np.inf,
            )

        best_path = self.discrete_to_continuous_path(discrete_path)

        return best_oid, best_path, path_cost

    def nearest_object_id_path_dist(self, object_ids):
        return self._nearest_object_id_path_dist_impl(
            object_ids, nearest_discrete_location_loc_and_dist
        )

    def nearest_object_id_path_dist_pickup(self, object_ids):
        # return self._nearest_object_id_path_dist_impl(object_ids, self.safest_loc_and_pose_for_oid)
        return self._nearest_object_id_path_dist_impl(object_ids, self.plan_for_pickup)

    def path_to_loc(self, loc: Dict[str, float], from_loc=None):
        target_row_col = self.digitizer.row_col(loc)
        if from_loc is not None:
            source_row_col = self.digitizer.row_col(from_loc)
        else:
            source_row_col = (None, None)
        discrete_path, path_cost = self.discrete_path_to_loc(
            None, *target_row_col, *source_row_col
        )
        if discrete_path is None:
            return self.task.controller.get_shortest_path_to_point(loc)
        path = self.discrete_to_continuous_path(discrete_path)
        return path

    def blacklist_or_replan_to_loc(
        self, loc, get_path=True, from_loc=None, *args, **kwargs
    ):
        if not get_path:
            self.black_list.append(self.task.controller.get_current_agent_position())
            row, col = self.digitizer.row_col(self.black_list[-1])
            # keep the current graph unless we are on a cell assumed to be valid
            if self.valid_grid[row, col]:
                self.invalidate("grid")
            return self
        else:
            try:
                return self.path_to_loc(loc, from_loc)
            except Exception:
                print("REPLANNING FAILED (edge case), returning None")
                return None

    def path_to_room(
        self,
        room_id,
        specific_agent_meshes=None,
        max_tries: int = 5,
        room_triangles=None,
    ):
        candidate_points = self.task.controller.get_candidate_points_in_room(
            room_id=room_id,
            room_triangles=room_triangles,
        )

        if len(candidate_points) == 0:
            return None

        candidate_points = [
            (loc, self.dist_transform[self.digitizer.row_col(loc)])
            for loc in candidate_points
        ]

        # Prioritize retrieved candidate points by maximum distance to obstacles
        candidate_points = [
            p[0] for p in sorted(candidate_points, key=lambda x: x[1], reverse=True)
        ]

        shortest_path = self.task.controller.get_shortest_path_to_room_candidate_points(
            candidate_points,
            specific_agent_meshes,
            max_tries,
            use_largest_possible_mesh=True,
        )

        if shortest_path is None:
            return None

        return self.path_to_loc(shortest_path[-1]) or shortest_path
