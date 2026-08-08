from typing import Tuple
from itertools import chain

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from skimage.draw import line

from sims.utils.distance_calculation_utils import position_dist


def safest_discrete_location_pose_and_dist(poses, dt, digitizer):
    # TODO safest distance is a cheap but poor proxy for avoiding obstacles when moving the arm, but it might
    #  allow for at least orienting the agent and wrist with low collision rate (besides allowing for object
    #  visibility when the object is very low)
    safest_row_col = None
    safest_pose = None
    safest_distance = 0
    for pose in poses:
        row, col = digitizer.row_col(pose)
        if dt[row, col] > safest_distance:
            safest_distance = dt[row, col]
            safest_row_col = (row, col)
            safest_pose = pose
    return safest_row_col, safest_pose, safest_distance


def nearest_safest_discrete_location_pose_and_dist(
    poses, dt, digitizer, object_id, planner
):
    shortest_path = planner.task.controller.get_shortest_path_to_object(object_id)

    if shortest_path is None:
        return None, None, None

    final_loc = shortest_path[-1]
    safest_row_col = None
    safest_pose = None
    safest_distance = 0
    shortest_to_loc = np.inf
    for pose in poses:
        row, col = digitizer.row_col(pose)
        cur_dist = position_dist(pose["position"], final_loc, ignore_y=True)
        if cur_dist < shortest_to_loc:  # and dt[row, col] >= safest_distance:
            safest_distance = dt[row, col]
            safest_row_col = (row, col)
            safest_pose = pose
            shortest_to_loc = cur_dist
    return safest_row_col, safest_pose, safest_distance


def nearest_discrete_location_loc_and_dist(dt, digitizer, object_id, planner):
    shortest_path = planner.task.controller.get_shortest_path_to_object(object_id)

    if shortest_path is None:
        return None, None, 0

    final_loc = shortest_path[-1]
    row, col = digitizer.row_col(final_loc)
    safest_distance = dt[row, col]
    safest_row_col = (row, col)
    return safest_row_col, final_loc, safest_distance


def most_central_discrete_location_pose_and_dist(poses, digitizer):
    if len(poses) == 0:
        return None, None, None

    discrete_to_pose = {}
    max_row, max_col = -np.inf, -np.inf
    for pose in poses:
        row, col = digitizer.row_col(pose)
        discrete_to_pose[(row, col)] = pose
        max_row = max(max_row, row)
        max_col = max(max_col, col)

    if len(discrete_to_pose) == 1:
        return list(discrete_to_pose.keys())[0], poses[0], None

    im = np.zeros((max_row + 1, max_col + 1), dtype=bool)
    for row, col in discrete_to_pose.keys():
        im[row, col] = True

    upfactor = 3
    imup = cv2.resize(
        im.astype(np.uint8),
        None,
        fx=upfactor,
        fy=upfactor,
        interpolation=cv2.INTER_NEAREST,
    )
    imdist = make_distance_transform(imup)
    gc = np.unravel_index(np.argmax(imdist), imdist.shape)
    best_dist = imdist.max()
    gc = (int(round(gc[0] / upfactor)), int(round(gc[1] / upfactor)))
    if gc not in discrete_to_pose:
        all_discrete = list(discrete_to_pose.keys())
        nearest_gc = all_discrete.pop()
        nearest_dist = (gc[0] - nearest_gc[0]) ** 2 + (gc[1] - nearest_gc[1]) ** 2
        for other_gc in all_discrete:
            other_dist = (gc[0] - other_gc[0]) ** 2 + (gc[1] - other_gc[1]) ** 2
            if other_dist < nearest_dist:
                nearest_dist = other_dist
                nearest_gc = other_gc
        gc = nearest_gc

    # viz = im.astype(np.uint8)
    # viz[gc[0], gc[1]] = 2
    # plt.imshow(viz)
    # plt.title(f"{gc} {discrete_to_pose[gc]} {best_dist}")
    # plt.show()

    return gc, discrete_to_pose[gc], best_dist


def make_distance_transform(grid, grid_spacing=None, max_distance_to_obstacle=0.75):
    pad = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=grid.dtype)
    pad[1:-1, 1:-1] = grid
    dt = distance_transform_edt(pad)[1:-1, 1:-1]
    if grid_spacing is None:
        return dt
    assert grid_spacing >= 0, (
        f"grid_spacing should be positive, if given (got {grid_spacing})"
    )
    return np.clip(dt, 0.0, max_distance_to_obstacle / grid_spacing)


def cost_function(x, weight_exp, zeroish=1e-30):
    return np.power(np.maximum(x, zeroish), -weight_exp)


def get_pixel_cost(cost, p):
    return cost[round(p[0]), round(p[1])]


def get_segment_cost(cost, p1, p2):
    seg_cost = cost[line(round(p1[0]), round(p1[1]), round(p2[0]), round(p2[1]))]
    return seg_cost.sum(), seg_cost.max()


def make_grid_graph(
    grid,
    dt,
    weight_exp=2,
):
    def make_direction_edges(s0: Tuple[slice, slice], s1: Tuple[slice, slice]):
        locs = np.nonzero(grid[s0] * grid[s1])
        ws = np.maximum(cost[s0], cost[s1])[locs]
        labs0 = labels[s0][locs]
        labs1 = labels[s1][locs]
        return map(
            lambda labsw: (tuple(labsw[0]), tuple(labsw[1]), labsw[2]),
            zip(labs0, labs1, ws),
        )

    cost = cost_function(dt, weight_exp)
    labels = np.stack(
        np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij"), axis=2
    )
    edges = chain(
        make_direction_edges(
            (slice(None, -1), slice(None)), (slice(1, None), slice(None))
        ),
        make_direction_edges(
            (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        ),
    )
    g = nx.Graph()
    g.add_weighted_edges_from(edges)

    return g


def simplify_path_greedy(
    waypoints, dt, weight_exp, grid_spacing, max_distance_to_obstacle
):
    if len(waypoints) == 0:
        return [], np.inf

    cost = cost_function(dt, weight_exp)
    if weight_exp == 0:
        # Make obstacles effectively non-visitable
        cost[dt == 0] = 1e30

    if len(waypoints) == 1:
        return [waypoints[0]], get_pixel_cost(cost, waypoints[0])

    if len(waypoints) == 2:
        return list(waypoints), get_segment_cost(cost, waypoints[0], waypoints[1])

    # If shortcutting a section of the path doesn't increase the cost, that's what we should do!
    #  obviously, 4-connectivity won't bring us there (8-connectivity should? - but expensive)

    acceptable_cost = cost_function(max_distance_to_obstacle / grid_spacing, weight_exp)

    res = [waypoints[0]]
    path_cost = 0.0

    def add_to_path():
        ret = cur_cost - get_pixel_cost(cost, res[-1])
        res.append(cur_next)
        return ret

    cur_next = waypoints[1]
    cur_cost, cur_peak = get_segment_cost(cost, res[-1], cur_next)
    for it in range(2, len(waypoints)):
        pos_next = waypoints[it]

        # We add extra cost to each last step in each segment, which is incorrect, but works okay
        pos_cost, pos_peak = get_segment_cost(cost, res[-1], pos_next)
        pos_local_cost, local_peak = get_segment_cost(cost, cur_next, pos_next)

        if pos_cost <= cur_cost + pos_local_cost and (
            pos_peak <= max(cur_peak, local_peak) or pos_peak < acceptable_cost
        ):
            cur_cost = pos_cost
            cur_peak = pos_peak
        else:
            path_cost += add_to_path()

            cur_cost = pos_local_cost
            cur_peak = local_peak

        cur_next = pos_next

    path_cost += add_to_path()

    return res, path_cost


def make_discrete_path(
    graph,
    source_row,
    source_col,
    target_row,
    target_col,
    distance_transform,
    weight_exp,
    grid_spacing,
    max_distance_to_obstacle,
):
    locs = nx.astar_path(graph, (source_row, source_col), (target_row, target_col))
    waypoints, path_cost = simplify_path_greedy(
        locs, distance_transform, weight_exp, grid_spacing, max_distance_to_obstacle
    )
    # print(f"{len(locs)} locs to {len(waypoints)} waypoints")
    return waypoints, locs, path_cost


def viz_path(
    dt,
    locs,
    waypoints,
    planner,
    object_id,
    raw_path_color=(1.0, 0.0, 0.5),
    shortest_path_color=(0.0, 0.5, 1.0),
    simplified_path_color=(0.0, 1.0, 0.5),
    waypoints_color=(1.0, 0.5, 0.0),
    shortest_path=None,
):
    path = dt.copy()
    path[path == 0] = np.max(path) + 1
    path = path / np.max(path)
    path = np.stack([path, path, path], axis=2)
    for p in locs:
        path[p[0], p[1], :] = raw_path_color

    if object_id is not None:
        shortest_path = planner.task.controller.get_shortest_path_to_object(object_id)
        if shortest_path is not None:
            for source, goal in zip(shortest_path[:-1], shortest_path[1:]):
                src = planner.digitizer.row_col(source)
                dst = planner.digitizer.row_col(goal)
                cv2.line(
                    path,
                    src[::-1],
                    dst[::-1],
                    color=shortest_path_color,
                )
        else:
            print("no shortest path via meshnav")

    if shortest_path is not None:
        for source, goal in zip(shortest_path[:-1], shortest_path[1:]):
            src = planner.digitizer.row_col(source)
            dst = planner.digitizer.row_col(goal)
            cv2.line(
                path,
                src[::-1],
                dst[::-1],
                color=shortest_path_color,
            )

    for src, dst in zip(waypoints[:-1], waypoints[1:]):
        cv2.line(
            path,
            (round(src[1]), round(src[0])),
            (round(dst[1]), round(dst[0])),
            color=simplified_path_color,
        )

    for dst in waypoints:
        path[round(dst[0]), round(dst[1]), :] = waypoints_color

    plt.imshow(path)
    plt.show()
