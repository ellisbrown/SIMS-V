from typing import Tuple

import numpy as np

BBOX_DIST_THRESHOLD = 0.1


def get_box_from_object(obj, verbose=False):
    if obj.get("objectOrientedBoundingBox") is not None:
        box = obj["objectOrientedBoundingBox"]["cornerPoints"]
    else:
        if verbose:
            print(
                f"Using axisAlignedBoundingBox for {obj['objectId']} ({obj['synset']})"
            )
        box = obj["axisAlignedBoundingBox"]["cornerPoints"]

    return np.array(box)


# lightly adapted from https://github.com/allenai/ai2thor-rearrangement
def get_basis_for_3d_box_from_bbox_corners(
    bbox_corners,
) -> Tuple[np.ndarray, np.ndarray]:
    # stacked along columns
    without_first = first_corner_to_other_vertices_vectors(bbox_corners)
    magnitudes1 = vector_lengths(without_first)
    v0_ind = np.argmin(magnitudes1)
    v0_mag = magnitudes1[v0_ind]

    if v0_mag < 1e-8:
        raise RuntimeError(f"Could not find basis for {bbox_corners}")

    v0 = without_first[np.argmin(magnitudes1)] / v0_mag

    orth_to_v0 = (v0.reshape(1, -1) * without_first).sum(-1) < v0_mag / 2.0
    inds_orth_to_v0 = np.where(orth_to_v0)[0]
    v1_ind = inds_orth_to_v0[np.argmin(magnitudes1[inds_orth_to_v0])]
    v1_mag = magnitudes1[v1_ind]
    v1 = without_first[v1_ind, :] / magnitudes1[v1_ind]

    orth_to_v1 = (v1.reshape(1, -1) * without_first).sum(-1) < v1_mag / 2.0
    inds_orth_to_v0_and_v1 = np.where(orth_to_v0 & orth_to_v1)[0]

    if len(inds_orth_to_v0_and_v1) != 1:
        raise RuntimeError(f"Could not find basis for {bbox_corners}")

    v2_ind = inds_orth_to_v0_and_v1[0]
    v2 = without_first[v2_ind, :] / magnitudes1[v2_ind]

    orth_mat = np.stack((v0, v1, v2), axis=1)  # Orthonormal matrix, stacked by columns!

    return orth_mat, magnitudes1[[v0_ind, v1_ind, v2_ind]]


def get_basis_for_3d_box(obj) -> Tuple[np.ndarray, np.ndarray]:
    # should get object aligned but might return axis aligned
    bbox_corners = get_box_from_object(obj, verbose=False)

    return get_basis_for_3d_box_from_bbox_corners(bbox_corners)


def get_min_object_height(obj):
    box = get_box_from_object(obj)
    if len(box.shape) == 0:
        return None
    y_list = [xyz[1] for xyz in box]
    return min(y_list)


# Adapted from Chat-GPT, AST (Axis Separation Theorem), often used for collision detection:
def compute_bbox_distance(box1, box2):
    # Each box is described as 8 vertices
    box1 = np.array(box1)
    box2 = np.array(box2)

    # Compute the (up to) 15 potential separating axes:
    # 3 normal directions per box (2x3=6)
    # cross products of the 3 edge directions of one box against those from the other box (3x3=9)
    axes = get_separating_axes(box1, box2)

    # Initialize max distance to -1
    max_distance = -1

    # Iterate over each potential separating axis
    for axis in axes:
        # Project vertices onto the separating axis
        min_proj1, max_proj1 = vertex_projection_extrema(box1, axis)
        min_proj2, max_proj2 = vertex_projection_extrema(box2, axis)

        # Check for overlap between projections
        overlap = max_proj1 >= min_proj2 and max_proj2 >= min_proj1

        if not overlap:
            # The boxes are separated along this axis
            # Compute the distance between the axis and the boxes' projections
            distance = min(abs(max_proj1 - min_proj2), abs(max_proj2 - min_proj1))
            max_distance = max(max_distance, distance)

    if max_distance < 0:
        # The boxes intersect, distance is zero
        return 0.0

    return max_distance


def get_separating_axes(box1, box2):
    # Compute the (up to) 15 potential separating axes

    # Add the cross products of edges
    edges1, _ = get_basis_for_3d_box_from_bbox_corners(box1)
    edges2, _ = get_basis_for_3d_box_from_bbox_corners(box2)
    edges1 = edges1.transpose()
    edges2 = edges2.transpose()

    axes = []
    for edge1 in edges1:
        for edge2 in edges2:
            cross_product = np.cross(edge1, edge2)
            nrm = np.linalg.norm(cross_product)
            if nrm > 1e-3:  # Consider only non-parallel edges
                axes.append(cross_product / nrm)

    return np.concatenate(
        [
            np.stack(axes, axis=0),
            get_box_normals_from_edges(edges1),
            get_box_normals_from_edges(edges2),
        ],
        axis=0,
    )


def get_box_normals_from_edges(edges):
    # note that sign of vectors is random
    normals = np.stack(
        [
            np.cross(edges[0], edges[1]),
            np.cross(edges[1], edges[2]),
            np.cross(edges[2], edges[0]),
        ],
        axis=0,
    )
    return normals / np.linalg.norm(
        normals, axis=1, keepdims=True
    )  # renormalize, just in case


def vertex_projection_extrema(vertices, axis):
    projections = np.dot(vertices, axis)

    return np.min(projections), np.max(projections)


def first_corner_to_other_vertices_vectors(box):
    corner_to_others = box - box[:1, :]
    assert np.isclose(corner_to_others[0].sum(), 0.0)
    without_corner = corner_to_others[1:]
    return without_corner


def vector_lengths(vectors):
    return np.sqrt((vectors * vectors).sum(1))


def main_box_diagonal(box):
    return max(vector_lengths(first_corner_to_other_vertices_vectors(box)))
