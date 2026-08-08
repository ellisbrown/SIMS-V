import random
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np

from sims.qa import question_templates as qt
from sims.qa import vsi_question_templates as vsi_qt
from sims.qa.generators.common import (
    VSI_OBJECTS_CLEAN,
    clean_obj_type,
    gen_distance_options,
    gen_mc_question,
    generate_qa_batch,
)


#  >> SPATIAL << #


def calc_3d_bbox_distance_between_objects(
    bbox1: Dict[str, Any], bbox2: Dict[str, Any]
) -> float:
    """
    Calculate the minimum distance between two 3D bounding boxes.

    Args:
        bbox1: Dictionary containing 'min' and 'max' coordinates of first bbox
        bbox2: Dictionary containing 'min' and 'max' coordinates of second bbox

    Returns:
        float: Minimum distance between the two bounding boxes
    """
    # Convert min/max points to numpy arrays for easier computation
    min1, max1 = np.array(bbox1["min"]), np.array(bbox1["max"])
    min2, max2 = np.array(bbox2["min"]), np.array(bbox2["max"])

    # Check for overlap in each dimension
    def get_axis_distance(min1, max1, min2, max2):
        if max1 < min2:
            return min2 - max1
        elif max2 < min1:
            return min1 - max2
        return 0.0

    # Calculate distance for each axis
    dx = get_axis_distance(min1[0], max1[0], min2[0], max2[0])
    dy = get_axis_distance(min1[1], max1[1], min2[1], max2[1])
    dz = get_axis_distance(min1[2], max1[2], min2[2], max2[2])

    # Return Euclidean distance
    return np.sqrt(dx**2 + dy**2 + dz**2)


def gen_obj_abs_dist_qa(
    preprocessed_data: Dict[str, Any],
    ambiguity_threshold: float = 0.15,
    digits: int = 1,
    vsi: bool = False,
    minimal: bool = False,
) -> Dict[str, Any]:
    """
    Generate questions about absolute distances between objects using closest points.

    Args:
        preprocessed_data: Dictionary containing spatial metadata and object information
        ambiguity_threshold: Distance threshold (in meters) below which answers are considered ambiguous.
            Default is 0.15 (15cm)
        digits: Number of decimal places to round distances to
        vsi: Whether to generate VSI questions
        minimal: Whether to only generate questions for VSI objects

    Returns:
        Dictionary containing the generated question and related information

    Raises:
        ValueError: If there aren't enough objects or if distances are ambiguous
    """
    spatial_metadata = preprocessed_data["spatial_metadata"]
    salient_cts = spatial_metadata["salient_type_counts"]
    salient_bboxes = spatial_metadata["salient_object_bbox"]

    if minimal:
        salient_cts = {
            obj_type: ct
            for obj_type, ct in salient_cts.items()
            if clean_obj_type(obj_type) in VSI_OBJECTS_CLEAN
        }

    # Get object types seen exactly once to avoid ambiguity
    objs_seen_once = [obj_type for obj_type, ct in salient_cts.items() if ct == 1]
    all_objects = list(salient_cts.keys())

    if len(objs_seen_once) == 0:
        raise ValueError("No objects seen once to generate question")

    if len(all_objects) < 2:
        raise ValueError(
            f"Not enough objects to generate question. Got {len(all_objects)}"
        )

    # Select reference object and other objects
    ref_obj = random.choice(objs_seen_once)
    all_objects.remove(ref_obj)  # Remove reference object from choices
    other_obj = random.choice(all_objects)

    # clean formatting of object names
    ref_obj_clean = clean_obj_type(ref_obj)
    other_obj_clean = clean_obj_type(other_obj)

    # Generate question text
    if vsi:
        task_name = "vsi_obj_abs_distance"
        question = vsi_qt.OBJ_ABS_DISTANCE_TEMPLATE.format(
            object1=ref_obj_clean, object2=other_obj_clean
        )
    else:
        task_name = "obj_abs_distance"
        question = qt.OBJ_ABS_DISTANCE_TEMPLATE.format(
            ref_obj=ref_obj_clean, other_obj=other_obj_clean
        )
    if minimal:
        task_name += "_minimal"

    # Calculate minimum distances between reference object and each choice
    ref_bbox = salient_bboxes[ref_obj][0]  # We know there's exactly one

    # Find minimum distance among all instances of this object type
    obj_min_dist = float("inf")
    obj_bboxes = salient_bboxes[other_obj]
    all_dists = []
    for obj_bbox in obj_bboxes:
        dist = calc_3d_bbox_distance_between_objects(ref_bbox, obj_bbox)
        obj_min_dist = min(obj_min_dist, dist)
        all_dists.append(dist)

    # Check for ambiguous answers
    if obj_min_dist <= ambiguity_threshold:
        raise ValueError(f"Ambiguous distance between objects: {obj_min_dist}")

    gt = round(obj_min_dist, digits)

    # MCQ choices
    options = gen_distance_options(gt)
    mc_question, mc_answer, choices = gen_mc_question(
        question, gt, options, shuffle_options=True
    )

    return {
        "task": task_name,
        "question": question,
        "gt_answer": gt,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": choices,
        "ref_obj": ref_obj_clean,
        "other_obj": other_obj_clean,
        "n_other_objs": len(obj_bboxes),
        "other_dists": all_dists,
    }


def gen_obj_abs_dist_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    ambiguity_threshold: float = 0.15,
    max_retries: int = 10,
    vsi: bool = False,
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    """Generate multiple relative distance QAs."""
    return generate_qa_batch(
        qa_generator=gen_obj_abs_dist_qa,
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        ambiguity_threshold=ambiguity_threshold,
        vsi=vsi,
        minimal=minimal,
    )


def gen_obj_rel_dist_qa(
    preprocessed_data: Dict[str, Any],
    ambiguity_threshold: float = 0.15,
    vsi: bool = False,
    minimal: bool = False,
) -> Dict[str, Any]:
    """
    Generate questions about relative distances between objects using closest points.

    Args:
        preprocessed_data: Dictionary containing spatial metadata and object information
        ambiguity_threshold: Distance threshold (in meters) below which answers are considered ambiguous.
            Default is 0.15 (15cm)
        vsi: Whether to generate VSI questions
        minimal: Whether to only generate questions for VSI objects

    Returns:
        Dictionary containing the generated question and related information

    Raises:
        ValueError: If there aren't enough objects or if distances are ambiguous
    """
    spatial_metadata = preprocessed_data["spatial_metadata"]
    salient_cts = spatial_metadata["salient_type_counts"]
    salient_bboxes = spatial_metadata["salient_object_bbox"]

    if minimal:
        salient_cts = {
            obj_type: ct
            for obj_type, ct in salient_cts.items()
            if clean_obj_type(obj_type) in VSI_OBJECTS_CLEAN
        }

    # Get object types seen exactly once to avoid ambiguity
    objs_seen_once = [obj_type for obj_type, ct in salient_cts.items() if ct == 1]
    all_objects = list(salient_cts.keys())

    if len(objs_seen_once) == 0:
        raise ValueError("No objects seen once to generate question")

    if len(all_objects) < 5:
        raise ValueError(
            f"Not enough objects to generate question. Got {len(all_objects)}"
        )

    # Select reference object and other objects
    ref_obj = random.choice(objs_seen_once)
    all_objects.remove(ref_obj)  # Remove reference object from choices
    other_objs = random.sample(all_objects, 4)

    # clean formatting of object names
    ref_obj_clean = clean_obj_type(ref_obj)
    other_objs_clean = [clean_obj_type(obj) for obj in other_objs]

    # Generate question text
    if vsi:
        task_name = "vsi_obj_rel_distance"
        question = vsi_qt.OBJ_REL_DISTANCE_TEMPLATE.format(
            category=ref_obj_clean,
            choice_a=other_objs_clean[0],
            choice_b=other_objs_clean[1],
            choice_c=other_objs_clean[2],
            choice_d=other_objs_clean[3],
        )
    else:
        task_name = "obj_rel_distance"
        question = qt.OBJ_REL_DISTANCE_TEMPLATE.format(
            type=ref_obj_clean,
            a=other_objs_clean[0],
            b=other_objs_clean[1],
            c=other_objs_clean[2],
            d=other_objs_clean[3],
        )

    if minimal:
        task_name += "_minimal"

    # Calculate minimum distances between reference object and each choice
    ref_bbox = salient_bboxes[ref_obj][0]  # We know there's exactly one
    min_dists = []

    for obj in other_objs:
        obj_bboxes = salient_bboxes[obj]
        # Find minimum distance among all instances of this object type
        obj_min_dist = float("inf")
        for obj_bbox in obj_bboxes:
            dist = calc_3d_bbox_distance_between_objects(ref_bbox, obj_bbox)
            obj_min_dist = min(obj_min_dist, dist)
        min_dists.append(obj_min_dist)

    # Check for ambiguous answers
    sorted_dists = sorted(min_dists)
    if (
        len(sorted_dists) >= 2
        and (sorted_dists[1] - sorted_dists[0]) <= ambiguity_threshold
    ):
        raise ValueError("Ambiguous distances between objects")

    # Get the closest object
    closest_obj = other_objs_clean[np.argmin(min_dists)]

    # Generate multiple choice question
    mc_question, mc_answer, choices = gen_mc_question(
        question, closest_obj, other_objs_clean, shuffle_options=True
    )

    return {
        "task": task_name,
        "question": question,
        "gt_answer": closest_obj,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": choices,
        "ref_obj": ref_obj_clean,
        "distances": min_dists,
    }


def gen_obj_rel_dist_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    ambiguity_threshold: float = 0.15,
    max_retries: int = 10,
    vsi: bool = False,
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    """Generate multiple relative distance QAs."""
    return generate_qa_batch(
        qa_generator=gen_obj_rel_dist_qa,
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        ambiguity_threshold=ambiguity_threshold,
        vsi=vsi,
        minimal=minimal,
    )


def calculate_relative_direction(
    positioning_obj: Dict[str, Any],
    orienting_obj: Dict[str, Any],
    querying_obj: Dict[str, Any],
    ambiguity_threshold: float = 10.0,
    difficulty: str = "easy",
) -> Tuple[str, float, List[str]]:
    """
    Calculate if querying_obj is to the left/right/back or in a specific quadrant relative to orienting_obj
    from positioning_obj's perspective.
    Assumes coordinates are in (X, Z, Y) order where Z is vertical.

    Args:
        positioning_obj: Dict containing bbox info of object we're positioned at
        orienting_obj: Dict containing bbox info of object we're facing
        querying_obj: Dict containing bbox info of object whose relative position we're determining
        ambiguity_threshold: Angle threshold in degrees for ambiguous cases
        difficulty: Difficulty level of the question:
            - "easy": 2 options, left/right
            - "medium": 3 options, left/right/back (an object is "back" if the turning angle is at least 135°)
            - "hard": 4 options, front-left/front-right/back-left/back-right

    Returns:
        Tuple of (direction, angle, options) where direction is:
            - For "easy": "left", "right", or "ambiguous"
            - For "medium": "left", "right", "back", or "ambiguous"
            - For "hard": "front-left", "front-right", "back-left", "back-right", or "ambiguous"
    """
    # Get centroids - note the (X, Z, Y) ordering.
    pos_point = np.array(positioning_obj["centroid"])
    ori_point = np.array(orienting_obj["centroid"])
    query_point = np.array(querying_obj["centroid"])

    # Vector from our position to the object we're facing (forward direction)
    forward_vec = ori_point - pos_point

    # Vector from our position to the querying object
    query_vec = query_point - pos_point

    # Project vectors to horizontal plane (X, Y) by taking the first and third coordinates.
    forward_vec_2d = np.array([forward_vec[0], forward_vec[2]])
    query_vec_2d = np.array([query_vec[0], query_vec[2]])

    # Calculate the angle between the two vectors
    dot_product = np.dot(forward_vec_2d, query_vec_2d)
    forward_mag = np.linalg.norm(forward_vec_2d)
    query_mag = np.linalg.norm(query_vec_2d)

    if forward_mag == 0 or query_mag == 0:
        return "ambiguous", 0, []

    cos_angle = dot_product / (forward_mag * query_mag)
    # Clamp to avoid numerical issues
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = np.degrees(np.arccos(cos_angle))  # in [0, 180]

    # Determine side using the cross product.
    # In the (X,Y) plane, a positive cross product indicates a counter-clockwise rotation (left)
    cross_product = np.cross(forward_vec_2d, query_vec_2d)

    if difficulty == "easy":
        # Ambiguous if nearly straight ahead or behind
        options = ["left", "right"]
        if abs(angle) < ambiguity_threshold or abs(angle - 180) < ambiguity_threshold:
            return "ambiguous", angle, options
        if cross_product > 0:
            return "left", angle, options
        elif cross_product < 0:
            return "right", angle, options
        else:
            return "ambiguous", angle, options

    elif difficulty == "medium":
        # For medium: if turning angle is less than 135°, decide left/right; otherwise, classify as "back".
        # Also consider near-boundary cases as ambiguous.
        options = ["left", "right", "back"]
        if (
            abs(angle) < ambiguity_threshold
            or abs(angle - 180) < ambiguity_threshold
            or abs(angle - 135) < ambiguity_threshold
        ):
            return "ambiguous", angle, options
        if angle < 135 and cross_product > 0:
            return "left", angle, options
        elif angle < 135 and cross_product < 0:
            return "right", angle, options
        elif angle >= 135:
            return "back", angle, options
        else:
            raise ValueError(f"Invalid angle/cross product: {angle}, {cross_product}")

    elif difficulty == "hard":
        # For hard: subdivide into quadrants.
        # Ambiguous if near 0°, 90°, or 180° boundaries.
        options = ["front-left", "front-right", "back-left", "back-right"]
        if (
            abs(angle) < ambiguity_threshold
            or abs(angle - 90) < ambiguity_threshold
            or abs(angle - 180) < ambiguity_threshold
        ):
            return "ambiguous", angle, options
        if 0 < angle < 90 and cross_product > 0:
            return "front-left", angle, options
        elif 0 < angle < 90 and cross_product < 0:
            return "front-right", angle, options
        elif 90 < angle < 180 and cross_product > 0:
            return "back-left", angle, options
        elif 90 < angle < 180 and cross_product < 0:
            return "back-right", angle, options
        else:
            raise ValueError(f"Invalid angle/cross product: {angle}, {cross_product}")

    else:
        raise ValueError(f"Invalid difficulty level: {difficulty}")


def gen_obj_rel_dir_qa(
    preprocessed_data: Dict[str, Any],
    ambiguity_threshold: float = 10.0,
    max_distance: float = 10.0,
    version: str = "v1",
    minimal: bool = False,
) -> Dict[str, Any]:
    """
    Generate a question about relative object directions.

    Args:
        preprocessed_data: Dictionary containing scene data
        ambiguity_threshold: Angle threshold for ambiguous cases in degrees
        max_distance: Maximum allowed distance between any pair of objects in meters
        version: Which version of the question to generate. Options are "v1", "vsi_easy", "vsi_medium", "vsi_hard"
        minimal: Whether to only generate questions for VSI objects

    Returns:
        Dictionary containing question data and answer

    Raises:
        ValueError: If objects are too far apart or spatial relationship is ambiguous
    """
    spatial_metadata = preprocessed_data["spatial_metadata"]
    salient_cts = spatial_metadata["salient_type_counts"]
    salient_bboxes = spatial_metadata["salient_object_bbox"]

    if minimal:
        salient_cts = {
            obj_type: ct
            for obj_type, ct in salient_cts.items()
            if clean_obj_type(obj_type) in VSI_OBJECTS_CLEAN
        }

    # Get objects that are unique
    single_objects = [obj for obj, count in salient_cts.items() if count == 1]

    if len(single_objects) < 3:
        raise ValueError(
            "Need at least 3 unique objects to generate direction question"
        )

    # Randomly select 3 objects for the question
    selected_objects = random.sample(single_objects, 3)
    positioning_obj, orienting_obj, querying_obj = selected_objects

    # Get their bounding boxes
    pos_bbox = salient_bboxes[positioning_obj][0]
    ori_bbox = salient_bboxes[orienting_obj][0]
    query_bbox = salient_bboxes[querying_obj][0]

    # Check distances between all pairs
    pos_ori_dist = calc_3d_bbox_distance_between_objects(pos_bbox, ori_bbox)
    pos_query_dist = calc_3d_bbox_distance_between_objects(pos_bbox, query_bbox)
    ori_query_dist = calc_3d_bbox_distance_between_objects(ori_bbox, query_bbox)

    if any(
        dist > max_distance for dist in [pos_ori_dist, pos_query_dist, ori_query_dist]
    ):
        raise ValueError("Objects are too far apart")

    # Clean object names
    positioning_obj_clean = clean_obj_type(positioning_obj)
    orienting_obj_clean = clean_obj_type(orienting_obj)
    querying_obj_clean = clean_obj_type(querying_obj)

    # Generate question
    if version == "v1":
        task_name = "obj_rel_direction"
        direction, angle, options = calculate_relative_direction(
            pos_bbox, ori_bbox, query_bbox, ambiguity_threshold, difficulty="easy"
        )

        if direction == "ambiguous":
            raise ValueError("Ambiguous spatial relationship between objects")
        assert len(options) == 2, f"Expected 2 options, got {len(options)}: {options}"

        question = qt.OBJ_REL_DIRECTION_TEMPLATE.format(
            positioning_object=positioning_obj_clean,
            orienting_object=orienting_obj_clean,
            querying_object=querying_obj_clean,
        )
    elif version == "vsi_easy":
        task_name = "vsi_obj_rel_direction_easy"
        direction, angle, options = calculate_relative_direction(
            pos_bbox, ori_bbox, query_bbox, ambiguity_threshold, difficulty="easy"
        )
        if direction == "ambiguous":
            raise ValueError("Ambiguous spatial relationship between objects")
        assert len(options) == 2, f"Expected 2 options, got {len(options)}: {options}"

        question = vsi_qt.OBJ_REL_DIRECTION_EASY_TEMPLATE.format(
            positioning_object=positioning_obj_clean,
            orienting_object=orienting_obj_clean,
            querying_object=querying_obj_clean,
        )
    elif version == "vsi_medium":
        task_name = "vsi_obj_rel_direction_medium"
        direction, angle, options = calculate_relative_direction(
            pos_bbox, ori_bbox, query_bbox, ambiguity_threshold, difficulty="medium"
        )
        if direction == "ambiguous":
            raise ValueError("Ambiguous spatial relationship between objects")
        assert len(options) == 3, f"Expected 3 options, got {len(options)}: {options}"

        question = vsi_qt.OBJ_REL_DIRECTION_MEDIUM_TEMPLATE.format(
            positioning_object=positioning_obj_clean,
            orienting_object=orienting_obj_clean,
            querying_object=querying_obj_clean,
        )
    elif version == "vsi_hard":
        task_name = "vsi_obj_rel_direction_hard"
        direction, angle, options = calculate_relative_direction(
            pos_bbox, ori_bbox, query_bbox, ambiguity_threshold, difficulty="hard"
        )
        if direction == "ambiguous":
            raise ValueError("Ambiguous spatial relationship between objects")
        assert len(options) == 4, f"Expected 4 options, got {len(options)}: {options}"

        question = vsi_qt.OBJ_REL_DIRECTION_HARD_TEMPLATE.format(
            positioning_object=positioning_obj_clean,
            orienting_object=orienting_obj_clean,
            querying_object=querying_obj_clean,
        )
    else:
        raise ValueError(f"Invalid version: {version}")

    if minimal:
        task_name += "_minimal"

    # Generate multiple choice question
    mc_question, mc_answer, mc_choices = gen_mc_question(
        question, direction, options, shuffle_options=True
    )

    return {
        "task": task_name,
        "question": question,
        "gt_answer": direction,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": mc_choices,
        "positioning_obj": positioning_obj_clean,
        "orienting_obj": orienting_obj_clean,
        "querying_obj": querying_obj_clean,
        "angle": float(angle),
        "distances": {
            "positioning_orienting": float(pos_ori_dist),
            "positioning_querying": float(pos_query_dist),
            "orienting_querying": float(ori_query_dist),
        },
    }


def gen_obj_rel_dir_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    ambiguity_threshold: float = 10.0,
    max_distance: float = 10.0,
    max_retries: int = 10,
    version: str = "v1",
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    """Generate multiple relative direction QAs."""
    return generate_qa_batch(
        qa_generator=gen_obj_rel_dir_qa,
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        ambiguity_threshold=ambiguity_threshold,
        max_distance=max_distance,
        version=version,
        minimal=minimal,
    )


SPATIAL_QA_GEN_FNS = {
    "obj_abs_distance": gen_obj_abs_dist_qas,
    "obj_rel_distance": gen_obj_rel_dist_qas,
    "obj_rel_direction": gen_obj_rel_dir_qas,
    # VSI-specific
    "vsi_obj_abs_distance": partial(gen_obj_abs_dist_qas, vsi=True),
    "vsi_obj_rel_distance": partial(gen_obj_rel_dist_qas, vsi=True),
    "vsi_obj_rel_direction_easy": partial(gen_obj_rel_dir_qas, version="vsi_easy"),
    "vsi_obj_rel_direction_medium": partial(gen_obj_rel_dir_qas, version="vsi_medium"),
    "vsi_obj_rel_direction_hard": partial(gen_obj_rel_dir_qas, version="vsi_hard"),
    # minimal object VSI questions
    "vsi_obj_abs_distance_minimal": partial(
        gen_obj_abs_dist_qas, vsi=True, minimal=True
    ),
    "vsi_obj_rel_distance_minimal": partial(
        gen_obj_rel_dist_qas, vsi=True, minimal=True
    ),
    "vsi_obj_rel_direction_easy_minimal": partial(
        gen_obj_rel_dir_qas, version="vsi_easy", minimal=True
    ),
    "vsi_obj_rel_direction_medium_minimal": partial(
        gen_obj_rel_dir_qas, version="vsi_medium", minimal=True
    ),
    "vsi_obj_rel_direction_hard_minimal": partial(
        gen_obj_rel_dir_qas, version="vsi_hard", minimal=True
    ),
}
