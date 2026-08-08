import random
from functools import partial
from typing import Any, Dict, List

from sims.qa import question_templates as qt
from sims.qa import vsi_question_templates as vsi_qt
from sims.qa.generators.common import (
    ALL_OBJECTS,
    VSI_OBJECTS_CLEAN,
    clean_obj_type,
    gen_count_options,
    gen_distance_options,
    gen_mc_question,
    generate_qa_batch,
)


# >> DESCRIPTIVE / RECOGNITION << #


def gen_binary_qa(
    preprocessed_data: Dict[str, Any], prob_seen: float = 0.7
) -> Dict[str, Any]:
    """
    Generate a single binary question about object presence.

    Args:
        preprocessed_data: Dictionary containing preprocessed data
        prob_seen: Probability of generating a question about a seen object

    Returns:
        Dictionary containing question information

    Raises:
        ValueError: If unable to generate a valid question
    """
    salient_objects_n_frames = preprocessed_data["salient_objects_n_frames"]
    object_info = preprocessed_data["object_info"]
    max_visibility_map = preprocessed_data["max_visibility_map"]
    object_keys = list(salient_objects_n_frames.keys())

    if not object_keys:
        raise ValueError("No objects available to generate question")

    # Get seen and unseen objects
    seen_objects = {object_info[key]["object_type"] for key in object_keys}
    seen_categories = {clean_obj_type(obj_type) for obj_type in seen_objects}
    unseen_objects = {
        obj_type
        for obj_type in ALL_OBJECTS
        if clean_obj_type(obj_type) not in seen_categories
    }

    # Decide whether to use a seen or unseen object
    use_seen = random.random() < prob_seen

    if use_seen:
        if not seen_objects:
            raise ValueError("No seen objects available")

        # Select a random seen object
        key = random.choice(object_keys)
        obj = object_info[key]
        obj_type = obj["object_type"]
        gt_answer = "Yes"
        max_visibility = max_visibility_map[key]

    else:
        if not unseen_objects:
            raise ValueError("No unseen objects available")

        obj_type = random.choice(sorted(unseen_objects))
        gt_answer = "No"
        max_visibility = 0

    if obj_type is None or len(obj_type) == 0:
        raise ValueError("Invalid object type")

    obj_type_clean = clean_obj_type(obj_type)
    question = qt.BINARY_RECOG_TEMPLATE.format(type=obj_type_clean)

    # MCQ choices
    options = ["Yes", "No"]
    mc_question, mc_answer, choices = gen_mc_question(question, gt_answer, options)

    return {
        "task": "descriptive_binary",
        "object_type": obj_type,
        "max_visibility": max_visibility,
        "question": question,
        "gt_answer": gt_answer,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": choices,
    }


def gen_binary_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    prob_seen: float = 0.7,
    max_retries: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate multiple binary questions about object presence.

    Args:
        preprocessed_data: Dictionary containing preprocessed data
        n: Number of questions to generate
        prob_seen: Probability of generating a question about a seen object
        max_retries: Maximum number of attempts for generation

    Returns:
        List of binary questions

    Raises:
        ValueError: If unable to generate requested number of questions
    """
    # Calculate maximum possible unique questions
    salient_objects_n_frames = preprocessed_data["salient_objects_n_frames"]
    seen_count = len(salient_objects_n_frames)
    unseen_count = len(ALL_OBJECTS) - seen_count
    max_possible = seen_count + unseen_count

    if n > max_possible:
        raise ValueError(
            f"Requested {n} questions but only {max_possible} unique objects available"
        )

    questions = generate_qa_batch(
        qa_generator=gen_binary_qa,
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        prob_seen=prob_seen,
    )

    return questions


def gen_obj_count_qas(
    preprocessed_data: Dict[str, Any], n: int, vsi: bool = False, minimal: bool = False
) -> List[Dict[str, Any]]:
    """Count distinct salient instances observed along the video trajectory.

    This intentionally uses ``salient_type_counts`` rather than the complete
    house inventory so that it reproduces the labels used by the paper data.
    """
    if vsi:
        task_name = "vsi_obj_count"
    else:
        task_name = "obj_count"

    if minimal:
        task_name += "_minimal"

    # NOTE: handle n manually because object types are naturally unique

    spatial_metadata = preprocessed_data["spatial_metadata"]
    salient_cts = spatial_metadata["salient_type_counts"]
    type_cts = list(salient_cts.items())
    random.shuffle(type_cts)

    qas = []

    if not type_cts:
        raise ValueError("No salient objects to generate count questions")

    if all(ct < 1 for _, ct in type_cts):
        raise ValueError(
            "All objects seen less than once, cannot generate count questions"
        )

    for obj_type, ct in type_cts:
        if len(qas) >= n:
            assert len(qas) == n, f"Expected {n} questions, got {len(qas)}"
            break

        obj_type_clean = clean_obj_type(obj_type)

        if minimal and obj_type_clean not in VSI_OBJECTS_CLEAN:
            continue

        if vsi:
            question = vsi_qt.OBJ_COUNTING_TEMPLATE.format(category=obj_type_clean)
        else:
            question = qt.OBJ_COUNTING_TEMPLATE.format(type=obj_type_clean)

        # MCQ choices
        options = gen_count_options(ct)
        mc_question, mc_answer, choices = gen_mc_question(
            question, ct, options, shuffle_options=True
        )

        qas.append(
            {
                "task": task_name,
                "object_type": obj_type,
                "question": question,
                "gt_answer": ct,
                "mc_question": mc_question,
                "mc_answer": mc_answer,
                "mc_choices": choices,
            }
        )

    return qas


def gen_obj_size_est_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    use_longest: bool = True,
    round_to: int = 5,
    version: str = "v1",
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate questions about the estimated size of objects in the scene.

    Args:
        preprocessed_data: Preprocessed data containing "spatial_metadata", including:
              - "salient_type_counts": a Counter mapping object types to counts.
              - "salient_object_bbox": a dict mapping object types to lists of bbox dicts.
        n: Number of size-estimation QA pairs to generate.
        use_longest: Whether to ask about the longest dimension (True) or the shortest (False).
        round_to: How many cm to round the answer to. Default is 5.
        version: Which version of the question to generate. Options are "v1", "v2", or "vsi".
        minimal: Whether to only generate questions for VSI objects.

    Returns:
        A list of dictionaries, each representing a size-estimation QA pair.

    Raises:
        ValueError: If no valid objects are available to generate questions.
    """
    if not isinstance(round_to, int) or round_to < 1:
        raise ValueError("round_to must be a positive integer")

    if version not in ["v1", "v2", "vsi"]:
        raise ValueError("Invalid version, must be 'v1', 'v2', or 'vsi'")

    spatial_metadata = preprocessed_data["spatial_metadata"]
    salient_cts = spatial_metadata["salient_type_counts"]
    type_cts = list(salient_cts.items())
    random.shuffle(type_cts)

    if minimal:
        type_cts = [
            (obj_type, ct)
            for obj_type, ct in type_cts
            if clean_obj_type(obj_type) in VSI_OBJECTS_CLEAN
        ]

    # The VSI wording identifies only an object category and does not define
    # which instance to measure when several are visible.  Restrict it to
    # unambiguous, single-instance categories.  The v1/v2 wording explicitly
    # asks for the first instance seen and is handled below.
    if version == "vsi":
        type_cts = [(obj_type, ct) for obj_type, ct in type_cts if ct == 1]

    if not type_cts:
        raise ValueError("No salient objects to generate size estimation questions.")

    qas: List[Dict[str, Any]] = []

    for obj_type, ct in type_cts:
        if len(qas) >= n:
            # Once we have generated enough questions, break out.
            assert len(qas) == n, f"Expected {n} questions, got {len(qas)}"
            break

        # Ensure there are bounding boxes available for this object type.
        if obj_type not in spatial_metadata["salient_object_bbox"]:
            continue

        bboxes = spatial_metadata["salient_object_bbox"][obj_type]
        if not bboxes:
            continue

        if version in {"v1", "v2"}:
            salient_obj_idx_map = preprocessed_data["salient_obj_idx_map"]
            first_seen_bboxes = []
            for bbox_candidate in bboxes:
                object_id = bbox_candidate.get("object_id")
                appearance_indices = salient_obj_idx_map.get(object_id, [])
                if appearance_indices:
                    first_seen_bboxes.append(
                        (min(appearance_indices), str(object_id), bbox_candidate)
                    )
            if not first_seen_bboxes:
                continue
            earliest_frame = min(item[0] for item in first_seen_bboxes)
            earliest_bboxes = [
                item[2] for item in first_seen_bboxes if item[0] == earliest_frame
            ]
            # The viewer cannot disambiguate two category instances that first
            # appear in the same frame, so do not choose one via a hidden ID
            # tie-break.
            if len(earliest_bboxes) != 1:
                continue
            bbox = earliest_bboxes[0]
        else:
            # VSI categories are restricted to one visible instance above.
            bbox = bboxes[0]

        # Compute the three dimensions (in centimeters) from the bbox.
        # bbox["axesLengths"] are the lengths of the principal axes of= the object in meters.
        dims = [dim * 100 for dim in bbox["axesLengths"]]

        # Optionally, round each dimension. Here we round to the nearest integer.
        dims = [round(dim) for dim in dims]

        if use_longest:
            long_short = "longest"
            task_name = "obj_long_size_est"
            true_answer = max(dims)
        else:
            long_short = "shortest"
            task_name = "obj_short_size_est"
            true_answer = min(dims)

        # Form the question text using the provided template.
        clean_type = clean_obj_type(obj_type)
        if version == "v1":
            question_text = qt.OBJ_SIZE_ESTIMATION_TEMPLATE.format(
                long_short=long_short, obj=clean_type
            )
        elif version == "v2":
            task_name += "_v2"
            question_text = qt.OBJ_SIZE_ESTIMATION_TEMPLATE_V2.format(
                long_short=long_short, obj=clean_type
            )
        elif version == "vsi":
            # overwrite task name for VSI questions
            task_name = "vsi_obj_size_est"
            question_text = vsi_qt.OBJ_SIZE_ESTIMATE_TEMPLATE.format(
                category=clean_type
            )
        else:
            raise ValueError(f"Invalid version: {version}")

        if minimal:
            task_name += "_minimal"

        # Generate multiple-choice options.
        # --> reuse gen_distance_options with digits=0 (to produce integer answers)
        float_options = gen_distance_options(
            true_answer / round_to,
            num_options=4,
            prob_outlier=0.1,
            digits=0,
            # x cm threshold to avoid options too close
            ambiguity_threshold=1,  # use the unit of the round_to
        )
        options = [round(opt * round_to) for opt in float_options]
        rounded_answer = round(round(true_answer / round_to) * round_to)
        mc_question, mc_answer, choices = gen_mc_question(
            question_text, rounded_answer, options, shuffle_options=True
        )

        qa = {
            "task": task_name,
            "object_type": obj_type,
            "type": long_short,
            "question": question_text,
            "gt_answer": rounded_answer,
            "gt_size": true_answer,
            "mc_question": mc_question,
            "mc_answer": mc_answer,
            "mc_choices": choices,
            "bbox": bbox,
            "dims_cm": dims,  # Provides all three computed dimensions.
            "count": ct,
        }
        qas.append(qa)

    if len(qas) < 1:
        raise ValueError("Could not generate any valid size estimation questions.")
    return qas


DESCRIPTIVE_QA_GEN_FNS = {
    "descriptive_binary": gen_binary_qas,
    "obj_count": gen_obj_count_qas,
    "obj_long_size_est": gen_obj_size_est_qas,
    "obj_long_size_est_v2": partial(gen_obj_size_est_qas, version="v2"),
    "obj_short_size_est": partial(gen_obj_size_est_qas, use_longest=False, round_to=1),
    # VSI-specific
    "vsi_obj_count": partial(gen_obj_count_qas, vsi=True),
    "vsi_obj_size_est": partial(gen_obj_size_est_qas, version="vsi"),
    # minimal object VSI questions
    "vsi_obj_count_minimal": partial(gen_obj_count_qas, vsi=True, minimal=True),
    "vsi_obj_size_est_minimal": partial(
        gen_obj_size_est_qas, version="vsi", minimal=True
    ),
}
