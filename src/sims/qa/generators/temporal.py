import itertools
import random
from functools import partial
from typing import Any, Dict, List

from sims.qa import question_templates as qt
from sims.qa import vsi_question_templates as vsi_qt
from sims.qa.generators.common import (
    VSI_OBJECTS_CLEAN,
    clean_obj_type,
    gen_mc_question,
    generate_qa_batch,
)


def _category_appearances(
    preprocessed_data: Dict[str, Any], minimal: bool = False
) -> Dict[int, List[str]]:
    """Return cleaned object categories grouped by first-appearance frame.

    Multiple simulator instances of the same cleaned category represent one
    category in the question, so its first appearance is the earliest frame
    of any of those instances.
    """
    first_frame_by_category = {}
    for key in sorted(preprocessed_data["salient_objects_n_frames"]):
        category = clean_obj_type(preprocessed_data["object_info"][key]["object_type"])
        if minimal and category not in VSI_OBJECTS_CLEAN:
            continue
        indices = preprocessed_data["salient_obj_idx_map"].get(key, [])
        if not indices:
            continue
        first_frame = min(indices)
        first_frame_by_category[category] = min(
            first_frame, first_frame_by_category.get(category, first_frame)
        )

    categories_by_frame = {}
    for category, first_frame in sorted(first_frame_by_category.items()):
        categories_by_frame.setdefault(first_frame, []).append(category)
    return categories_by_frame


# >> TEMPORAL << #
# TODO: vary the number of objects in the questions
def gen_temporal_rel_qa(
    preprocessed_data: Dict[str, Any], shuffle: bool = True
) -> Dict[str, Any]:
    """
    Generate a single temporal relation question comparing appearance times of different objects.

    Args:
        preprocessed_data: Dictionary containing preprocessed data
        shuffle: Whether to shuffle MCQ choices

    Returns:
        Dictionary containing question information

    Raises:
        ValueError: If unable to generate a valid question
    """
    categories_by_frame = _category_appearances(preprocessed_data)
    frames = sorted(categories_by_frame)
    if len(frames) < 2:
        raise ValueError("Not enough object categories with distinct appearance times")

    frame_1, frame_2 = random.sample(frames, 2)
    type_1 = random.choice(categories_by_frame[frame_1])
    type_2 = random.choice(categories_by_frame[frame_2])

    # Determine which object appeared first
    gt_answer = type_1 if frame_1 < frame_2 else type_2

    question = qt.TEMPORAL_RELATION_TEMPLATE.format(
        type_1=type_1,
        type_2=type_2,
    )

    # MCQ choices
    options = [type_1, type_2]
    mc_question, mc_answer, choices = gen_mc_question(
        question, gt_answer, options, shuffle_options=shuffle
    )

    return {
        "task": "temporal_rel",
        "question": question,
        "gt_answer": gt_answer,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": choices,
        "object_1": type_1,
        "object_2": type_2,
    }


def gen_temporal_rel_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    shuffle: bool = True,
    max_retries: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate multiple temporal relation questions.

    Args:
        preprocessed_data: Dictionary containing preprocessed data
        n: Number of questions to generate
        shuffle: Whether to shuffle MCQ choices
        max_retries: Maximum number of attempts for generation

    Returns:
        List of temporal relation questions
    """
    # Calculate maximum possible unique questions
    categories_by_frame = _category_appearances(preprocessed_data)
    frame_counts = [len(values) for values in categories_by_frame.values()]
    max_possible = sum(
        left_count * right_count
        for index, left_count in enumerate(frame_counts)
        for right_count in frame_counts[index + 1 :]
    )

    if max_possible < 1:
        raise ValueError("Not enough objects to generate comparison questions")

    if n > max_possible:
        # print(f"Warning: Requested {n} questions, but only {max_possible} unique questions possible.")
        n = max_possible

    questions = generate_qa_batch(
        qa_generator=gen_temporal_rel_qa,
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        shuffle=shuffle,
    )

    return questions


def gen_temporal_order_qa(
    preprocessed_data: Dict[str, Any],
    num_objects: int = 4,
    num_options: int = 4,
    shuffle_question_objects: bool = True,
    vsi: bool = False,
    minimal: bool = False,
) -> Dict[str, Any]:
    """
    Generate a temporal QA that asks for the appearance order of multiple objects.

    Args:
        preprocessed_data: Dictionary containing preprocessed data.
        num_objects: Number of objects to include in the question.
        num_options: Number of multiple-choice options to generate (one correct + distractors).
        shuffle_question_objects: Whether to display the list of objects in random order in the question.
        vsi: Whether to generate a VSI question.
        minimal: Whether to only generate questions for VSI objects.

    Returns:
        A dictionary containing the QA information.

    Raises:
        ValueError: If there are not enough objects.
    """
    categories_by_frame = _category_appearances(preprocessed_data, minimal=minimal)
    appearance_frames = sorted(categories_by_frame)

    if len(appearance_frames) < num_objects:
        raise ValueError(
            "Not enough object categories with distinct first-appearance frames "
            f"to generate an appearance order question: found "
            f"{len(appearance_frames)} but need {num_objects}"
        )

    sampled_frames = random.sample(appearance_frames, num_objects)
    appearance_info = [
        (random.choice(categories_by_frame[first_idx]), first_idx)
        for first_idx in sampled_frames
    ]

    # Sort by appearance time (the lower the frame index, the earlier it appears)
    sorted_info = sorted(appearance_info, key=lambda tup: tup[1])
    correct_order = [obj for (obj, _) in sorted_info]
    correct_order_str = ", ".join(correct_order)

    # For the question text, list the objects in random order (so the candidate must sort them)
    question_objects = [obj for (obj, _) in appearance_info]
    if shuffle_question_objects:
        random.shuffle(question_objects)

    if vsi:
        assert num_objects == 4, "VSI questions are only supported for 4 objects"
        task_name = "vsi_obj_appearance_order"
        question_text = vsi_qt.OBJ_APPEARANCE_ORDER_TEMPLATE.format(
            choice_a=question_objects[0],
            choice_b=question_objects[1],
            choice_c=question_objects[2],
            choice_d=question_objects[3],
        )
    else:
        task_name = f"temporal_order_{num_objects}"
        question_text = qt.TEMPORAL_APPEARANCE_ORDER_TEMPLATE.format(
            objects=", ".join(question_objects)
        )

    if minimal:
        task_name += "_minimal"

    # --- Generate distractor options ---
    # We generate all possible permutations (if the number is small) and remove the correct one.
    all_perms = set(itertools.permutations(correct_order))
    all_perms.discard(tuple(correct_order))
    # Convert each permutation tuple to a string, e.g., "door, basket, plant, bed"
    all_perm_strs = sorted({", ".join(perm) for perm in all_perms})

    # If there are fewer distractors than needed, use all available; otherwise sample randomly.
    num_distractors_needed = num_options - 1
    if len(all_perm_strs) < num_distractors_needed:
        distractors = all_perm_strs
    else:
        distractors = random.sample(all_perm_strs, num_distractors_needed)

    # Build the complete options list (including the correct ordering)
    options = distractors + [correct_order_str]

    # It is a good idea to shuffle the answer choices so the correct answer is not always in the same position.
    mc_question, mc_answer, choices = gen_mc_question(
        question_text, correct_order_str, options, shuffle_options=True
    )

    return {
        "task": task_name,
        "question": question_text,
        "gt_answer": correct_order_str,
        "mc_question": mc_question,
        "mc_answer": mc_answer,
        "mc_choices": choices,
        "selected_objects": correct_order,  # sorted by appearance time
    }


def gen_temporal_order_qas(
    preprocessed_data: Dict[str, Any],
    n: int,
    num_objects: int = 4,
    num_options: int = 4,
    shuffle: bool = True,
    max_retries: int = 10,
    vsi: bool = False,
    minimal: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate multiple temporal order questions.

    Args:
        preprocessed_data: Dictionary with preprocessed scene data.
        n: Number of QA pairs to generate.
        num_objects: Number of objects to include in each question.
        num_options: Number of multiple-choice options (default is 4).
        shuffle: Whether to shuffle the MC options.
        max_retries: Maximum attempts per QA generation.
        vsi: Whether to generate VSI questions.
        minimal: Whether to only generate questions for VSI objects.

    Returns:
        List of temporal order QA dictionaries.
    """
    # We use the generic QA batch generator with our new function.
    return generate_qa_batch(
        qa_generator=partial(
            gen_temporal_order_qa,
            num_objects=num_objects,
            num_options=num_options,
            shuffle_question_objects=shuffle,
        ),
        preprocessed_data=preprocessed_data,
        n=n,
        max_retries=max_retries,
        vsi=vsi,
        minimal=minimal,
    )


TEMPORAL_QA_GEN_FNS = {
    "temporal_rel": gen_temporal_rel_qas,
    "temporal_order_2": partial(gen_temporal_order_qas, num_objects=2),
    "temporal_order_3": partial(gen_temporal_order_qas, num_objects=3),
    "temporal_order_4": partial(gen_temporal_order_qas, num_objects=4),
    "temporal_order_5": partial(gen_temporal_order_qas, num_objects=5),
    "temporal_order_6": partial(gen_temporal_order_qas, num_objects=6),
    "vsi_obj_appearance_order": partial(
        gen_temporal_order_qas, num_objects=4, vsi=True
    ),
    "vsi_obj_appearance_order_minimal": partial(
        gen_temporal_order_qas, num_objects=4, vsi=True, minimal=True
    ),
}
