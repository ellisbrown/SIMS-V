import hashlib
import math
import os
import random
import re
from typing import Any, Callable, Dict, List, Set


# Path constants — resolve relative to the parent `qa/` package directory
_QA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJECTS_PATH = os.path.join(_QA_DIR, "objects.txt")
VSI_OBJECTS_PATH = os.path.join(_QA_DIR, "vsi_objects.txt")

# Object lists — read at import time. These are small local text files (~100 lines each),
# not slow network/dataset loads, so module-level I/O is acceptable here.
with open(OBJECTS_PATH, "r") as _f:
    ALL_OBJECTS = [line.strip() for line in _f]
with open(VSI_OBJECTS_PATH, "r") as _f:
    VSI_OBJECTS = [line.strip() for line in _f]

MIN_NUM_FRAMES = 25


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-independent random seed from stable input values."""
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def split_camel_case(value: str) -> str:
    """Split camel case and letter-digit boundaries into lowercase words."""
    if not value:
        return value

    return re.sub(
        r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])",
        " ",
        value,
    ).lower()


def clean_obj_type(obj_type: str) -> str:
    return split_camel_case(obj_type.replace("Obja", ""))


ALL_OBJECTS_CLEAN = [clean_obj_type(obj) for obj in ALL_OBJECTS]
VSI_OBJECTS_CLEAN = [clean_obj_type(obj) for obj in VSI_OBJECTS]


def gen_mc_question(
    question, answer, options, shuffle_options=True, shuffle_letters=False
):
    n = len(options)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[0:n])

    options = options.copy()
    if shuffle_options:
        random.shuffle(options)
    if shuffle_letters:
        random.shuffle(letters)

    mc_answer = letters[options.index(answer)]
    mc_options = [f"{letter}. {option}" for letter, option in zip(letters, options)]
    mc_question = f"{question}\n" + "\n".join(mc_options)

    return mc_question, mc_answer, mc_options


def generate_qa_batch(
    qa_generator: Callable,
    preprocessed_data: Dict[str, Any],
    n: int,
    max_retries: int = 10,
    **generator_kwargs,
) -> List[Dict[str, Any]]:
    """
    Generic function to generate multiple QA pairs with retry logic and duplicate prevention.

    Args:
        qa_generator: Function that generates a single QA pair
        preprocessed_data: Dictionary containing scene data
        n: Number of QAs to generate
        max_retries: Maximum number of attempts to generate each QA
        **generator_kwargs: Additional kwargs to pass to qa_generator

    Returns:
        List of QA dictionaries

    Raises:
        ValueError: If unable to generate requested number of QAs
    """
    qas = []
    used_combinations: Set[str] = set()
    total_retries = 0

    while len(qas) < n and total_retries < max_retries * n:
        try:
            # Generate a new QA
            qa = qa_generator(preprocessed_data, **generator_kwargs)

            # Get objects involved in the question
            task = qa["task"]
            # if task in {
            #     'obj_rel_direction',
            #     'vsi_obj_rel_direction_easy',
            #     'vsi_obj_rel_direction_medium',
            #     'vsi_obj_rel_direction_hard'
            # }:
            if "obj_rel_direction" in task:
                objects = [
                    qa["positioning_obj"],
                    qa["orienting_obj"],
                    qa["querying_obj"],
                ]
            # elif task in {'obj_abs_distance', 'vsi_obj_abs_distance'}:
            elif "obj_abs_distance" in task:
                objects = [qa["ref_obj"], qa["other_obj"]]
            # elif task in {'obj_rel_distance', 'vsi_obj_rel_distance'}:
            elif "obj_rel_distance" in task:
                objects = [qa["ref_obj"]] + [
                    choice.split(". ")[1] for choice in qa["mc_choices"]
                ]
            # elif task == "vsi_obj_appearance_order" or task.startswith('temporal_order_'):
            elif "obj_appearance_order" in task or task.startswith("temporal_order_"):
                objects = qa["selected_objects"]
            elif task == "temporal_rel":
                objects = [qa["object_1"], qa["object_2"]]
            elif task == "descriptive_binary":
                objects = [qa["object_type"]]
            else:
                raise NotImplementedError(f"Unknown task type: {task}")

            # Create unique key for this combination
            combination_key = "|".join(sorted(objects))

            # Skip if we've already used this combination
            if combination_key in used_combinations:
                total_retries += 1
                continue

            # Add QA and mark combination as used
            qa["id"] = len(qas)  # Add sequential ID
            qas.append(qa)
            used_combinations.add(combination_key)

        except NotImplementedError as e:
            raise e

        except ValueError:
            total_retries += 1
            continue

    if len(qas) < 1:
        raise ValueError(
            f"Could not generate any valid QAs after {total_retries} attempts."
        )

    return qas


def gen_count_options(answer):
    # Generate plausible choices
    options = [answer]
    possible_offsets = set(range(-3, 4)) - {0}
    while len(options) < 4:
        offset = random.choice(sorted(possible_offsets))
        new_choice = max(1, answer + offset)
        if new_choice not in options:
            options.append(new_choice)
            possible_offsets.remove(offset)

    return options


def gen_distance_options(
    answer: float,
    num_options: int = 4,
    prob_outlier: float = 0.1,
    digits: int = 1,
    ambiguity_threshold: float = 0.2,
) -> list[float]:
    """
    Generate plausible multiple-choice distances around the correct answer.
    Ensures distinct choices and avoids very close options.
    """

    if answer < 0:
        raise ValueError(f"Invalid negative answer: {answer}")
    if num_options < 1:
        raise ValueError("num_options must be positive")
    if digits < 0:
        raise ValueError("digits must be non-negative")
    if ambiguity_threshold < 0:
        raise ValueError("ambiguity_threshold must be non-negative")
    if not 0 <= prob_outlier <= 1:
        raise ValueError("prob_outlier must be between 0 and 1")

    rounded_answer = round(answer, digits)
    if num_options == 1:
        return [rounded_answer]

    # Build a finite pool of multiplicative distractors first.  This retains
    # the old helper's preference for values near the answer without relying
    # on an unbounded retry loop (which could hang after coarse rounding).
    near_factors = [0.4, 0.55, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0, 2.3]
    outlier_factors = [2.5, 2.8, 3.1, 3.5]
    random.shuffle(near_factors)
    random.shuffle(outlier_factors)
    factors = (
        outlier_factors + near_factors
        if random.random() < prob_outlier
        else near_factors + outlier_factors
    )

    candidates = []
    seen = {rounded_answer}

    def add_candidate(candidate):
        candidate = round(candidate, digits)
        if candidate < 0 or candidate in seen:
            return
        if abs(candidate - answer) < ambiguity_threshold:
            return
        seen.add(candidate)
        candidates.append(candidate)

    for factor in factors:
        add_candidate(answer * factor)

    # Coarse representations can collapse every multiplicative candidate to
    # the same value.  A bounded grid fallback always supplies enough
    # non-negative, representable values, including for answer=.2, digits=0.
    resolution = 10 ** (-digits)
    first_up = max(
        1,
        math.ceil((ambiguity_threshold + answer - rounded_answer) / resolution - 1e-12),
    )
    for offset in range(first_up, first_up + num_options * 2 + 2):
        add_candidate(rounded_answer + offset * resolution)
        add_candidate(rounded_answer - offset * resolution)

    if len(candidates) < num_options - 1:
        raise RuntimeError("Could not construct enough distinct distance options")

    return [rounded_answer, *candidates[: num_options - 1]]
