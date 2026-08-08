from functools import partial
from typing import Any, Dict, List

from sims.qa import question_templates as qt
from sims.qa import vsi_question_templates as vsi_qt
from sims.qa.generators.common import (
    gen_count_options,
    gen_distance_options,
    gen_mc_question,
)


# house layout related


def gen_n_rooms_qas(
    preprocessed_data: Dict[str, Any],
    _: int = 1,
) -> List[Dict[str, Any]]:
    """
    Generate a single QA about the total number of rooms in the house.

    NOTE: it is only possible to generate one question per house layout.
    """
    spatial_metadata = preprocessed_data["spatial_metadata"]
    if "room_data" not in spatial_metadata:
        raise ValueError(
            f"No 'room_data' in spatial metadata. Did you store it during preprocessing? {spatial_metadata.keys()}"
        )

    room_count = len(spatial_metadata["room_data"])

    question = qt.N_ROOMS_TEMPLATE  # "How many total rooms are in this house?"

    # Reuse your existing multiple-choice "count" helper
    # or just use gen_count_options() + gen_mc_question()
    options = gen_count_options(room_count)
    mc_question, mc_answer, mc_choices = gen_mc_question(
        question, room_count, options, shuffle_options=True
    )

    return [
        {
            "task": "n_rooms",
            "question": question,
            "gt_answer": room_count,
            "mc_question": mc_question,
            "mc_answer": mc_answer,
            "mc_choices": mc_choices,
            "id": 0,
        }
    ]


def gen_house_size_est_qas(
    preprocessed_data: Dict[str, Any],
    _: int = 1,
    round_to: float = 1,
    vsi: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a single QA about the estimated size of the house.

    NOTE: it is only possible to generate one question per house layout.
    """

    spatial_metadata = preprocessed_data["spatial_metadata"]
    if "house_area" not in spatial_metadata:
        raise ValueError(
            "No 'house_area' in spatial metadata. Did you store it during preprocessing?"
        )
    if "room_data" not in spatial_metadata:
        raise ValueError(
            "No 'room_data' in spatial metadata. Did you store it during preprocessing?"
        )

    house_area = spatial_metadata["house_area"]
    # rounded_area = round(round(house_area / round_to) * round_to)
    rounded_area = round(house_area / round_to) * round_to

    # Generate multiple-choice options
    # NOTE: generate options in units of the round_to value, then scale back
    options = gen_distance_options(
        answer=house_area / round_to,
        num_options=4,
        prob_outlier=0.1,
        digits=0,
        ambiguity_threshold=1,  # use the unit of the round_to
    )
    # scale back to original units
    rounded_options = [opt * round_to for opt in options]

    room_count = len(spatial_metadata["room_data"])

    if vsi:
        task_name = "vsi_room_size_est"
        question = vsi_qt.ROOM_SIZE_TEMPLATE
    else:
        task_name = "house_size_est"
        question = qt.HOUSE_SIZE_ESTIMATION_TEMPLATE.format(n=room_count)

    mc_question, mc_answer, mc_choices = gen_mc_question(
        question, rounded_area, rounded_options, shuffle_options=True
    )

    return [
        {
            "task": task_name,
            "question": question,
            "gt_answer": rounded_area,
            "mc_question": mc_question,
            "mc_answer": mc_answer,
            "mc_choices": mc_choices,
            "house_area": house_area,
            "num_rooms": room_count,
            "id": 0,
        }
    ]


LAYOUT_QA_GEN_FNS = {
    "n_rooms": gen_n_rooms_qas,
    "house_size_est": partial(gen_house_size_est_qas, round_to=1),
    "vsi_room_size_est": partial(gen_house_size_est_qas, round_to=0.1, vsi=True),
}
