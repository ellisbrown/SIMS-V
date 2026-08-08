"""Natural-language rendering for the SIMS paper trajectory task."""

from typing import Any, Dict

from sims.utils.type_utils import REGISTERED_TASK_PARAMS


def house_walkthrough(task_params: Dict[str, Any]) -> str:
    """Describe the room tour generated for each SIMS trajectory."""
    return (
        f"Go to all {task_params['num_rooms_in_house']} rooms in the house. "
        "Indicate when you have seen a new room and when you are done"
    ).lower()


REGISTERED_INSTRUCTION_TYPES = {
    "HouseWalkthrough": house_walkthrough,
}

assert set(REGISTERED_INSTRUCTION_TYPES) == set(REGISTERED_TASK_PARAMS)
