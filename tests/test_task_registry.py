from sims.data_generation.path_planner_utils import REGISTERED_PLANNERS
from sims.tasks import REGISTERED_TASK_SAMPLERS, REGISTERED_TASKS
from sims.tasks.house_walkthrough_task import HouseWalkthroughTask
from sims.tasks.house_walkthrough_task_sampler import HouseWalkthroughTaskSampler
from sims.utils.string_utils import json_templated_task_string
from sims.utils.task_spec_to_instruction import REGISTERED_INSTRUCTION_TYPES
from sims.utils.type_utils import REGISTERED_TASK_PARAMS


SUPPORTED_TASK = "HouseWalkthrough"


def test_public_task_registries_only_expose_paper_task():
    expected = {SUPPORTED_TASK}

    assert set(REGISTERED_TASKS) == expected
    assert set(REGISTERED_TASK_SAMPLERS) == expected
    assert set(REGISTERED_PLANNERS) == expected
    assert set(REGISTERED_TASK_PARAMS) == expected
    assert set(REGISTERED_INSTRUCTION_TYPES) == expected


def test_house_walkthrough_registry_entries_are_consistent():
    assert REGISTERED_TASKS[SUPPORTED_TASK] is HouseWalkthroughTask
    assert REGISTERED_TASK_SAMPLERS[SUPPORTED_TASK] is HouseWalkthroughTaskSampler
    assert set(REGISTERED_TASK_PARAMS[SUPPORTED_TASK]) == {"num_rooms_in_house"}


def test_house_walkthrough_instruction_and_template_include_room_count():
    task_info = {
        "task_type": SUPPORTED_TASK,
        "num_rooms_in_house": 3,
        "extras": {},
    }

    assert REGISTERED_INSTRUCTION_TYPES[SUPPORTED_TASK](task_info) == (
        "go to all 3 rooms in the house. "
        "indicate when you have seen a new room and when you are done"
    )
    assert '"num_rooms_in_house": 3' in json_templated_task_string(task_info)
