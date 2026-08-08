"""Task definitions supported by the SIMS paper release."""

from typing import Dict, Type

from sims.tasks.abstract_task import AbstractSimsTask
from sims.tasks.base_task_sampler import BaseTaskSampler
from sims.tasks.house_walkthrough_task import HouseWalkthroughTask
from sims.tasks.house_walkthrough_task_sampler import HouseWalkthroughTaskSampler

REGISTERED_TASKS: Dict[str, Type[AbstractSimsTask]] = {
    HouseWalkthroughTask.task_type_str: HouseWalkthroughTask,
}
REGISTERED_TASK_SAMPLERS: Dict[str, Type[BaseTaskSampler]] = {
    HouseWalkthroughTaskSampler.task_type_str: HouseWalkthroughTaskSampler,
}

__all__ = [
    "HouseWalkthroughTask",
    "HouseWalkthroughTaskSampler",
    "REGISTERED_TASKS",
    "REGISTERED_TASK_SAMPLERS",
]
