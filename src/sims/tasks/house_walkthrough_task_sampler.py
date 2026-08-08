from typing import Any, Dict

from sims.tasks.base_task_sampler import BaseTaskSampler


class HouseWalkthroughTaskSampler(BaseTaskSampler):
    """Sample the room-tour task used to generate the SIMS trajectories."""

    task_type_str = "HouseWalkthrough"

    def sample_task_parameters(self) -> Dict[str, Any]:
        return {
            "num_rooms_in_house": len(self.controller.current_scene_json["rooms"]),
        }
