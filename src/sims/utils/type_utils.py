from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from allenact.base_abstractions.sensor import Sensor
from attrs import define

from sims.environment.action_spaces import AbstractActionSpace


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class KeyedDefaultDict(defaultdict):
    """A defaultdict that passes the missing key to its default factory."""

    def __missing__(self, key: Any):
        return self.default_factory(key)


@define
class RewardConfig:
    step_penalty: float
    goal_success_reward: float
    failed_stop_reward: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool
    failed_action_penalty: float = 0.0


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3


class AbstractTaskArgs(TypedDict):
    sensors: List[Sensor]
    max_steps: int
    action_space: AbstractActionSpace
    reward_config: Optional[RewardConfig]


REGISTERED_TASK_PARAMS: Dict[str, List[str]] = {}


def register_task_specific_params(cls):
    REGISTERED_TASK_PARAMS[cls.__name__] = sorted(cls.__required_keys__)
    return cls


@register_task_specific_params
class HouseWalkthrough(TypedDict):
    num_rooms_in_house: int


def get_task_relevant_synsets(task_spec: Dict[str, Any]) -> List[str]:
    """Return all WordNet synsets referenced by a task specification."""
    synsets = set()
    for key, value in task_spec.items():
        if "synset" not in key:
            continue
        if key.endswith("synset_to_object_ids"):
            assert isinstance(value, dict)
            synsets.update(value.keys())
        elif key in ["synsets", "reference_synsets"]:
            assert isinstance(value, Sequence)
            synsets.update(value)
        elif key in [
            "dest_receptacle_synset",
            "dest_synset",
            "condition_synset",
            "positive_dest_synset",
            "negative_dest_synset",
        ]:
            assert isinstance(value, str)
            synsets.add(value)
        else:
            raise NotImplementedError(f"Unsupported synset field: {key}")
    return list(synsets)
