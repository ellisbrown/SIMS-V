import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super

from sims.environment.stretch_controller import StretchController
from sims.environment.stretch_state import StretchState
from sims.tasks import AbstractSimsTask


class AnObjectIsInHand(Sensor):
    def __init__(self, uuid: str = "an_object_is_in_hand") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        objects_in_hand = env.get_held_objects()
        # return np.array([len(objects_in_hand) > 0], dtype=np.int64)
        return np.array(len(objects_in_hand) > 0, dtype=np.int64)


class RelativeArmLocationMetadata(Sensor):
    def __init__(self, uuid: str = "relative_arm_location_metadata") -> None:
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(3)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> np.ndarray:
        wrist_pose = StretchState(env).wrist_pose
        array_pose = [0, wrist_pose["y"], wrist_pose["z"], wrist_pose["yaw"]]
        return np.array(array_pose, dtype=np.float64)
