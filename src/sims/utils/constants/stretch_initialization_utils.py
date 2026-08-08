import os

import ai2thor.fifo_server
from torch.distributions.utils import lazy_property

from sims.data_generation.paths import configured_objaverse_data_dir

# from sims.utils.type_utils import THORActions

STRETCH_COMMIT_ID = "bfdffefc19494d2e807133a11741d18054a09a86"


try:
    from ai2thor.hooks.procedural_asset_hook import (
        ProceduralAssetHookRunner,
        get_all_asset_ids_recursively,
        create_assets_if_not_exist,
    )
except ImportError:
    raise ImportError(
        "Cannot import `ProceduralAssetHookRunner`. Please install ai2thor from main branch:\n"
        "```\nuv sync\n```"
    )

# AGENT_ROTATION_DEG = 30
# AGENT_MOVEMENT_CONSTANT = 0.2
HORIZON = 0  # RH: Do not change from 0! this is now set elsewhere with RotateCameraMount actions
GRID_SIZE = 0.2
# ARM_MOVE_CONSTANT = 0.1
# WRIST_ROTATION = 10

EMPTY_BBOX = [1000, 1000, 1000, 1000, 0]
EMPTY_DOUBLE_BBOX = EMPTY_BBOX + EMPTY_BBOX

ORIGINAL_INTEL_W, ORIGINAL_INTEL_H = 1280, 720
INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 396, 224
STRETCH_CAMERA_HORIZONTAL_CROP = 6


def cropped_stretch_camera_width(width: int) -> int:
    """Return the RGB/depth width after the Stretch camera's edge crop."""
    if width <= 2 * STRETCH_CAMERA_HORIZONTAL_CROP:
        raise ValueError(
            "Stretch camera width must exceed its 12-pixel horizontal crop; "
            f"got {width}."
        )
    return width - 2 * STRETCH_CAMERA_HORIZONTAL_CROP


INTEL_WIDTH_CROPPED, INTEL_HEIGHT_CROPPED = (
    cropped_stretch_camera_width(INTEL_CAMERA_WIDTH),
    INTEL_CAMERA_HEIGHT,
)
INTEL_VERTICAL_FOV = 59
AGENT_RADIUS_LIST = [(0, 0.5), (1, 0.4), (2, 0.3), (3, 0.2)]

MAXIMUM_SERVER_TIMEOUT = 1000  # default : 100 Need to increase this for cloudrendering

AGENTS_BASE_HEIGHT = 0.900992214679718


class ProceduralAssetHookRunnerResetOnNewHouse(ProceduralAssetHookRunner):
    @lazy_property
    def last_asset_id_set(self):
        return set()

    def Initialize(self, action, controller):
        if self.asset_limit > 0:
            return controller.step(
                action="DeleteLRUFromProceduralCache", assetLimit=self.asset_limit
            )

    def CreateHouse(self, action, controller):
        house = action["house"]
        asset_ids = get_all_asset_ids_recursively(house["objects"], [])
        asset_ids_set = set(asset_ids)
        if not asset_ids_set.issubset(self.last_asset_id_set):
            controller.step(action="DeleteLRUFromProceduralCache", assetLimit=0)
            self.last_asset_id_set = set(asset_ids)

        return create_assets_if_not_exist(
            controller=controller,
            asset_ids=asset_ids,
            asset_directory=self.asset_directory,
            asset_symlink=self.asset_symlink,
            stop_if_fail=self.stop_if_fail,
            copy_to_dir=os.path.join(controller._build.base_dir, self.target_dir),
            load_file_in_unity=True,
        )


PHYSICS_SETTLING_TIME = 1.0

MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER = (
    0.8673349051766235  # Computed with fixed arm agent, should have pairity with real
)

STRETCH_ENV_ARGS = dict(
    gridSize=GRID_SIZE
    * 0.75,  # Intentionally make this smaller than AGENT_MOVEMENT_CONSTANT to improve fidelity
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER,
    visibilityScheme="Distance",
    fieldOfView=INTEL_VERTICAL_FOV,
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=True,
    agentMode="stretch",
    renderDepthImage=False,
    cameraNearPlane=0.01,  # Avoid clipping geometry close to the robot camera.
    branch=None,  # IMPORTANT do not use branch
    commit_id=STRETCH_COMMIT_ID,
    server_timeout=MAXIMUM_SERVER_TIMEOUT,
    snapToGrid=False,
    fastActionEmit=True,
    headless=False,  # Must be False for CloudRendering to produce frames (headless=True disables rendering)
    # antiAliasing="smaa", # We can get nicer looking videos if we turn on antiAliasing and change the quality
    # quality="Ultra",
)

assert (
    STRETCH_ENV_ARGS.get("branch") is None
    and STRETCH_ENV_ARGS.get("commit_id") is not None
), "Should not use branch in STRETCH_ENV_ARGS (use commit_id)."


def get_stretch_env_args():
    """Build controller arguments using the current Objaverse configuration."""
    action_hook_runner = ProceduralAssetHookRunnerResetOnNewHouse(
        asset_directory=str(configured_objaverse_data_dir() / "processed"),
        asset_symlink=True,
        verbose=True,
        asset_limit=200,
    )
    return {**STRETCH_ENV_ARGS, "action_hook_runner": action_hook_runner}


ADDITIONAL_ARM_ARGS = {
    "returnToStart": True,
    "speed": 1,
}

ADDITIONAL_NAVIGATION_ARGS = {
    **ADDITIONAL_ARM_ARGS,
    "returnToStart": False,
}

STRETCH_WRIST_BOUND_1 = 75
STRETCH_WRIST_BOUND_2 = -260
#
# ALL_STRETCH_ACTIONS = [
#     THORActions.move_ahead,
#     THORActions.rotate_right,
#     THORActions.rotate_left,
#     THORActions.move_back,
#     THORActions.done,
#     THORActions.sub_done,
#     THORActions.rotate_left_small,
#     THORActions.rotate_right_small,
#     THORActions.pickup,
#     THORActions.move_arm_in,
#     THORActions.move_arm_out,
#     THORActions.move_arm_up,
#     THORActions.move_arm_down,
#     THORActions.wrist_open,
#     THORActions.wrist_close,
#     THORActions.move_arm_down_small,
#     THORActions.move_arm_in_small,
#     THORActions.move_arm_out_small,
#     THORActions.move_arm_up_small,
#     THORActions.dropoff,
# ]
#
#
# ##
# # actions = [move ahead, move back, left , right, done,...]
#
# stretch_long_names = {
#     THORActions.move_ahead: "move_ahead",
#     THORActions.rotate_right: "rotate_right",
#     THORActions.rotate_left: "rotate_left",
#     THORActions.move_back: "move_back",
#     THORActions.done: "done",
#     THORActions.sub_done: "sub_done",
#     THORActions.rotate_left_small: "rotate_left_small",
#     THORActions.rotate_right_small: "rotate_right_small",
#     THORActions.pickup: "pickup",
#     THORActions.dropoff: "dropoff",
#     THORActions.move_arm_in: "move_arm_in",
#     THORActions.move_arm_out: "move_arm_out",
#     THORActions.move_arm_up: "move_arm_up",
#     THORActions.move_arm_down: "move_arm_down",
#     THORActions.wrist_open: "wrist_open",
#     THORActions.wrist_close: "wrist_close",
#     THORActions.move_arm_down_small: "move_arm_down_small",
#     THORActions.move_arm_in_small: "move_arm_in_small",
#     THORActions.move_arm_out_small: "move_arm_out_small",
#     THORActions.move_arm_up_small: "move_arm_up_small",
# }
#
# robot_action_mapping = {
#     THORActions.move_ahead: {
#         "action": "MoveAgent",
#         "args": {"move_scalar": AGENT_MOVEMENT_CONSTANT},
#     },
#     THORActions.move_back: {
#         "action": "MoveAgent",
#         "args": {"move_scalar": -AGENT_MOVEMENT_CONSTANT},
#     },
#     THORActions.rotate_right: {
#         "action": "RotateAgent",
#         "args": {"move_scalar": AGENT_ROTATION_DEG},
#     },
#     THORActions.rotate_left: {
#         "action": "RotateAgent",
#         "args": {"move_scalar": -AGENT_ROTATION_DEG},
#     },
#     THORActions.rotate_right_small: {
#         "action": "RotateAgent",
#         "args": {"move_scalar": AGENT_ROTATION_DEG / 5},
#     },
#     THORActions.rotate_left_small: {
#         "action": "RotateAgent",
#         "args": {"move_scalar": -AGENT_ROTATION_DEG / 5},
#     },
#     THORActions.done: {"action": "Pass", "args": {}},
#     THORActions.sub_done: {"action": "Pass", "args": {}},
#     THORActions.move_arm_up: {"action": "MoveArmBase", "args": {"move_scalar": ARM_MOVE_CONSTANT}},
#     THORActions.move_arm_up_small: {
#         "action": "MoveArmBase",
#         "args": {"move_scalar": ARM_MOVE_CONSTANT / 5},
#     },
#     THORActions.move_arm_down: {
#         "action": "MoveArmBase",
#         "args": {"move_scalar": -ARM_MOVE_CONSTANT},
#     },
#     THORActions.move_arm_down_small: {
#         "action": "MoveArmBase",
#         "args": {"move_scalar": -ARM_MOVE_CONSTANT / 5},
#     },
#     THORActions.move_arm_out: {
#         "action": "MoveArmExtension",
#         "args": {"move_scalar": ARM_MOVE_CONSTANT},
#     },
#     THORActions.move_arm_out_small: {
#         "action": "MoveArmExtension",
#         "args": {"move_scalar": ARM_MOVE_CONSTANT / 5},
#     },
#     THORActions.move_arm_in: {
#         "action": "MoveArmExtension",
#         "args": {"move_scalar": -ARM_MOVE_CONSTANT},
#     },
#     THORActions.move_arm_in_small: {
#         "action": "MoveArmExtension",
#         "args": {"move_scalar": -ARM_MOVE_CONSTANT / 5},
#     },
#     THORActions.wrist_open: {"action": "MoveWrist", "args": {"move_scalar": -WRIST_ROTATION}},
#     THORActions.wrist_close: {"action": "MoveWrist", "args": {"move_scalar": WRIST_ROTATION}},
#     THORActions.pickup: {"action": "GraspTo", "args": {"move_to": -10}},
#     THORActions.dropoff: {"action": "GraspTo", "args": {"move_to": 30}},
# }
