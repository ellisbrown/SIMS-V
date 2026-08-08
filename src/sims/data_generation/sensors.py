import json
from typing import Any, Dict, Optional

import ai2thor
import ai2thor.controller
import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import EnvType, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super

from sims.environment.stretch_controller import StretchController
from sims.tasks.abstract_task import AbstractSimsTask
from sims.utils.string_utils import convert_string_to_byte
from sims.utils.visualization_utils import add_bbox_sequence_to_frame_sequence


class RawRGBSensorTHOR(Sensor):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        if isinstance(env, ai2thor.controller.Controller):
            return env.last_event.frame.copy()
        else:
            return env.current_frame.copy()


class RawRGBWDet(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int, rgb_sensor, det_sensor):
        self.height = height
        self.width = width
        super().__init__(uuid=uuid, height=height, width=width)
        self.rgb_sensor = rgb_sensor
        self.det_sensor = det_sensor

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        rgb_image = self.rgb_sensor.get_observation(env, task, *args, **kwargs)
        det_bbox = self.det_sensor.get_observation(env, task, *args, **kwargs)

        new_rgb_image = add_bbox_sequence_to_frame_sequence(
            rgb_image[np.newaxis, :, :, :], det_bbox[np.newaxis, :]
        )[0]

        return new_rgb_image


class RawNavigationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.navigation_camera.copy()


class RawManipulationStretchRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        return env.manipulation_camera.copy()


class TopDownPathViewRGBSensor(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self, env: EnvType, task: Optional[SubTaskType], *args: Any, **kwargs: Any
    ) -> Any:
        target_ids = None  # task.task_info.get("synset_to_object_ids", None)
        return env.get_top_down_path_view(
            task.task_info["followed_path"], target_ids
        ).copy()


# Offline annotation and visualization sensors


class OfflineAnnoSensor(Sensor):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width
        observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        super().__init__(**prepare_locals_for_super(locals()))

    @staticmethod
    def npify(obj, max_len=25_000):
        # return np.frombuffer(json.dumps({
        #     key: (value if isinstance(value, np.ndarray) else json.dumps(value))
        #     for key, value in obj.items()
        # }).encode('utf-8'), dtype=np.uint8)

        json_str = json.dumps(obj)
        if len(json_str) > max_len:
            raise ValueError(f"String too long: {len(json_str)}")
        return convert_string_to_byte(json_str, max_len)

    def get_observation(  # type:ignore
        self,
        env: StretchController,
        task: AbstractSimsTask,
        *args,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        event = env.controller.last_event

        # render instance segmentation if not already done
        # REF: environment/stretch_controller.py:navigation_camera_segmentation()
        if event.instance_segmentation_frame is None:
            env.controller.step("Pass", renderImageSynthesis=True)
            event = env.controller.last_event
            assert event.instance_segmentation_frame is not None, (
                "Must pass `renderInstanceSegmentation=True` on initialization"
                " to obtain a navigation_camera_segmentation"
            )

        objs = env.get_objects()
        obj_map = {obj["objectId"]: obj for obj in objs}

        time = event.metadata["currentTime"]
        obj_annos = {}

        # instance_masks = env.navigation_camera_segmentation
        instance_masks = event.instance_masks

        if not instance_masks:
            # something must be visible. otherwise, issue
            raise RuntimeError("No instance masks found")

        for key, instance_mask in instance_masks.items():
            if key in obj_map:
                obj = obj_map[key]
                num_pixels = np.sum(instance_mask)
                total_pixels = instance_mask.shape[0] * instance_mask.shape[1]
                pct_pixels = num_pixels / total_pixels
                obj_annos[key] = {
                    "num_pixels": int(num_pixels),
                    "pct_pixels": float(pct_pixels),
                    "visible": obj["visible"],
                    "is_objaverse": obj["isObjaverse"],
                    "distance": obj["distance"],
                    "object_type": obj["objectType"],
                    "asset_id": obj["assetId"],
                    "synset": obj["synset"],
                    "salientMaterials": obj["salientMaterials"],
                    "parent_receptacles": obj["parentReceptacles"],
                }
            elif key.startswith("Ceiling"):
                # Ignore ceiling masks
                continue
            else:
                print(
                    f"Object {key} not found in metadata. Metadata keys: {obj_map.keys()}"
                )

        return {
            "time": time,
            "agent": self.npify(
                {
                    "position": event.metadata["agent"]["position"],
                    "rotation": event.metadata["agent"]["rotation"],
                }
            ),
            "objects": self.npify(obj_annos),
        }


### Save Other Visualization Types


def get_depth_image_fixed_scale(
    depth_frame, inverted=False, lower_bound=0.1, upper_bound=5, add_outliers=False
):
    # Clip the depth values to remove outliers
    depth_frame_clipped = np.clip(depth_frame, lower_bound, upper_bound)
    depth_normalized = (depth_frame_clipped - lower_bound) / (upper_bound - lower_bound)
    if inverted:
        depth_normalized = 1 - depth_normalized
    depth_image_array = (depth_normalized * 255).astype(np.uint8)

    if not add_outliers:
        return depth_image_array

    # Add back outliers in red for upper and green for lower
    upper_outliers_mask = depth_frame > upper_bound
    lower_outliers_mask = depth_frame < lower_bound
    depth_image_array[upper_outliers_mask] = [255, 0, 0]  # Red color for upper outliers
    depth_image_array[lower_outliers_mask] = [
        0,
        255,
        0,
    ]  # Green color for lower outliers

    return depth_image_array


class DepthSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        depth_frame = event.depth_frame.copy()
        return get_depth_image_fixed_scale(depth_frame, inverted=True)


class SemanticSegmentationSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return event.semantic_segmentation_frame.copy()


class InstanceSegmentationSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return event.instance_segmentation_frame.copy()


# Function to generate an edge overlay with optional color
def get_edge_overlay(event, use_color=False, square_size=3, non_overlapping=False):
    import numpy as np
    from skimage.morphology import erosion, square
    from matplotlib.colors import to_rgb

    instance_masks = event.class_masks.instance_masks

    # Get the dimensions from one of the masks (assumed to be grayscale)
    height, width = next(iter(instance_masks.values())).shape

    # Use uint8 for image data so that matplotlib displays correctly
    edge_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for obj_id, mask in instance_masks.items():
        mask_bool = mask.astype(bool)

        if non_overlapping:
            # Shrink the mask to prevent overlap
            mask_bool = erosion(mask_bool, square(2))

        # Compute the edge using a morphological operation
        edges = mask_bool ^ erosion(mask_bool, square(square_size))
        if use_color:
            # Get the object's color (assumes it's a hex string or a tuple)
            color = event.object_id_to_color[obj_id]
            try:
                rgb = to_rgb(color)
            except Exception:
                rgb = color  # assume it's already in RGB format
            edge_overlay[edges] = rgb
        else:
            # White edges for grayscale output
            edge_overlay[edges] = (255, 255, 255)

    return edge_overlay


class EdgeSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return get_edge_overlay(event)


class ColoredEdgeSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return get_edge_overlay(event, use_color=True)


class NonOverlappingColoredEdgeSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return get_edge_overlay(event, use_color=True, non_overlapping=True)


def apply_mask_avg_color(event, method="mean", alpha=1.0):
    """
    For each instance mask in the event, compute the average color (mean or median)
    from the corresponding region of the RGB image and overwrite that region with the computed color.

    Parameters:
        event: An event object with attributes:
            - frame: the RGB image as a NumPy array.
            - class_masks.instance_masks: dictionary of masks.
        method: 'mean' or 'median' to specify the aggregation method.
    Returns:
        Modified RGB image with each mask region colored with its aggregated color.
    """
    import numpy as np

    # Make a copy of the rgb image to avoid modifying the original
    rgb = event.frame.copy()
    instance_masks = event.class_masks.instance_masks

    for mask in instance_masks.values():
        mask_bool = mask.astype(bool)

        # Compute the aggregated color for each channel (R, G, B)
        if method == "mean":
            avg_color = np.array([np.mean(rgb[..., c][mask_bool]) for c in range(3)])
        elif method == "median":
            avg_color = np.array([np.median(rgb[..., c][mask_bool]) for c in range(3)])
        else:
            raise ValueError("Invalid method specified. Use 'mean' or 'median'.")

        # Ensure the color is an integer value in the uint8 range
        avg_color = np.round(avg_color).astype(np.uint8)

        # Overwrite the mask region with the computed average color
        rgb[mask_bool] = avg_color

    raw = event.frame.copy()
    blended = (
        (1 - alpha) * raw.astype(np.float32) + alpha * rgb.astype(np.float32)
    ).astype(np.uint8)

    return blended


class MeanMaskOverlaySensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return apply_mask_avg_color(event, method="mean")


def remove_masked_rgb(event, keywords=["wall", "room"]):
    """
    For each instance mask in the event whose key includes "floor", "wall" or "ceiling"
    (case insensitive), set the corresponding RGB region to zero (black).

    Parameters:
        event: An event object with attributes:
            - frame: the RGB image as a NumPy array.
            - class_masks.instance_masks: dictionary of masks with keys as identifiers.
        keywords: List of keywords to match in the mask key names.
    Returns:
        Modified RGB image with specified mask regions zeroed out.
    """
    import numpy as np

    rgb = event.frame.copy()
    instance_masks = event.class_masks.instance_masks

    for key, mask in instance_masks.items():
        if any(keyword in key.lower() for keyword in keywords):
            mask_bool = mask.astype(bool)
            rgb[mask_bool] = np.array([0, 0, 0])

    return rgb


class MaskedBackgroundSensorTHOR(RawRGBSensorTHOR):
    def __init__(self, uuid: str, height: int, width: int):
        self.height = height
        self.width = width

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchController,
        task: Optional[SubTaskType],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # if not isinstance(env, ai2thor.controller.Controller):
        #     raise NotImplementedError
        event = env.controller.last_event
        return remove_masked_rgb(event)
