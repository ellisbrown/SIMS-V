import copy
import traceback
from typing import Sequence, Dict
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np
import torch

from sims.environment.stretch_controller import StretchController
from sims.utils.bounding_box_processing_utils import BBOX_COORDINATE_DUMMY_VALUE
from sims.utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
)
from sims.utils.pointing_utils import unnormalize_point

DISTINCT_COLORS = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),  # Dark Red
    (0, 128, 0),  # Dark Green
    (0, 0, 128),  # Dark Blue
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203),  # Pink
    (255, 255, 255),  # White
    (0, 0, 0),  # Black
    (0, 0, 139),  # DarkBlue
    (0, 100, 0),  # DarkGreen
    (139, 0, 139),  # DarkMagenta
    (165, 42, 42),  # Brown
    (255, 215, 0),  # Gold
    (64, 224, 208),  # Turquoise
    (240, 230, 140),  # Khaki
    (70, 130, 180),  # Steel Blue
]


def add_bboxes_to_frame(
    frame: np.ndarray,
    bboxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    inplace=False,
    colors=DISTINCT_COLORS,
    thinkness=1,
):
    """
    Visualize bounding boxes on an image and save the image to disk.

    Parameters:
    - frame: numpy array of shape (height, width, 3) representing the image.
    - bboxes: list of bounding boxes. Each bounding box is a list of [min_row, min_col, max_row, max_col].
    - labels: list of labels corresponding to each bounding box.
    - inplace: whether to modify the input frame in place or return a new frame.
    """
    # Convert numpy image to PIL Image for visualization

    assert frame.dtype == np.uint8
    if not inplace:
        frame = copy.deepcopy(frame)

    bboxes_cleaned = [[int(v) for v in bbox] for bbox in bboxes if -1 not in bbox]
    if labels is None:
        labels = ["" for i in range(len(bboxes_cleaned))]

    h, w, _ = frame.shape

    # Plot bounding boxes and labels
    for bbox, label, color in zip(bboxes_cleaned, labels, colors):
        if np.all(bbox == 0):
            continue
        cv2.rectangle(frame, bbox[:2], bbox[2:], color=color, thickness=thinkness)

        cv2.putText(
            frame,
            label,
            (int(bbox[0]), int(bbox[1] + 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=color,
            thickness=2,
        )

    return frame


def add_bbox_sequence_to_frame_sequence(frames, double_bboxes):
    T, num_coords = double_bboxes.shape
    assert num_coords == 10
    assert T == len(frames)

    convert_to_torch = False
    if torch.is_tensor(frames):
        frames = frames.numpy()
        convert_to_torch = True

    double_bboxes[double_bboxes == 1000] = 0

    for i, frame in enumerate(frames):
        bbox_list = [double_bboxes[i][:4], double_bboxes[i][5:9]]
        add_bboxes_to_frame(
            frame,
            bbox_list,
            labels=None,
            inplace=True,
            colors=[(255, 0, 0), (0, 255, 0)],
            thinkness=2,
        )
    if convert_to_torch:
        result = torch.Tensor(frames).to(torch.uint8)
    else:
        result = frames
    return result


# from sims.utils.data_generation_utils.mp4_utils import save_frames_to_mp4; save_frames_to_mp4(frames, 'after_changes.mp4', fps=20)


def add_bbox_sensor_to_image(
    curr_frame, task_observations, det_sensor_key, which_image
):
    task_relevant_object_bbox = task_observations[det_sensor_key]
    (bbox_dim,) = task_relevant_object_bbox.shape
    assert bbox_dim in [5, 10]
    if bbox_dim == 5:
        task_relevant_object_bboxes = [task_relevant_object_bbox[:4]]
    if bbox_dim == 10:
        task_relevant_object_bboxes = [
            task_relevant_object_bbox[:4],
            task_relevant_object_bbox[5:9],
        ]
        task_relevant_object_bboxes = [
            b for b in task_relevant_object_bboxes if b[1] <= curr_frame.shape[0]
        ]
    if which_image == "nav":
        pass
    elif which_image == "manip":
        start_index = curr_frame.shape[1] // 2
        for i in range(len(task_relevant_object_bboxes)):
            task_relevant_object_bboxes[i][0] += start_index
            task_relevant_object_bboxes[i][2] += start_index
    else:
        raise NotImplementedError
    if len(task_relevant_object_bboxes) > 0:
        # This works because the navigation frame comes first in curr_frame
        add_bboxes_to_frame(
            frame=curr_frame,
            bboxes=task_relevant_object_bboxes,
            labels=None,
            inplace=True,
        )


POINTS_COLOR_CODES = {
    "goal_in_camera_2d_first_step": (255, 0, 0),
    "repeat_goal_in_camera_2d": (0, 255, 0),
    "nav_goal_point": (0, 0, 255),
    "manip_goal_point": (0, 0, 255),
    "task_info_points": (255, 255, 0),
}


def add_point_on_frame(
    frame,
    point,
    which_image,
    color=(255, 0, 0),
    radius=5,
    point_is_normalized=True,
):
    """
    Add a point to the frame.

    Parameters:
    - frame: numpy array of shape (height, width, 3) representing the image.
    - point: tuple of (x, y) coordinates.
    - color: tuple of (R, G, B) values.
    - radius: radius of the point.
    """
    assert which_image in ["nav", "manip"]
    assert frame.dtype == np.uint8

    assert len(point) == 2
    if (
        point[0] == BBOX_COORDINATE_DUMMY_VALUE
        or point[1] == BBOX_COORDINATE_DUMMY_VALUE
    ):
        return frame
    if point_is_normalized:
        assert 0 <= point[0] <= 1 and 0 <= point[1] <= 1
        unnormalized_point = unnormalize_point(point)
    else:
        assert isinstance(point[0], (int, np.integer)), (
            f"point[0] is not an int {(point[0], type(point[0]))}"
        )
        assert isinstance(point[1], (int, np.integer)), (
            f"point[1] is not an int {(point[1], type(point[1]))}"
        )
        unnormalized_point = point

    try:
        if which_image == "manip":
            # Adjust x-coordinate for manipulation camera (right half of the frame)
            unnormalized_point = (
                unnormalized_point[0] + INTEL_CAMERA_WIDTH,
                unnormalized_point[1],
            )

        if (
            unnormalized_point[1] < 1
            or unnormalized_point[0] < 0
            or unnormalized_point[0] > frame.shape[1]
            or unnormalized_point[1] > INTEL_CAMERA_HEIGHT
        ):
            print(
                "WARNING POINT IS OUT OF DOMAIN",
                unnormalized_point,
                "before unnormalization",
                point,
            )
        cv2.circle(
            frame, (unnormalized_point[0], unnormalized_point[1]), radius, color, -1
        )
    except Exception:
        print(traceback.format_exc())
        print("Failed to add point to frame")
        print("point", point)
        print("unnormalized_point", unnormalized_point)
        pass

    return frame


def get_top_down_path_view(
    controller: StretchController,
    agent_path: Sequence[Dict[str, float]],
    targets_to_highlight=None,
    orthographic: bool = True,
    map_height_width=(1000, 1000),
    path_width: float = 0.045,
):
    thor_controller = controller.controller

    original_hw = thor_controller.last_event.frame.shape[:2]

    if original_hw != map_height_width:
        event = thor_controller.step(
            "ChangeResolution",
            x=map_height_width[1],
            y=map_height_width[0],
            raise_for_failure=True,
        )

    if len(thor_controller.last_event.third_party_camera_frames) < 2:
        event = thor_controller.step(
            "GetMapViewCameraProperties", raise_for_failure=True
        )
        cam = copy.deepcopy(event.metadata["actionReturn"])
        if not orthographic:
            bounds = event.metadata["sceneBounds"]["size"]
            max_bound = max(bounds["x"], bounds["z"])

            cam["fieldOfView"] = 50
            cam["position"]["y"] += 1.1 * max_bound
            cam["orthographic"] = False
            cam["farClippingPlane"] = 50
            del cam["orthographicSize"]

        event = thor_controller.step(
            action="AddThirdPartyCamera",
            **cam,
            skyboxColor="white",
            raise_for_failure=True,
        )

    waypoints = []
    for target in targets_to_highlight or []:
        target_position = controller.get_object_position(target)
        target_dict = {
            "position": target_position,
            "color": {"r": 1, "g": 0, "b": 0, "a": 1},
            "radius": 0.5,
            "text": "",
        }
        waypoints.append(target_dict)

    if len(agent_path) != 0:
        thor_controller.step(
            action="VisualizeWaypoints",
            waypoints=waypoints,
            raise_for_failure=True,
        )
        # put this over the waypoints just in case
        event = thor_controller.step(
            action="VisualizePath",
            positions=agent_path,
            pathWidth=path_width,
            raise_for_failure=True,
        )
        thor_controller.step({"action": "HideVisualizedPath"})

    map = event.third_party_camera_frames[-1]

    if original_hw != map_height_width:
        thor_controller.step(
            "ChangeResolution",
            x=original_hw[1],
            y=original_hw[0],
            raise_for_failure=True,
        )

    return map


def create_multiline_text_image(
    text,
    width,
    height,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_path=None,
    font_size=20,
):
    # Create a blank image with the background color
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Load a font
    font = ImageFont.truetype(font_path if font_path else "Arial.ttf", font_size)

    # Split the text into lines
    lines = text.split("\n")
    line_heights = [30 for line in lines]
    total_text_height = (
        sum(line_heights) + (len(lines) - 1) * 10
    )  # add spacing between lines

    # Calculate initial y position (top margin)
    y = (height - total_text_height) / 2

    for line in lines:
        # Calculate text width and height using textbbox for precise bounding box
        text_width, text_height = draw.textbbox((0, 0), line, font=font)[2:]

        # Calculate x position (left margin)
        x = (width - text_width) / 2

        # Draw each line
        draw.text((x, y), line, font=font, fill=text_color)
        y += text_height + 10  # move y to the next line position with spacing

    # Convert the PIL Image to a numpy array
    numpy_image = np.array(image)

    return numpy_image
