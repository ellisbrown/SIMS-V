"""
Format of the output json file per scene:
{
    "room_size": float,
    "room_center": [float, float, float],
    "object_counts": {
        "type": int,
        ...
    },
    "object_bbox": [
        # centroid: box centers
        # the extent is along the principal axes of the object
        {
            "centroid": [float, float, float],
            "axesLengths": [float, float, float],
            # rotation matrix
            "normalizedAxes": [float, float, float, float, float, float, float, float, float],
            "min": [float, float, float],
            "max": [float, float, float],
            # additional
            "object_id": str,
            "asset_id": str,
            "object_type": str,
        },
        ...
    ]
}
"""

import copy
import os
import sys
import json
from collections import defaultdict
from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse

import ai2thor.platform
import numpy as np
from shapely.geometry import Polygon

from sims.environment.stretch_controller import StretchController
from sims.utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
    get_stretch_env_args,
)

path_to_this_file = os.path.abspath(__file__)

HOUSE_SPEC_FNAME = "house_spec.json"
METADATA_FNAME = "spatial_metadata.json"
FRAME_FNAME = "top_down_frame.png"
MAP_FNAME = "top_down_map.png"


def get_top_down_frame(controller):
    # Setup the top-down camera
    event = controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
    pose = copy.deepcopy(event.metadata["actionReturn"])

    bounds = event.metadata["sceneBounds"]["size"]
    max_bound = max(bounds["x"], bounds["z"])

    pose["fieldOfView"] = 50
    pose["position"]["y"] += 1.1 * max_bound
    pose["orthographic"] = False
    pose["farClippingPlane"] = 50
    del pose["orthographicSize"]

    # add the camera to the scene
    event = controller.step(
        action="AddThirdPartyCamera",
        **pose,
        skyboxColor="white",
        raise_for_failure=True,
    )
    top_down_frame = event.third_party_camera_frames[-1]
    return Image.fromarray(top_down_frame)


def get_top_down_map(house_json):
    # Extract room polygons and labels based on `roomType` and `id`
    rooms_with_labels = []

    # Iterate over objects to find room polygons and labels
    for room in house_json.get("rooms", []):
        if "floorPolygon" in room and "roomType" in room and "id" in room:
            room_polygon = [
                (point["x"], point["z"])  # Use x and z for the top-down view
                for point in room["floorPolygon"]
            ]
            room_label = f"{room['roomType']} ({room['id']})"
            rooms_with_labels.append((room_polygon, room_label))

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot rooms with labels
    for room_polygon, room_label in rooms_with_labels:
        x, z = zip(*room_polygon)
        ax.fill(x, z, alpha=0.5, label=room_label)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Z Coordinate")
    ax.set_title("Top-Down Map View of Rooms")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.axis("equal")

    # return the figure
    return fig


def get_house_measurements(house_json):
    # Calculate the total area of the house
    total_area = 0.0

    # Calculate the center of the house using the centroid of all room polygons
    house_polygon = Polygon()

    # Iterate over the rooms and calculate the area using Shapely polygons
    # Merge all room polygons into a single shapely polygon

    # room_id -> area, type, floor_material
    room_data = {}

    # Iterate over objects to find room polygons and labels
    rooms = house_json.get("rooms", [])
    for room in rooms:
        floor_poly = room.get("floorPolygon", [])
        if not floor_poly:
            continue

        room_polygon = [
            (point["x"], point["z"])  # Use x and z for the top-down view
            for point in floor_poly
        ]

        polygon = Polygon(room_polygon)

        room_id = room.get("id")
        room_area = polygon.area
        room_bounds = polygon.bounds
        min_x, min_z, max_x, max_z = room_bounds
        length = abs(max_x - min_x)
        width = abs(max_z - min_z)

        if room_area - (length * width) > 1e-6:
            raise ValueError(
                f"Room area mismatch: polygon area {room_area} != "
                f"bounding box area {length * width} (length={length}, width={width})"
            )

        room_type = room.get("roomType")
        room_material = room.get("floorMaterial", "")
        if isinstance(room_material, dict):
            room_material = room_material.get("name", "")

        room_data[room_id] = {
            "area": room_area,
            "length": length,
            "width": width,
            "type": room_type,
            "floor_material": room_material,
        }

        total_area += polygon.area
        house_polygon = house_polygon.union(polygon)

    # Calculate the centroid
    house_center = house_polygon.centroid.coords[0]
    return house_center, total_area, room_data


def objaverse_object_id_to_type(obj_id: str) -> str:
    """Convert Objaverse object ID to object type.
    Example: "ObjaTankStorageVessel|2|5" -> "TankStorageVessel
    """
    return obj_id.split("|")[0].replace("Obja", "")


def transform_house_data(metadata, house_json):
    """Transform AI2-THOR metadata and house JSON into the requested format."""
    object_counts = defaultdict(int)
    object_bbox = list()

    # Calculate room dimensions using Shapely
    house_polygon = Polygon()
    for room in house_json.get("rooms", []):
        if "floorPolygon" not in room:
            continue

        room_polygon = [(point["x"], point["z"]) for point in room["floorPolygon"]]
        polygon = Polygon(room_polygon)
        house_polygon = house_polygon.union(polygon)

    # Get room size and centerget_house_measurements
    house_center, house_area, room_data = get_house_measurements(house_json)

    # Process objects from metadata
    for obj in metadata["objects"]:
        obj_type = obj["objectType"]
        obj_id = obj["objectId"]
        asset_id = obj["assetId"]
        bbox = obj["axisAlignedBoundingBox"]

        # Objaverse objects have 'Undefined' type. Derive type from object ID
        if obj_type == "Undefined" and obj_id.startswith("Obja"):
            obj_type = objaverse_object_id_to_type(obj_id)

        object_counts[obj_type] += 1
        corners = np.array(bbox["cornerPoints"])

        object_bbox.append(
            {
                "centroid": [
                    bbox["center"]["x"],
                    bbox["center"]["y"],
                    bbox["center"]["z"],
                ],
                "axesLengths": [
                    bbox["size"]["x"],
                    bbox["size"]["y"],
                    bbox["size"]["z"],
                ],
                "normalizedAxes": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                # "cornerPoints": corners.tolist(),
                "min": corners.min(axis=0).tolist(),
                "max": corners.max(axis=0).tolist(),
                "object_id": obj_id,
                "asset_id": asset_id,
                "object_type": obj_type,
            }
        )

    # Convert defaultdict to regular dict
    object_counts = dict(object_counts)

    return {
        "house_area": house_area,
        "house_center": house_center,
        "room_data": room_data,
        "object_counts": object_counts,
        "object_bbox": object_bbox,
    }


def process_house(
    controller: StretchController,
    house_json_path: str,
    retries: int = 3,
    save: bool = True,
    overwrite: bool = False,
    save_images: bool = False,
):
    """Process a single house scene to generate spatial metadata.

    Args:
        controller (StretchController): The simulator controller.
        house_json_path (str): Path to the house JSON file.
        retries (int): Number of retries for metadata extraction.
        save (bool): Whether to save the generated data to disk.
        overwrite (bool): Whether to overwrite existing files.
        save_images (bool): Whether to save the top-down frame and map.
    """
    if not os.path.exists(house_json_path):
        raise FileNotFoundError(f"House spec file {house_json_path} not found.")

    scene_dir = os.path.dirname(house_json_path)

    # Define paths for the outputs
    metadata_path = os.path.join(scene_dir, METADATA_FNAME)
    frame_path = os.path.join(scene_dir, FRAME_FNAME)
    map_path = os.path.join(scene_dir, MAP_FNAME)

    # Skip processing if all outputs already exist and overwrite is False
    if (
        not overwrite
        and os.path.exists(metadata_path)
        and (
            not save_images
            or all([os.path.exists(path) for path in [frame_path, map_path]])
        )
    ):
        print(f"Skipping {scene_dir}: All outputs already exist.")
        return None

    if save and overwrite:
        # Once an overwrite attempt starts, old outputs are no longer valid
        # evidence for this run. Remove them up front so a failed partial run
        # cannot feed stale metadata into downstream QA generation.
        paths_to_replace = [metadata_path]
        if save_images:
            paths_to_replace.extend([frame_path, map_path])
        for output_path in paths_to_replace:
            if os.path.exists(output_path):
                os.remove(output_path)

    # Load the house JSON
    with open(house_json_path, "r") as file:
        house_json = json.load(file)

    # Reset controller with scene and get metadata
    controller.reset(scene=house_json)
    metadata = None
    for i in range(retries):
        print(f"[Try #{i + 1}] Extracting metadata for {scene_dir}...")
        with controller.include_object_metadata_context():
            event = controller.step(action="Pass")
            metadata = event.metadata
        if (
            metadata is not None
            and "objects" in metadata
            and len(metadata["objects"]) > 0
        ):
            break

    # Transform the metadata and house JSON into the requested format
    house_data = transform_house_data(metadata, house_json)
    if house_data["object_counts"] is None or len(house_data["object_counts"]) < 1:
        raise RuntimeError(f"Error processing {house_json_path}: No objects returned.")

    if not save:
        return house_data

    # Save the metadata to a JSON file
    if overwrite or not os.path.exists(metadata_path):
        tmp_metadata_path = f"{metadata_path}.tmp-{os.getpid()}"
        try:
            with open(tmp_metadata_path, "w") as file:
                json.dump(house_data, file, indent=2)
            os.replace(tmp_metadata_path, metadata_path)
        finally:
            if os.path.exists(tmp_metadata_path):
                os.remove(tmp_metadata_path)

    # Save top-down frame + map
    if save_images:
        if overwrite or not os.path.exists(frame_path):
            top_down_frame = get_top_down_frame(controller)
            top_down_frame.save(frame_path)

        if overwrite or not os.path.exists(map_path):
            top_down_map = get_top_down_map(house_json)
            top_down_map.savefig(map_path)

        print(f"""Processed {scene_dir}:
        - Saved top-down frame to {frame_path}
        - Saved top-down map to {map_path}
        - Saved metadata to {metadata_path}""")
    else:
        print(f"Processed {scene_dir}: Saved metadata to {metadata_path}")

    return house_data


def main(
    dataset_dir,
    split="val",
    resolution_scale: int = 1,
    overwrite: bool = False,
    save_images=False,
):
    controller_args = {
        **get_stretch_env_args(),
        "width": int(INTEL_CAMERA_WIDTH * resolution_scale),
        "height": int(INTEL_CAMERA_HEIGHT * resolution_scale),
    }

    # Use CloudRendering on Linux with GPU (Linux64 builds don't exist)
    import torch

    if torch.cuda.is_available() and sys.platform != "darwin":
        controller_args["platform"] = ai2thor.platform.CloudRendering

    controller = StretchController(True, **controller_args)

    # REQUIRED TO WORK WITH THE OBJAVERSE SCENES
    print("Running the AdvancePhysicsStep...")
    controller.step(
        action="AdvancePhysicsStep",
        simSeconds=2.0,
        raise_for_failure=True,
    )
    print("Controller ready.")

    # Iterate over all scene directories
    split_dir = os.path.join(dataset_dir, split)
    failed = []
    all_scenes = sorted(os.listdir(split_dir))
    for scene_dir in tqdm(all_scenes, desc="Scenes"):
        scene_path = os.path.join(split_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue

        try:
            house_json_path = os.path.join(scene_path, HOUSE_SPEC_FNAME)
            process_house(
                controller,
                house_json_path,
                overwrite=overwrite,
                save_images=save_images,
            )
        except Exception as e:
            print(f"Error processing house {scene_dir}: {e}")
            failed.append(scene_dir)
            continue

    print(f"Finished processing all scene directories in {split_dir}.")
    if len(failed) > 0:
        print(f"{len(failed)}/{len(all_scenes)} scenes failed to generate:\n{failed}")
    else:
        print(f"All {len(all_scenes)} scenes generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Spatial Metatadata for a house."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="The split to process (train/val/test).",
    )
    parser.add_argument(
        "--resolution_scale",
        type=int,
        default=1,
        help="Resolution scale for the output images.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing metadata files."
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Save images of the house."
    )

    args = parser.parse_args()
    print(f"Parsed arguments: {args.__dict__}")

    main(
        args.dataset_dir,
        args.split,
        args.resolution_scale,
        args.overwrite,
        args.save_images,
    )
