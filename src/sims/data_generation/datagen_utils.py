import json
import os
import string
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List, Sequence
import hashlib
import canonicaljson
import filelock
import h5py
import imageio
import numpy as np
import torch
from allenact.base_abstractions.sensor import SensorSuite
from omegaconf import OmegaConf

from sims.environment.action_spaces import AbstractActionSpace
from sims.utils.constants.stretch_initialization_utils import (
    cropped_stretch_camera_width,
)
from sims.data_generation.queues import QueueMessage
from sims.utils.data_generation_utils.mp4_utils import save_frames_to_mp4
from sims.utils.string_utils import convert_byte_to_string

try:
    from allenact.base_abstractions.sensor import SensorSuite
    from allenact.embodiedai.sensors.vision_sensors import RGBSensor
except ImportError:
    print(
        "AllenAct not installed, skipping imports from allenact, this may mean that some things will break."
    )

from sims.data_generation.sensors import (
    RawRGBSensorTHOR,
    OfflineAnnoSensor,
)


GENERATION_CONFIG_SCHEMA_VERSION = 4
PACKAGE_ROOT = str(Path(__file__).resolve().parents[1])


def _qualified_class_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def build_generation_config(
    args: Any,
    action_space: AbstractActionSpace,
    *,
    task_type: str,
    width: int,
    height: int,
    video_modalities: Sequence[str],
) -> OmegaConf:
    """Build the stable, output-defining configuration for a dataset.

    Worker count and maximum house count are intentionally omitted: they change
    scheduling or the size of a generated prefix, not the contents of a given
    house trajectory.
    """
    return OmegaConf.create(
        {
            "schema_version": GENERATION_CONFIG_SCHEMA_VERSION,
            "task": {
                "type": task_type,
                "max_steps": args.max_steps,
                "trajectories_per_house": args.trajectories_per_house,
            },
            "source": {
                "house_dataset": args.house_dataset,
            },
            "action_space": {
                "id": "discrete-stretch-v1",
                "class": _qualified_class_name(action_space),
            },
            "rendering": {
                "resolution_scale": args.resolution_scale,
                "controller_width": width,
                "controller_height": height,
                "raw_camera_width": cropped_stretch_camera_width(width),
                "raw_camera_height": height,
                "quality": args.quality,
                "video_modalities": list(video_modalities),
            },
            "randomization": {
                "material_probability": args.material_randomization_probability,
                "rotation_noise_std_degrees": args.rotation_noise_std_degrees,
            },
        }
    )


def split_house_repeats_to_id(split: str, house: int, repeats: int):
    return f"split_{split}__house_{house:08d}__repeats_{repeats:02d}"


def recurse_write(group: h5py.Group, data: Dict[str, Any], should_keep):
    for k, v in data.items():
        if isinstance(v, Dict):
            recursively_write_to_hdf5(group=group.create_group(k), data=v)
        else:
            group.create_dataset(
                k,
                data=v.type(torch.float16 if v.dtype == torch.float32 else v.dtype)
                .cpu()
                .numpy()[should_keep, ...],
                compression="gzip",
                compression_opts=6,
            )


def skip_keys(obs, keys):
    obs_updated = {}
    for key in obs.keys():
        if key in keys:
            continue
        obs_updated[key] = obs[key]
    return obs_updated


def write_config_file(cfg: OmegaConf, top_level_save_dir: str):
    os.makedirs(top_level_save_dir, exist_ok=True)

    config_info_save_path = os.path.join(top_level_save_dir, "constants.yaml")

    with filelock.FileLock(os.path.join(top_level_save_dir, ".lock")):
        config_info = OmegaConf.to_container(cfg=cfg, resolve=True)
        canonical_encoding = canonicaljson.encode_canonical_json(config_info).decode(
            "utf-8"
        )

        if os.path.exists(config_info_save_path):
            with open(config_info_save_path, "r") as f:
                saved_info = f.read().strip()
            if saved_info != canonical_encoding.strip():
                raise ValueError(
                    "Generation configuration does not match the existing dataset. "
                    f"Delete or choose a different output directory: {config_info_save_path}. "
                    f"Saved: {saved_info}; current: {canonical_encoding}."
                )
        else:
            with open(config_info_save_path, "w") as f:
                f.write(canonical_encoding)


def parse_queue_message(message: "QueueMessage", expected_split: str):
    id = message.body.strip()
    assert set(id) <= set(string.ascii_letters + string.digits + "_"), (
        f"Invalid ID '{id}', id must only contain a-Z, 0-9, and _"
    )

    # Parse the ID into split, house, repeats
    # Use maxsplit=1 to handle splits like "train_unseen" that contain underscores
    run_info = {
        k: v for part in message.body.split("__") for k, v in [part.split("_", 1)]
    }

    split = run_info["split"]
    assert expected_split == split
    house_index = int(run_info["house"])
    repeats = int(run_info["repeats"])
    assert len(run_info) == 3
    assert split in ["train", "train_unseen", "val", "test"]

    return {
        "id": id,
        "split": split,
        "house_index": int(house_index),
        "repeats": int(repeats),
    }


def recursively_write_to_hdf5(
    group: h5py.Group,
    data: Dict[str, Any],
    compression="gzip",
    compression_opts=6,
):
    for k, v in data.items():
        if isinstance(v, Dict):
            recursively_write_to_hdf5(
                group=group.create_group(k),
                data=v,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            group.create_dataset(
                k,
                data=v,
                compression=compression,
                compression_opts=6,
            )


def save_trajectories(
    observations_list: List[Dict[str, Any]],
    sensor_suite: SensorSuite,
    save_dir: str,
    save_file_suffix: str = "",
    extra_obs_keys_to_save: List[str] = None,
    save_mp4s: bool = True,
):
    saved_paths = []
    try:
        sensor_uuids_for_hdf5 = []
        for sensor in sensor_suite.sensors.values():
            assert all(
                sensor.uuid in observations for observations in observations_list
            ), (
                f"All observations must be present in the `observation_list`.\nSensors in sensor suite: "
                f" {[sensor.uuid for sensor in sensor_suite.sensors.values()]}."
                f"\nSensors in observation lists:"
                f"\n{[list(observations.keys()) for observations in observations_list]}."
            )

            if isinstance(sensor, RGBSensor):
                raise NotImplementedError(
                    "To save RGB videos please use the RawRGBSensorTHOR sensor rather than an RGBSensor"
                )

            elif isinstance(sensor, RawRGBSensorTHOR):
                if save_mp4s:
                    for i, observations in enumerate(observations_list):
                        rgb_save_path = os.path.join(
                            save_dir, f"{sensor.uuid}__{i}{save_file_suffix}.mp4"
                        )
                        saved_paths.append(rgb_save_path)

                        assert observations[sensor.uuid].dtype == torch.uint8
                        save_frames_to_mp4(
                            observations[sensor.uuid].cpu().numpy(),
                            file_path=rgb_save_path,
                            fps=10,
                        )
            elif isinstance(sensor, OfflineAnnoSensor):
                # save as jsonl file, one line per observation
                for i, observations in enumerate(observations_list):
                    sensor_obs = observations[sensor.uuid]
                    # print(f"[{i}] Saving offline annotations for sensor {sensor.uuid}, {len(sensor_obs)} annotations, type {type(sensor_obs)}\n{sensor_obs}")

                    jsonl_save_path = os.path.join(
                        save_dir, f"{sensor.uuid}__{i}{save_file_suffix}.jsonl"
                    )
                    saved_paths.append(jsonl_save_path)

                    times = sensor_obs["time"]
                    agents = sensor_obs["agent"]
                    objects = sensor_obs["objects"]

                    assert len(times) == len(objects) == len(agents), (
                        f"Length mismatch between times, objects, and agent: {len(times)}, {len(objects)}, {len(agents)}"
                    )

                    with open(jsonl_save_path, "a") as f:
                        for i in range(len(times)):
                            _time = times[i].item()

                            _obj = objects[i]
                            obj = json.loads(convert_byte_to_string(np.array(_obj)))

                            _agent = agents[i]
                            agent = json.loads(convert_byte_to_string(np.array(_agent)))

                            anno = {
                                "idx": i,
                                "time": _time,
                                "agent": agent,
                                "objects": obj,
                            }
                            f.write(json.dumps(anno) + "\n")
            else:
                sensor_uuids_for_hdf5.append(sensor.uuid)

        sensor_uuids_for_hdf5.extend(extra_obs_keys_to_save or [])

        hdf5_save_path = os.path.join(save_dir, f"hdf5_sensors{save_file_suffix}.hdf5")
        saved_paths.append(hdf5_save_path)
        with h5py.File(hdf5_save_path, "w") as hdf5_file:
            for i, observations in enumerate(observations_list):
                group = hdf5_file.create_group(f"{i}")
                recursively_write_to_hdf5(
                    group=group,
                    data={k: observations[k] for k in sensor_uuids_for_hdf5},
                )
    except:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)
        raise


def save_top_view_as_png(top_view: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    imageio.imwrite(file_path, top_view)


def find_all(name: str, path: str):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def sha256sum_file(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def format_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = traceback.extract_tb(exc_traceback)

    for detail in reversed(traceback_details):
        if detail.filename.startswith(PACKAGE_ROOT):
            fname = detail.filename
            lineno = detail.lineno

            final_fname = traceback_details[-1].filename
            final_lineno = traceback_details[-1].lineno

            error_msg = f'{exc_type.__name__}("{str(exc_value)}") was raised on line {lineno} of {fname}'

            if fname != final_fname or lineno != final_lineno:
                error_msg += f" (originally thrown on {final_lineno} of {final_fname})"

            return error_msg

    return traceback.format_exc().replace("\n", "\\n")


def print_error(*args, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    print(*args, file=sys.stderr, **kwargs)
