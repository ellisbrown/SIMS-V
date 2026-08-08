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

import os
import sys
import queue
import multiprocessing as mp
from tqdm import tqdm

import ai2thor.platform
import torch
from sims.environment.stretch_controller import StretchController
from sims.utils.constants.stretch_initialization_utils import (
    INTEL_CAMERA_HEIGHT,
    INTEL_CAMERA_WIDTH,
    get_stretch_env_args,
)
from sims.qa.spatial_metadata_gen import process_house


def _discover_scene_paths(split_dir, proc_order="asc"):
    """Return only scene directories, in the requested processing order."""
    scene_paths = [
        os.path.join(split_dir, entry)
        for entry in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, entry))
    ]

    if proc_order == "asc":
        return sorted(scene_paths)
    if proc_order == "desc":
        return sorted(scene_paths, reverse=True)
    if proc_order == "rand":
        import random

        return random.sample(scene_paths, len(scene_paths))
    raise ValueError(f"Invalid proc_order: {proc_order}")


def _collect_worker_results(from_worker_queue, processes, expected_count):
    """Collect one result per scene and fail if workers exit prematurely."""
    results = []
    with tqdm(total=expected_count, desc="Processing scenes", ncols=100) as pbar:
        while len(results) < expected_count:
            try:
                result = from_worker_queue.get(timeout=1)
            except queue.Empty:
                failed_workers = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed_workers:
                    exit_codes = [process.exitcode for process in failed_workers]
                    raise RuntimeError(
                        f"Metadata workers exited before returning all results: {exit_codes}"
                    )
                if processes and not any(process.is_alive() for process in processes):
                    raise RuntimeError(
                        "Metadata workers exited before returning all scene results"
                    )
                continue

            results.append(result)
            pbar.update(1)

    return results


def worker_process(
    to_worker_queue,
    from_worker_queue,
    controller_args,
    gpu_device,
    worker_id,
    overwrite,
    save_images,
):
    """Worker function that processes scenes from the queue with GPU support."""
    controller = StretchController(True, **controller_args, gpu_device=gpu_device)

    # REQUIRED TO WORK WITH THE OBJAVERSE SCENES
    controller.step(
        action="AdvancePhysicsStep",
        simSeconds=2.0,
        raise_for_failure=True,
    )
    print(
        f"[G#{gpu_device}W#{worker_id}] Worker process # {worker_id} initialized on GPU {gpu_device}."
    )

    while True:
        try:
            scene_dir = to_worker_queue.get(timeout=5)  # Wait for a task from the queue
            if scene_dir is None:
                break  # Termination signal
            scene_path = os.path.abspath(scene_dir)

            house_json_path = os.path.join(scene_path, "house_spec.json")
            result = process_house(
                controller,
                house_json_path=house_json_path,
                retries=3,
                save=True,
                overwrite=overwrite,
                save_images=save_images,
            )
            from_worker_queue.put((scene_dir, "success"))
        except queue.Empty:
            continue  # Wait for more tasks
        except Exception as e:
            if "scene_dir" in locals():
                from_worker_queue.put((scene_dir, f"failed: {str(e)}"))
            else:
                from_worker_queue.put((None, f"failed: {str(e)}"))

    print(
        f"[G#{gpu_device}W#{worker_id}] Worker process # {worker_id} on GPU {gpu_device} finished."
    )
    return


def _summarize_worker_results(results, split_dir, allow_partial=False):
    """Report metadata coverage and reject unexpected partial output by default."""
    failed = [result for result in results if "failed" in result[1]]
    success = [result for result in results if "success" in result[1]]

    print(f"Finished processing all scene directories in {split_dir}")
    print(f"{len(success)} scenes processed successfully")
    if failed:
        details = "\n".join(f"  - {scene}: {status}" for scene, status in failed)
        message = f"{len(failed)} scene(s) failed metadata generation:\n{details}"
        if allow_partial:
            print(f"WARNING: {message}")
        else:
            raise RuntimeError(message)

    return {"success": len(success), "failed": len(failed)}


def main(
    dataset_dir,
    split="val",
    resolution_scale=1,
    overwrite=False,
    save_images=False,
    num_workers_per_gpu=4,
    gpus=None,
    proc_order="asc",
    allow_partial=False,
):
    # Setup GPU devices
    if gpus is None:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            gpus = [None]  # CPU
            print("No GPUs detected. Running on CPU.")
        else:
            gpus = list(range(torch.cuda.device_count()))
            print(f"Detected {len(gpus)} GPUs. Running on GPUs: {gpus}")
    else:
        print(f"Running on provided GPUs: {gpus}")

    controller_args = {
        **get_stretch_env_args(),
        "width": int(INTEL_CAMERA_WIDTH * resolution_scale),
        "height": int(INTEL_CAMERA_HEIGHT * resolution_scale),
    }

    # Use CloudRendering on Linux with GPU (Linux64 builds don't exist)
    if gpus[0] is not None and sys.platform != "darwin":
        controller_args["platform"] = ai2thor.platform.CloudRendering

    # Get all scene directories
    split_dir = os.path.join(dataset_dir, split)

    scene_paths = _discover_scene_paths(split_dir, proc_order=proc_order)

    # Initialize multiprocessing queues
    to_worker_queue = mp.Queue()
    from_worker_queue = mp.Queue()

    # Fill the task queue
    for scene in scene_paths:
        to_worker_queue.put(scene)

    # Calculate total workers and add termination signals (None) to the queue
    total_workers = len(gpus) * num_workers_per_gpu
    if total_workers < 1:
        raise ValueError("At least one metadata worker is required")
    for _ in range(total_workers):
        to_worker_queue.put(None)

    # Start worker processes
    processes = []
    worker_counter = 0
    for _ in range(num_workers_per_gpu):
        for gpu_id in gpus:
            p = mp.Process(
                target=worker_process,
                args=(
                    to_worker_queue,
                    from_worker_queue,
                    controller_args,
                    gpu_id,
                    worker_counter,
                    overwrite,
                    save_images,
                ),
            )
            p.start()
            processes.append(p)
            worker_counter += 1

    try:
        results = _collect_worker_results(
            from_worker_queue, processes, expected_count=len(scene_paths)
        )
    except Exception:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()

    return _summarize_worker_results(results, split_dir, allow_partial=allow_partial)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Spatial Metadata for a house."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--split", type=str, default="val", help="The train, val, or test split."
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
    parser.add_argument(
        "--num_workers_per_gpu", type=int, default=4, help="Number of workers per GPU"
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU indices to use (default: auto-detect)",
    )
    parser.add_argument(
        "--proc_order",
        type=str,
        default="asc",
        choices=["asc", "desc", "rand"],
        help="Processing order of scenes",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Keep successful metadata if another scene fails (default: fail).",
    )

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        split=args.split,
        resolution_scale=args.resolution_scale,
        overwrite=args.overwrite,
        save_images=args.save_images,
        num_workers_per_gpu=args.num_workers_per_gpu,
        gpus=args.gpus,
        proc_order=args.proc_order,
        allow_partial=args.allow_partial,
    )
