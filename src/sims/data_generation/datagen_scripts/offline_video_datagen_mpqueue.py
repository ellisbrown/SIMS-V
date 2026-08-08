"""Generate SIMS-V house walkthroughs with local multiprocessing."""

from sims.data_generation.arg_parsers import (
    EXTRA_VIDEO_MODALITIES,
    get_arg_parser_for_offline_datagen,
    resolve_extra_video_modalities,
)
from sims.data_generation.paths import resolve_objaverse_data_dir
from sims.video_paths import RGB_VIDEO_STEM


def run(args):
    """Run walkthrough generation from an already parsed argument namespace."""
    import torch

    from sims.data_generation.datagen_scripts.task_datagen_utils import (
        default_workers_per_device,
        get_house_dataset,
        get_walkthrough_task_args,
    )
    from sims.data_generation.datagen_utils import build_generation_config
    from sims.data_generation.manager_single_machine_mpqueue import (
        manager_single_machine_mpqueue,
    )
    from sims.data_generation.path_planner_utils import REGISTERED_PLANNERS
    from sims.data_generation.sensors import (
        ColoredEdgeSensorTHOR,
        DepthSensorTHOR,
        EdgeSensorTHOR,
        InstanceSegmentationSensorTHOR,
        MaskedBackgroundSensorTHOR,
        MeanMaskOverlaySensorTHOR,
        NonOverlappingColoredEdgeSensorTHOR,
        OfflineAnnoSensor,
        SemanticSegmentationSensorTHOR,
    )
    from sims.environment.action_spaces import DiscreteStretchActionSpace
    from sims.environment.stretch_controller import StretchController
    from sims.tasks import REGISTERED_TASK_SAMPLERS, REGISTERED_TASKS
    from sims.tasks.house_walkthrough_task import HouseWalkthroughTask
    from sims.utils.constants.stretch_initialization_utils import (
        INTEL_CAMERA_HEIGHT,
        INTEL_CAMERA_WIDTH,
        get_stretch_env_args,
    )

    task_type = HouseWalkthroughTask.task_type_str
    extra_video_modalities = resolve_extra_video_modalities(args.extra_video_modalities)
    n_gpus = torch.cuda.device_count()
    workers_per_device = default_workers_per_device()
    default_workers = max(1, n_gpus * workers_per_device)
    n_workers = args.workers or default_workers

    if args.workers is None:
        if n_gpus:
            print(
                f"Default: {workers_per_device} workers x {n_gpus} GPUs "
                f"=> {n_workers} workers"
            )
        else:
            print("Default: no GPU detected; using one CPU worker")

    controller_args = get_stretch_env_args()
    height = int(INTEL_CAMERA_HEIGHT * args.resolution_scale)
    width = int(INTEL_CAMERA_WIDTH * args.resolution_scale)
    depth_requested = "depth" in extra_video_modalities
    semantic_segmentation_requested = "semantic_seg" in extra_video_modalities
    controller_args.update(
        width=width,
        height=height,
        renderDepthImage=depth_requested,
        renderSemanticSegmentation=semantic_segmentation_requested,
        quality=args.quality,
        rotation_noise_std_degrees=args.rotation_noise_std_degrees,
    )

    action_space = DiscreteStretchActionSpace()
    optional_sensor_types = {
        "depth": DepthSensorTHOR,
        "semantic_seg": SemanticSegmentationSensorTHOR,
        "instance_seg": InstanceSegmentationSensorTHOR,
        "edge": EdgeSensorTHOR,
        "colored_edge": ColoredEdgeSensorTHOR,
        "non_overlapping_colored_edge": NonOverlappingColoredEdgeSensorTHOR,
        "mean_mask_overlay": MeanMaskOverlaySensorTHOR,
        "masked_background": MaskedBackgroundSensorTHOR,
    }
    assert tuple(optional_sensor_types) == EXTRA_VIDEO_MODALITIES
    extra_sensors = [OfflineAnnoSensor("offline_annos", height=height, width=width)]
    extra_sensors.extend(
        optional_sensor_types[modality](modality, height=height, width=width)
        for modality in extra_video_modalities
    )
    task_args = get_walkthrough_task_args(
        max_steps=args.max_steps,
        action_space=action_space,
        width=width,
        height=height,
        extra_sensors=extra_sensors,
        include_manipulation_sensor=False,
    )

    max_houses_per_split = {"train": 0, "val": 0, "test": 0}
    max_houses_per_split[args.split] = args.max_houses
    print(f"Generating at most {args.max_houses} houses from the {args.split} split")

    dataset = get_house_dataset(
        house_dataset=args.house_dataset,
        max_houses_per_split=max_houses_per_split,
    )[args.split]
    if len(dataset) > args.max_houses:
        dataset = dataset.select(list(range(args.max_houses)))
    if len(dataset) == 0:
        raise ValueError(f"No houses were loaded from the {args.split} split")

    task_sampler_args = {
        "task_args": task_args,
        "houses": dataset,
        "house_inds": list(range(len(dataset))),
        "controller_args": controller_args,
        "max_tasks": float("inf"),
        "task_type": REGISTERED_TASKS[task_type],
        "controller_type": StretchController,
        "sample_per_house": args.trajectories_per_house,
        "prob_randomize_materials": args.material_randomization_probability,
    }

    task_sampler_type = REGISTERED_TASK_SAMPLERS[task_type]
    path_planner = REGISTERED_PLANNERS[task_type]()
    generation_config = build_generation_config(
        args,
        action_space,
        task_type=task_type,
        width=width,
        height=height,
        video_modalities=(RGB_VIDEO_STEM, *extra_video_modalities),
    )

    total_houses = len(dataset) * args.trajectories_per_house
    n_workers = min(n_workers, total_houses)

    manager_single_machine_mpqueue(
        nworkers=n_workers,
        house_repeats=args.trajectories_per_house,
        split=args.split,
        top_level_save_dir=args.dataset_dir,
        constants=generation_config,
        task_sampler_type=task_sampler_type,
        device_to_task_sampler_kwargs=lambda device: {
            **task_sampler_args,
            "device": device,
        },
        path_planner=path_planner,
    )


def main(argv=None, *, prog=None):
    parser = get_arg_parser_for_offline_datagen(prog=prog)
    args = parser.parse_args(argv)
    try:
        resolve_objaverse_data_dir(
            args.objaverse_dir,
            required=args.house_dataset == "objaverse",
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    return run(args)


if __name__ == "__main__":
    main()
