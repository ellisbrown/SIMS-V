import json
import random

import pytest

from sims.qa.generators.common import clean_obj_type, gen_distance_options
from sims.qa.generators.descriptive import (
    gen_binary_qa,
    gen_obj_count_qas,
    gen_obj_size_est_qas,
)
from sims.qa.generators.temporal import gen_temporal_order_qa, gen_temporal_rel_qa
from sims.qa.qa_combine_mp import combine_qa_files
from sims.qa.qa_to_ov_train_multiturn import (
    build_multi_turn_convos,
    create_multiturn_jsonl,
)
from sims.qa.spatial_qa_gen import (
    DEFAULT_VISIBILITY_THRESHOLD,
    QAProcessingError,
    generate_qa_for_dataset,
    generate_qa_for_dataset_slow,
    generate_spatial_qa_pairs_for_video,
    preprocess_salient_objects,
    process_scene,
)


def test_distance_options_terminate_after_coarse_rounding():
    options = gen_distance_options(0.2, num_options=4, digits=0, ambiguity_threshold=1)

    assert len(options) == len(set(options)) == 4
    assert round(0.2, 0) in options
    assert all(option == round(option, 0) for option in options)
    assert all(option == round(0.2, 0) or abs(option - 0.2) >= 1 for option in options)


def _temporal_data():
    return {
        "salient_objects_n_frames": {
            "chair-early": 10,
            "chair-late": 8,
            "lamp": 7,
            "sofa": 6,
        },
        "salient_obj_idx_map": {
            "chair-early": [0, 1],
            "chair-late": [2, 3],
            "lamp": [0, 4],
            "sofa": [5, 6],
        },
        "object_info": {
            "chair-early": {"object_type": "ObjaChair"},
            "chair-late": {"object_type": "Chair"},
            "lamp": {"object_type": "FloorLamp"},
            "sofa": {"object_type": "Sofa"},
        },
    }


def test_temporal_questions_use_distinct_cleaned_categories_and_frames():
    random.seed(1)
    relation = gen_temporal_rel_qa(_temporal_data(), shuffle=False)
    assert relation["object_1"] != relation["object_2"]

    # Chair and floor lamp tie at frame zero, so a three-object ordering is
    # impossible even though three cleaned categories are present.
    with pytest.raises(ValueError, match="distinct first-appearance frames"):
        gen_temporal_order_qa(_temporal_data(), num_objects=3)

    order = gen_temporal_order_qa(_temporal_data(), num_objects=2)
    assert len(order["selected_objects"]) == len(set(order["selected_objects"])) == 2


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_size_question_uses_first_instance_seen(version):
    early_bbox = {
        "object_id": "chair-early",
        "axesLengths": [0.2, 0.3, 0.4],
    }
    late_bbox = {
        "object_id": "chair-late",
        "axesLengths": [1.0, 1.1, 1.2],
    }
    preprocessed = {
        "salient_obj_idx_map": {"chair-early": [1], "chair-late": [10]},
        "spatial_metadata": {
            "salient_type_counts": {"Chair": 2},
            # Deliberately put the later-seen instance first.
            "salient_object_bbox": {"Chair": [late_bbox, early_bbox]},
        },
    }

    qa = gen_obj_size_est_qas(preprocessed, 1, version=version)[0]
    assert qa["bbox"]["object_id"] == "chair-early"
    assert qa["gt_size"] == 40


def test_vsi_size_rejects_ambiguous_multiple_instances():
    preprocessed = {
        "salient_obj_idx_map": {"chair-a": [1], "chair-b": [2]},
        "spatial_metadata": {
            "salient_type_counts": {"Chair": 2},
            "salient_object_bbox": {
                "Chair": [
                    {"object_id": "chair-a", "axesLengths": [0.2, 0.3, 0.4]},
                    {"object_id": "chair-b", "axesLengths": [0.4, 0.5, 0.6]},
                ]
            },
        },
    }

    with pytest.raises(ValueError, match="No salient objects"):
        gen_obj_size_est_qas(preprocessed, 1, version="vsi")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_size_question_rejects_instances_first_seen_in_same_frame(version):
    preprocessed = {
        "salient_obj_idx_map": {"chair-a": [1, 2], "chair-b": [1, 3]},
        "spatial_metadata": {
            "salient_type_counts": {"Chair": 2},
            "salient_object_bbox": {
                "Chair": [
                    {"object_id": "chair-a", "axesLengths": [0.2, 0.3, 0.4]},
                    {"object_id": "chair-b", "axesLengths": [1.0, 1.1, 1.2]},
                ]
            },
        },
    }

    with pytest.raises(ValueError, match="valid size estimation"):
        gen_obj_size_est_qas(preprocessed, 1, version=version)


def _qa_entry(filename, task, idx=0):
    return {
        "idx": idx,
        "id": idx,
        "type": "v0",
        "filename": filename,
        "source": "dataset",
        "task": task,
        "question": f"Question for {task}?",
        "gt_answer": "yes",
        "mc_question": f"Question for {task}?\nA. yes\nB. no",
        "mc_answer": "A",
        "mc_choices": ["A. yes", "B. no"],
    }


def test_conversation_ids_are_stable_and_unique_across_task_files(tmp_path):
    video = "val/scene/rgb__0.mp4"
    video_path = tmp_path / video
    video_path.parent.mkdir(parents=True)
    video_path.touch()
    count_chunk = [[_qa_entry(video, "obj_count")]]
    temporal_chunk = [[_qa_entry(video, "temporal_rel")]]

    count, _ = build_multi_turn_convos(count_chunk, "mc", str(tmp_path))
    count_again, _ = build_multi_turn_convos(count_chunk, "mc", str(tmp_path))
    temporal, _ = build_multi_turn_convos(temporal_chunk, "mc", str(tmp_path))

    assert count[0]["id"] == count_again[0]["id"]
    assert count[0]["id"] != temporal[0]["id"]
    assert count[0]["id"].startswith("sims_dataset__")
    assert count[0]["data_source"] == "sims_dataset"


def test_combine_filters_exact_requested_question_types(tmp_path):
    scene = tmp_path / "val" / "scene"
    scene.mkdir(parents=True)
    video = "scene/rgb__0.mp4"
    (scene / "qa_pairs_obj_count__0.jsonl").write_text(
        json.dumps(_qa_entry(video, "obj_count")) + "\n"
    )
    (scene / "qa_pairs_temporal_rel__0.jsonl").write_text(
        json.dumps(_qa_entry(video, "temporal_rel")) + "\n"
    )

    combine_qa_files(
        str(tmp_path),
        "val",
        "combined_qa_pairs.jsonl",
        num_workers=1,
        question_types=["obj_count"],
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "val" / "combined_qa_pairs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["task"] for record in records] == ["obj_count"]


def test_formatter_filters_requested_types_and_removes_only_owned_stale_files(
    tmp_path,
):
    scene = tmp_path / "val" / "scene"
    scene.mkdir(parents=True)
    video = "val/scene/rgb__0.mp4"
    (tmp_path / video).touch()
    records = [
        _qa_entry(video, "obj_count"),
        _qa_entry(video, "temporal_rel", idx=1),
    ]
    (tmp_path / "val" / "combined_qa_pairs.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    output_dir = tmp_path / "qas" / "val" / "rgb"
    output_dir.mkdir(parents=True)
    stale = output_dir / "mt1_temporal_rel_mc.jsonl"
    stale.write_text("stale\n")
    existing_other_mode = output_dir / "mt1_obj_count_oe.jsonl"
    existing_other_mode.write_text("keep other mode\n")
    user_file = output_dir / "notes.jsonl"
    user_file.write_text("keep\n")

    create_multiturn_jsonl(
        dataset_dir=str(tmp_path),
        input_filename="combined_qa_pairs.jsonl",
        output_subdir="qas",
        output_filename_base="",
        split="val",
        group_by_task=True,
        question_types=["obj_count"],
    )

    assert not stale.exists()
    assert existing_other_mode.read_text() == "keep other mode\n"
    assert user_file.read_text() == "keep\n"
    assert (output_dir / "mt1_obj_count_mc.jsonl").is_file()


def _write_video_inputs(
    root,
    *,
    scene_name="scene",
    malformed=False,
    frame_count=25,
    object_type="Chair",
):
    scene = root / "val" / scene_name
    scene.mkdir(parents=True)
    annotation = scene / "offline_annos__0.jsonl"
    object_id = object_type.lower()
    if malformed:
        annotation.write_text("{not json}\n")
    else:
        frame = {
            "time": 0,
            "objects": {
                object_id: {
                    "object_type": object_type,
                    "pct_pixels": 0.5,
                    "synset": f"{object_id}.n.01",
                }
            },
        }
        annotation.write_text(
            "".join(
                json.dumps({**frame, "time": index}) + "\n"
                for index in range(frame_count)
            )
        )
    metadata = {
        "object_bbox": [
            {
                "object_id": object_id,
                "object_type": object_type,
                "axesLengths": [0.2, 0.3, 0.4],
            }
        ]
    }
    (scene / "spatial_metadata.json").write_text(json.dumps(metadata))
    (scene / "rgb__0.mp4").touch()
    return scene, annotation


def test_binary_negative_excludes_seen_normalized_category():
    preprocessed = {
        "salient_objects_n_frames": {"apple": 25},
        "object_info": {"apple": {"object_type": "Apple"}},
        "max_visibility_map": {"apple": 0.5},
    }

    # This seed previously selected the unseen raw alias ``ObjaApple`` and
    # produced a negative answer to "Did you see an apple?".
    random.seed(658)
    qa = gen_binary_qa(preprocessed, prob_seen=0)

    assert qa["gt_answer"] == "No"
    assert clean_obj_type(qa["object_type"]) != "apple"


def test_video_qa_terminates_when_generator_returns_empty_list(tmp_path):
    scene, annotation = _write_video_inputs(tmp_path, object_type="Apple")

    assert (
        generate_spatial_qa_pairs_for_video(
            str(annotation),
            str(scene / "spatial_metadata.json"),
            "scene/rgb__0.mp4",
            "dataset",
            num_questions=1,
            question_type="vsi_obj_count_minimal",
        )
        == []
    )


def test_object_count_uses_strict_released_data_visibility_threshold(tmp_path):
    annotation = tmp_path / "offline_annos__0.jsonl"
    metadata_path = tmp_path / "spatial_metadata.json"
    visibility_by_id = {
        "chair-above": 0.06,
        "chair-barely-above": 0.0501,
        "chair-at-threshold": DEFAULT_VISIBILITY_THRESHOLD,
    }
    frame = {
        "objects": {
            object_id: {
                "object_type": "Chair",
                "pct_pixels": visibility,
                "synset": "chair.n.01",
            }
            for object_id, visibility in visibility_by_id.items()
        }
    }
    annotation.write_text(
        "".join(json.dumps({**frame, "time": index}) + "\n" for index in range(25))
    )
    metadata_path.write_text(
        json.dumps(
            {
                "object_bbox": [
                    {"object_id": object_id, "object_type": "Chair"}
                    for object_id in visibility_by_id
                ]
            }
        )
    )

    preprocessed = preprocess_salient_objects(annotation, metadata_path)
    assert preprocessed["spatial_metadata"]["salient_type_counts"] == {"Chair": 2}

    generic = gen_obj_count_qas(preprocessed, 1)[0]
    vsi = gen_obj_count_qas(preprocessed, 1, vsi=True)[0]
    assert generic["gt_answer"] == vsi["gt_answer"] == 2
    assert generic["question"] == "How many total chair(s) are in this house?"
    assert vsi["question"] == "How many chair(s) are in this room?"


@pytest.mark.parametrize(
    ("missing_file", "missing_message"),
    [
        ("rgb__0.mp4", "missing videos for indices ['0']"),
        ("offline_annos__0.jsonl", "missing annotations for indices ['0']"),
    ],
)
def test_process_scene_rejects_annotation_video_index_mismatch(
    tmp_path, missing_file, missing_message
):
    scene, _ = _write_video_inputs(tmp_path)
    (scene / missing_file).unlink()

    with pytest.raises(
        QAProcessingError, match="Annotation/video index mismatch"
    ) as error:
        process_scene(
            str(scene),
            "dataset",
            num_questions_per_video=1,
            question_type="obj_count",
        )

    assert missing_message in str(error.value)


def test_process_scene_accepts_released_rgb_filename(tmp_path):
    scene, _ = _write_video_inputs(tmp_path)
    (scene / "rgb__0.mp4").rename(scene / "raw_navigation_camera__0.mp4")

    summary = process_scene(
        str(scene),
        "dataset",
        num_questions_per_video=1,
        question_type="obj_count",
    )

    qa_record = json.loads((scene / "qa_pairs_obj_count__0.jsonl").read_text())
    assert summary["videos"] == 1
    assert qa_record["filename"] == "scene/raw_navigation_camera__0.mp4"


def test_process_scene_rejects_ambiguous_rgb_filenames(tmp_path):
    scene, _ = _write_video_inputs(tmp_path)
    (scene / "raw_navigation_camera__0.mp4").touch()

    with pytest.raises(QAProcessingError, match="Ambiguous RGB videos"):
        process_scene(
            str(scene),
            "dataset",
            num_questions_per_video=1,
            question_type="obj_count",
        )


def test_video_qa_seed_is_stable_and_preprocessing_errors_are_not_silenced(tmp_path):
    scene, annotation = _write_video_inputs(tmp_path)
    kwargs = dict(
        anno_jsonl_path=str(annotation),
        metadata_json_path=str(scene / "spatial_metadata.json"),
        video_filename="scene/rgb__0.mp4",
        source="dataset",
        num_questions=1,
        question_type="obj_count",
        seed=17,
    )

    first = generate_spatial_qa_pairs_for_video(**kwargs)
    random.seed(99999)
    second = generate_spatial_qa_pairs_for_video(**kwargs)
    assert first == second

    annotation.write_text("{not json}\n")
    with pytest.raises(QAProcessingError, match="Failed to preprocess"):
        generate_spatial_qa_pairs_for_video(**kwargs)


def test_video_qa_seed_is_stable_across_rgb_filename_rename(tmp_path):
    scene, annotation = _write_video_inputs(tmp_path)
    kwargs = dict(
        anno_jsonl_path=str(annotation),
        metadata_json_path=str(scene / "spatial_metadata.json"),
        source="dataset",
        num_questions=1,
        question_type="obj_count",
        seed=17,
    )

    canonical = generate_spatial_qa_pairs_for_video(
        video_filename="scene/rgb__0.mp4", **kwargs
    )
    released = generate_spatial_qa_pairs_for_video(
        video_filename="scene/raw_navigation_camera__0.mp4", **kwargs
    )

    assert [{k: v for k, v in qa.items() if k != "filename"} for qa in canonical] == [
        {k: v for k, v in qa.items() if k != "filename"} for qa in released
    ]


def test_dataset_qa_is_independent_of_worker_count(tmp_path):
    for scene_name in ("scene-a", "scene-b"):
        _write_video_inputs(tmp_path, scene_name=scene_name)

    generate_qa_for_dataset(
        str(tmp_path),
        "dataset",
        question_type="obj_count",
        num_questions_per_video=1,
        num_workers=1,
        seed=123,
    )
    first_outputs = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.glob("val/*/qa_pairs_obj_count__0.jsonl"))
    }

    generate_qa_for_dataset(
        str(tmp_path),
        "dataset",
        question_type="obj_count",
        num_questions_per_video=1,
        num_workers=2,
        seed=123,
    )
    second_outputs = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.glob("val/*/qa_pairs_obj_count__0.jsonl"))
    }

    assert first_outputs == second_outputs


def test_short_video_is_ineligible_but_scene_errors_fail_by_default(tmp_path):
    scene, annotation = _write_video_inputs(tmp_path, frame_count=2)
    assert (
        generate_spatial_qa_pairs_for_video(
            str(annotation),
            str(scene / "spatial_metadata.json"),
            "scene/rgb__0.mp4",
            "dataset",
            question_type="obj_count",
        )
        == []
    )

    (scene / "spatial_metadata.json").unlink()
    with pytest.raises(QAProcessingError, match="failed for 1 scene"):
        generate_qa_for_dataset_slow(
            str(tmp_path), "dataset", question_type="obj_count"
        )

    summary = generate_qa_for_dataset_slow(
        str(tmp_path), "dataset", question_type="obj_count", allow_partial=True
    )
    assert summary["errors"] == 1


def test_partial_qa_run_removes_failed_scene_outputs_before_combine(tmp_path):
    good_scene, _ = _write_video_inputs(tmp_path, scene_name="good")
    bad_scene, _ = _write_video_inputs(tmp_path, scene_name="bad", malformed=True)
    stale = bad_scene / "qa_pairs_obj_count__0.jsonl"
    stale.write_text(json.dumps(_qa_entry("bad/video.mp4", "obj_count")) + "\n")

    summary = generate_qa_for_dataset_slow(
        str(tmp_path),
        "dataset",
        question_type="obj_count",
        num_questions_per_video=1,
        allow_partial=True,
    )

    assert summary["errors"] == 1
    assert not stale.exists()
    assert (good_scene / "qa_pairs_obj_count__0.jsonl").is_file()

    combine_qa_files(
        str(tmp_path),
        "val",
        "combined_qa_pairs.jsonl",
        num_workers=1,
        question_types=["obj_count"],
    )
    combined = (tmp_path / "val" / "combined_qa_pairs.jsonl").read_text()
    assert "STALE" not in combined
    assert "good/rgb__0.mp4" in combined
