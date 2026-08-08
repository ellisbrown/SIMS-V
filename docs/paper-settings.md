# SIMS-V paper settings

This page collects settings specific to the experiments reported in the SIMS-V
paper. The normal defaults in the public CLI are intentionally smaller and
disable optional material and rotation randomization for first-time use.

## Trajectory generation

The main ProcTHOR-Objaverse corpus used two room walkthroughs per candidate
house with the following rendering and randomization settings:

```bash
uv run sims-v generate \
    --dataset-dir outputs/sims-v-paper \
    --house-dataset objaverse \
    --split train \
    --max-houses 1500 \
    --trajectories-per-house 2 \
    --max-steps 1000 \
    --resolution-scale 1.718 \
    --quality Ultra \
    --extra-video-modalities all \
    --material-randomization-probability 0 \
    --rotation-noise-std-degrees 0.5
```

The resolution scale configures the controller at 680x384. The raw navigation
RGB stream crops six pixels from each horizontal edge and is therefore
668x384; optional modalities such as depth, segmentation, and edge maps retain
the full 680x384 controller dimensions. `--max-houses` counts candidates;
invalid houses can yield fewer successful trajectories.

These options match the reported generation configuration. Starting poses and
walkthrough planning are stochastic, so fresh runs are not intended to recreate
the released videos frame-for-frame.

The 0.5-degree rotation noise was inherited from the embodied-agent controller
used for the paper launches. It adds small yaw jitter but is not required for a
good walkthrough, so ordinary generation defaults to zero. Forward translation
remains deterministic in both cases.

## Pinned public input datasets

| Type | Dataset | Revision |
|---|---|---|
| houses | [`ellisbrown/procthor-objaverse`](https://github.com/ellisbrown/procthor-objaverse) | `251d104d900f5694ffdf2e3e868f3ed22c291a99` |
| houses | [`ellisbrown/procthor-100k`](https://github.com/ellisbrown/procthor-100k) | `dae36fb48906fdbeecfaf4360cdb4f1b2cc4cf16` |
| annotations | [`ellisbrown/objaverse-plus`](https://github.com/ellisbrown/objaverse-plus) | `1bd4b77de24e76849e627af8e248437c6748e346` |
| annotations | [`ellisbrown/objaverse_sims`](https://huggingface.co/datasets/ellisbrown/objaverse_sims) | `9087d8d6df551c3ad0af85b1e2b24fa6f654ae7d` |

The AI2-THOR, AllenAct, and NLTK revisions are recorded in
[`pyproject.toml`](../pyproject.toml) and resolved by `uv.lock`.

## QA salience

An object instance contributes to QA when its segmentation mask covers strictly
more than 5% of at least one trajectory frame. Object-count and recognition
labels count these trajectory-visible instances, not every object in the house.

## 3Q Minimal Mix

The paper's 3Q mixture combines metric measurement, temporal tracking, and
perspective-dependent reasoning:

| Paper category | Generator task | Answer mode | Reference count | 25K target |
|---|---|---:|---:|---:|
| Measurement | `vsi_obj_abs_distance` | `oe` | 834 | 8,616 |
| Spatiotemporal | `vsi_obj_appearance_order` | `mc` | 618 | 6,384 |
| Perspective | `vsi_obj_rel_direction_medium` | `mc` | 378 | 3,905 |
| Perspective | `vsi_obj_rel_direction_hard` | `mc` | 373 | 3,853 |
| Perspective | `vsi_obj_rel_direction_easy` | `mc` | 217 | 2,242 |

Generate the source pool and combined records:

```bash
uv run sims-v qa \
    --dataset-dir outputs/sims-v-paper \
    --split train \
    --stages metadata qa combine \
    --num-questions-per-video 5 \
    --seed 42 \
    --question-types \
        vsi_obj_abs_distance \
        vsi_obj_appearance_order \
        vsi_obj_rel_direction_medium \
        vsi_obj_rel_direction_hard \
        vsi_obj_rel_direction_easy
```

Write measurement records as open-ended and the other tasks as multiple choice:

```bash
uv run sims-v qa \
    --dataset-dir outputs/sims-v-paper \
    --split train \
    --stages format \
    --answer-mode oe \
    --seed 42 \
    --question-types vsi_obj_abs_distance

uv run sims-v qa \
    --dataset-dir outputs/sims-v-paper \
    --split train \
    --stages format \
    --answer-mode mc \
    --seed 42 \
    --question-types \
        vsi_obj_appearance_order \
        vsi_obj_rel_direction_medium \
        vsi_obj_rel_direction_hard \
        vsi_obj_rel_direction_easy
```

This repository produces the source QA records. Proportional sampling to the
25K target and model training are downstream steps.
