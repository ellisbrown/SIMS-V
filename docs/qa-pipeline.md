# Video-QA pipeline

The QA command converts generated walkthroughs into spatial metadata,
question-answer pairs, and training JSONL:

```bash
uv run sims-v qa \
    --dataset-dir outputs/my-dataset \
    --split val \
    --question-types obj_count obj_rel_distance temporal_order_5
```

## Stages

The command runs four stages in order:

1. `metadata`: extract object geometry and room layout from each walkthrough
2. `qa`: generate requested question types for each eligible video
3. `combine`: merge per-video records into `combined_qa_pairs.jsonl`
4. `format`: write grouped training conversations under `qas/`

Select a subset when inspecting or rebuilding intermediate results:

```bash
uv run sims-v qa \
    --dataset-dir outputs/my-dataset \
    --split val \
    --stages metadata qa
```

Useful controls include:

- `--skip-completed`: skip complete metadata and QA stages; cheap derived
  combine and format stages are always rebuilt. Completion is based on existing
  files, so omit this flag after changing `--seed` or
  `--num-questions-per-video`
- `--overwrite-metadata`: regenerate existing spatial metadata
- `--num-workers-per-gpu`: metadata workers per detected GPU
- `--num-workers`: parallel QA workers
- `--num-questions-per-video`: attempted questions of each requested type
- `--seed`: deterministic QA and formatting seed
- `--allow-partial`: retain successful outputs when another scene or selected
  video is missing
- `--answer-mode {mc,mc_direct,oe}`: multiple choice, direct option letter, or
  open-ended formatted answers
- `--video-version`: video modality stored in formatted records
- `--max-qa-per-convo`: questions grouped into each conversation

The default is strict: processing failures and missing selected videos make the
command exit nonzero.

## Question types

The generator currently provides:

- Temporal: `temporal_rel`, `temporal_order_2` through `temporal_order_6`,
  `vsi_obj_appearance_order`, `vsi_obj_appearance_order_minimal`
- Descriptive: `descriptive_binary`, `obj_count`, `obj_long_size_est`,
  `obj_long_size_est_v2`, `obj_short_size_est`, `vsi_obj_count`,
  `vsi_obj_size_est`, `vsi_obj_count_minimal`, `vsi_obj_size_est_minimal`
- Spatial: `obj_abs_distance`, `obj_rel_distance`, `obj_rel_direction`,
  `vsi_obj_abs_distance`, `vsi_obj_rel_distance`,
  `vsi_obj_rel_direction_easy`, `vsi_obj_rel_direction_medium`,
  `vsi_obj_rel_direction_hard`, and their registered minimal variants
- Layout: `n_rooms`, `house_size_est`, `vsi_room_size_est`

`uv run sims-v qa --help` lists and validates every registered question type.
A requested type can still produce no record when a walkthrough lacks suitable
objects.

## Object salience

An object instance is eligible when its segmentation mask covers strictly more
than 5% of at least one trajectory frame. Walls and floors are excluded.
Recognition and counting labels use these trajectory-visible instances rather
than every object contained in the underlying house specification.

## Answer formatting

Multiple-choice formatting is the default:

```bash
uv run sims-v qa \
    --dataset-dir outputs/my-dataset \
    --split val \
    --stages format \
    --answer-mode mc \
    --question-types vsi_obj_appearance_order
```

Use `--answer-mode oe` for open-ended records. Mode-specific rebuilds replace
only files belonging to that answer mode, so open-ended and multiple-choice
outputs can coexist. The exact combination used in the paper is documented in
[paper settings](paper-settings.md).

## Video modalities

RGB is always generated. Alternative videos exist only when requested with
`sims-v generate --extra-video-modalities ...`:

| `sims-v qa --video-version` | Generation modality | Filename prefix |
|---|---|---|
| `rgb` | Always generated | `rgb` |
| `depth` | `depth` | `depth` |
| `edge` | `edge` | `edge` |
| `colored_edge` | `colored_edge` | `colored_edge` |
| `colored_edge_no` | `non_overlapping_colored_edge` | `non_overlapping_colored_edge` |
| `semantic_seg` | `semantic_seg` | `semantic_seg` |
| `instance_seg` | `instance_seg` | `instance_seg` |
| `mean_mask` | `mean_mask_overlay` | `mean_mask_overlay` |
| `masked_bg` | `masked_background` | `masked_background` |

Formatting changes the relative video path stored in a record; it does not
render another video. Selecting an alternative at formatting time therefore
requires that it was rendered during generation.

## Output layout

```text
outputs/my-dataset/
├── val/
│   ├── 000000/
│   │   ├── rgb__0.mp4
│   │   ├── offline_annos__0.jsonl
│   │   ├── spatial_metadata.json
│   │   ├── qa_pairs_obj_count__0.jsonl
│   │   └── qa_pairs_obj_rel_distance__0.jsonl
│   └── combined_qa_pairs.jsonl
└── qas/
    └── val/
        └── rgb/
            ├── mt1_obj_count_mc.jsonl
            └── mt1_obj_rel_distance_mc.jsonl
```

Trajectory indices use the `__0`, `__1`, ... suffix consistently across video,
annotation, and QA filenames. Formatted video paths remain relative to the
dataset directory. QA readers also accept the released
`raw_navigation_camera__N.mp4` filename; newly generated datasets use the
shorter `rgb__N.mp4` name.

## Troubleshooting

- Missing metadata: check that each house has matching
  `offline_annos__N.jsonl` and `rgb__N.mp4` files.
- Empty QA files: the scene may not contain eligible objects for the requested
  question type.
- Missing formatted videos: verify that the selected `--video-version` exists
  for the same trajectory suffix.
- Metadata GPU exhaustion: lower `--num-workers-per-gpu`.
