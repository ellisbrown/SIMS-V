# Development

This page describes the repository structure and checks used when extending the
SIMS-V generator.

## Environment and checks

```bash
uv sync --locked --dev
uv run pytest
uv run pre-commit run --all-files
```

The test suite exercises local multiprocessing queues and process pools. Run it
in a normal terminal or CI runner that permits POSIX semaphores; restricted
sandboxes can fail with `SemLock: Operation not permitted` before project code
is executed.

## Code map

- `src/sims/cli.py`: public `sims-v generate` and `sims-v qa` dispatcher
- `src/sims/data_generation/`: parsers, workers, persistence, queues, sensors,
  and walkthrough orchestration
- `src/sims/environment/`: AI2-THOR controller and discrete Stretch actions
- `src/sims/tasks/`: the house-walkthrough task and sampler
- `src/sims/planning/`: navigation planner support
- `src/sims/qa/`: metadata extraction, question generation, combining, and
  training-format conversion
- `src/sims/pipeline.py`: four-stage QA orchestrator
- `tests/`: deterministic unit and integration tests

The generator currently has one trajectory behavior: visit each room and record
a panoramic scan. The public CLI intentionally does not expose task or action
policy registries when there is no meaningful choice.

## Import behavior

Keep public help and CPU-only utilities lightweight:

- do not load datasets or simulator assets at import time;
- resolve `--objaverse-dir` before importing controller constants;
- keep stage-specific imports inside their stage runners; and
- avoid GPU detection or worker-count logging while constructing parsers.

The first real simulator or dataset load can require HTTPS access.

## Output compatibility

Each house is a flat directory under `<dataset-dir>/<split>/<house-id>/`.
Trajectory indices are encoded in filenames such as
`rgb__0.mp4` and `offline_annos__0.jsonl`; QA consumers rely
on that layout and on video paths relative to the dataset directory.

`constants.yaml` records output-defining generation settings. When changing a
setting, task label, action policy, or filename:

1. update the generation schema when the stored meaning changes;
2. keep resume checks consistent with the artifacts actually written;
3. add a focused deterministic test; and
4. update [getting started](getting-started.md) or the
   [QA reference](qa-pipeline.md).

Prefer the narrowest implementation that supports the documented workflow.
Archived launch modes and unused task families should remain in Git history
rather than the public runtime.
