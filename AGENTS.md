# SIMS-V agent guide

SIMS-V is public research code for generating ProcTHOR house walkthroughs and
spatial video QA. Keep changes focused on the documented release pipeline;
retired experiments and predecessor task families belong in Git history unless
the user explicitly asks to restore them.

## Start here

- [README.md](README.md): project overview and quick start
- [docs/getting-started.md](docs/getting-started.md): installation and data
  generation
- [docs/development.md](docs/development.md): architecture, import rules, and
  output-compatibility requirements
- [docs/qa-pipeline.md](docs/qa-pipeline.md): QA stages, modalities, and output
  layout
- [docs/paper-settings.md](docs/paper-settings.md): settings used for the paper

## Repository layout

- `src/sims/data_generation/`: generation CLI support, sensors, workers, and
  persistence
- `src/sims/environment/`: AI2-THOR controller and Stretch actions
- `src/sims/tasks/` and `src/sims/planning/`: walkthrough behavior and route
  planning
- `src/sims/qa/` and `src/sims/pipeline.py`: metadata, QA generation, and
  training-format conversion
- `tests/`: deterministic unit and integration tests

## Working conventions

- Use Python 3.9 and the checked-in `uv.lock`; do not replace the locked
  environment with an ad hoc `pip` setup.
- The public entry points are `sims-v generate` and `sims-v qa`.
- Treat the documented filenames and directory layout as a public interface.
  New RGB videos use `rgb__N.mp4`; QA readers intentionally retain support
  for released `raw_navigation_camera__N.mp4` files.
- Alternative video modalities are opt-in. Avoid adding default rendering,
  storage, or dependencies without a documented need.
- Keep imports and `--help` lightweight. Actual trajectory rendering requires
  an AI2-THOR graphics backend; ordinary tests do not require a GPU.

Run before handing off code changes:

```bash
uv sync --locked --dev
uv run pytest
uv run pre-commit run --all-files
```
