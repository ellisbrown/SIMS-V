# Cluster setup

Trajectory generation and spatial-metadata extraction use AI2-THOR and need a
simulator-capable rendering setup. QA generation, combination, and formatting
can run on CPU.

## Check the node

From an allocated GPU node:

```bash
nvidia-smi
vulkaninfo --summary
```

If both commands succeed, use the normal environment and CLI:

```bash
uv sync --locked
uv run sims-v generate --dataset-dir outputs/demo --max-houses 1
```

Warnings about `DISPLAY` or `XDG_RUNTIME_DIR` are expected on a headless node as
long as `vulkaninfo` lists the NVIDIA device.

## Optional Apptainer wrapper

Some clusters expose the NVIDIA driver but not `libvulkan.so.1` or a usable
Vulkan ICD. `run_vulkan.sh` provides one container-based route without sudo:
it requires Apptainer on the compute node and combines the image's Vulkan
loader with NVIDIA driver libraries bound from the host.

If the image does not provide `uv`, `git`, and Git LFS, first expose a small
host-side tools environment to the wrapper:

```bash
micromamba create -y -p "$HOME/.micromamba/envs/sims-tools" \
    -c conda-forge uv git git-lfs ca-certificates
export SIMS_TOOLS_PREFIX="$HOME/.micromamba/envs/sims-tools"
```

Then check the wrapper and create the container-specific project environment:

```bash
apptainer --version
./run_vulkan.sh vulkaninfo --summary
./run_vulkan.sh uv sync --locked --python 3.9
./run_vulkan.sh uv run sims-v generate \
    --dataset-dir outputs/demo \
    --max-houses 1
```

By default the wrapper uses `docker://nvidia/vulkan:1.3-470`. Pre-pull an image
when compute nodes cannot access the registry:

```bash
apptainer pull "$HOME/containers/nvidia-vulkan.sif" \
    docker://nvidia/vulkan:1.3-470
export SIMS_VULKAN_SIF="$HOME/containers/nvidia-vulkan.sif"
```

The wrapper keeps container-built packages in `.venv-container` so they are not
mixed with a host virtual environment.

## Slurm example

Request a GPU using the partition and account names for your cluster, then run
the same command inside the allocation:

```bash
srun -p <gpu-partition> \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=32G \
    --time=02:00:00 \
    --pty bash

./run_vulkan.sh uv run sims-v generate \
    --dataset-dir outputs/demo \
    --max-houses 1 \
    --workers 1
```

## Troubleshooting

- `libvulkan.so.1` missing or `ERROR_INCOMPATIBLE_DRIVER`: use the wrapper or
  your cluster's supported Vulkan module/container.
- `GLIBC_* not found`: do not reuse a host-created virtual environment inside a
  container; rerun `uv sync` through the wrapper.
- `uv`, `git`, or Git LFS unavailable inside the container: set
  `SIMS_TOOLS_PREFIX` as shown above.
- GPU out of memory: reduce `--workers` to one.

Cluster software differs substantially. The only project requirement is that
the process can create an AI2-THOR CloudRendering controller on the allocated
NVIDIA device; the wrapper is optional.
