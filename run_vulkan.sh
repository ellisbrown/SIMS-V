#!/usr/bin/env bash
set -euo pipefail

# --- prepare arguments for the container ---
# We'll pass the arguments as a properly quoted string to the container
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

IMG="${SIMS_VULKAN_SIF:-docker://nvidia/vulkan:1.3-470}"

# Prefer a user-provided tools prefix; otherwise try common conda/mamba envs
TOOLS_PREFIX="${SIMS_TOOLS_PREFIX:-}"
if [[ -z "${TOOLS_PREFIX}" ]]; then
  for p in "$HOME/.conda/envs/sims-tools" "$HOME/.micromamba/envs/tools" "$HOME/micromamba/envs/tools"; do
    if [[ -x "$p/bin/git" ]]; then TOOLS_PREFIX="$p"; break; fi
  done
fi

exec apptainer exec --nv \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  "$IMG" bash -lc '
set -e
# Vulkan wiring (headless)
export LD_LIBRARY_PATH="/.singularity.d/libs:${LD_LIBRARY_PATH:-}"
unset VIRTUAL_ENV
ICD_FILE="${TMPDIR:-/tmp}/sims-v-nvidia-icd-${UID}.json"
cat > "$ICD_FILE" <<EOF
{"file_format_version":"1.0.0","ICD":{"library_path":"/.singularity.d/libs/libGLX_nvidia.so.0","api_version":"1.3.280"}}
EOF
export VK_ICD_FILENAMES="$ICD_FILE"

# Keep container-built packages separate from any host-created project venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-container}"

# Host-provisioned git / git-lfs
if [[ -n "'"$TOOLS_PREFIX"'" ]]; then
  export PATH="'"$TOOLS_PREFIX"'/bin:$PATH"
  # Help HTTPS certs if needed (conda-forge ships a cert bundle here)
  [[ -f "'"$TOOLS_PREFIX"'/ssl/cert.pem" ]] && export SSL_CERT_FILE="'"$TOOLS_PREFIX"'/ssl/cert.pem"
fi

# sanity (optional):
# git --version && git lfs version >&2 || true
# git lfs install --skip-repo >/dev/null 2>&1 || true

# run the user command
exec "$@"
' bash "$@"
