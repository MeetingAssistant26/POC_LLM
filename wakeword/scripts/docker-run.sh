#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
image_name="${WAKEWORD_IMAGE:-meetingassistant-livekit-wakeword:local}"
entrypoint="${WAKEWORD_ENTRYPOINT:-livekit-wakeword}"

if ! docker image inspect "${image_name}" >/dev/null 2>&1; then
  "${script_dir}/docker-build.sh"
fi

mkdir -p \
  "${repo_root}/wakeword/data" \
  "${repo_root}/wakeword/output" \
  "${repo_root}/wakeword/logs" \
  "${repo_root}/wakeword/.cache/home" \
  "${repo_root}/wakeword/.cache/huggingface" \
  "${repo_root}/wakeword/.cache/xdg" \
  "${repo_root}/wakeword/.cache/torchinductor"

docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  --entrypoint "${entrypoint}" \
  -e HOME=/workspace/wakeword/.cache/home \
  -e HF_HOME=/workspace/wakeword/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e XDG_CACHE_HOME=/workspace/wakeword/.cache/xdg \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/wakeword/.cache/torchinductor \
  -e USER=meetingassistant \
  -e LOGNAME=meetingassistant \
  -e WAKEWORD_REQUIRE_CLASSIFIER="${WAKEWORD_REQUIRE_CLASSIFIER:-}" \
  -v "${repo_root}:/workspace" \
  -w /workspace \
  "${image_name}" \
  "$@"
