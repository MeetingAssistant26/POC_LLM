#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
livekit_wakeword_source="${LIVEKIT_WAKEWORD_SOURCE:-/private/tmp/livekit-wakeword-inspect}"
image_name="${WAKEWORD_IMAGE:-meetingassistant-livekit-wakeword:local}"
expected_source_revision="${LIVEKIT_WAKEWORD_REVISION:-5558b0a32c9316a0f22f2ba4e104658d3aa6238c}"

if [[ ! -d "${livekit_wakeword_source}" ]]; then
  echo "LIVEKIT_WAKEWORD_SOURCE does not exist: ${livekit_wakeword_source}" >&2
  exit 1
fi

if git -C "${livekit_wakeword_source}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_source_revision="$(git -C "${livekit_wakeword_source}" rev-parse HEAD)"
  if [[ "${actual_source_revision}" != "${expected_source_revision}" ]]; then
    cat >&2 <<EOF
LIVEKIT_WAKEWORD_SOURCE is at ${actual_source_revision}, expected ${expected_source_revision}.
Set LIVEKIT_WAKEWORD_REVISION to the reviewed revision or set
WAKEWORD_ALLOW_UNPINNED_SOURCE=1 to build from an unreviewed source checkout.
EOF
    if [[ "${WAKEWORD_ALLOW_UNPINNED_SOURCE:-}" != "1" ]]; then
      exit 1
    fi
  fi
  echo "Using livekit-wakeword source ${livekit_wakeword_source} @ ${actual_source_revision}"
else
  echo "LIVEKIT_WAKEWORD_SOURCE is not a Git checkout: ${livekit_wakeword_source}" >&2
  if [[ "${WAKEWORD_ALLOW_UNPINNED_SOURCE:-}" != "1" ]]; then
    exit 1
  fi
fi

DOCKER_BUILDKIT=1 docker build \
  --build-context "livekit_wakeword=${livekit_wakeword_source}" \
  -f "${repo_root}/wakeword/Dockerfile" \
  -t "${image_name}" \
  "${repo_root}"
