#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
config="${WAKEWORD_CONFIG:-wakeword/configs/hey_assistant.yaml}"
log_dir="${repo_root}/wakeword/logs"
mkdir -p "${log_dir}"

resolve_config_info() {
  WAKEWORD_ENTRYPOINT=python "${script_dir}/docker-run.sh" - "${config}" <<'PY'
from pathlib import Path
import sys
from livekit.wakeword.config import load_config

config = load_config(sys.argv[1])
model_output_dir = Path(config.model_output_dir)
data_path = Path(config.data_path)
print(config.model_name)
print(model_output_dir)
print(data_path)
PY
}

to_host_path() {
  local container_path="$1"
  case "${container_path}" in
    /workspace/*)
      printf '%s\n' "${repo_root}/${container_path#/workspace/}"
      ;;
    /*)
      echo "Unsupported absolute config path: ${container_path}" >&2
      return 1
      ;;
    ./*)
      printf '%s\n' "${repo_root}/${container_path#./}"
      ;;
    *)
      printf '%s\n' "${repo_root}/${container_path}"
      ;;
  esac
}

config_info="$(resolve_config_info)"
model_name="$(printf '%s\n' "${config_info}" | sed -n '1p')"
model_output_dir="$(printf '%s\n' "${config_info}" | sed -n '2p')"
data_path="$(printf '%s\n' "${config_info}" | sed -n '3p')"

if [[ -z "${model_name}" || -z "${model_output_dir}" || -z "${data_path}" ]]; then
  echo "Could not resolve model output path from config: ${config}" >&2
  exit 1
fi

host_model_dir="$(to_host_path "${model_output_dir}")"
host_data_dir="$(to_host_path "${data_path}")"

echo "==> wakeword config: ${config}"
echo "==> model: ${model_name}"
echo "==> data dir: ${host_data_dir}"
echo "==> output dir: ${host_model_dir}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "${path}" ]]; then
    echo "Expected ${label} missing or empty: ${path}" >&2
    exit 1
  fi
}

run_stage_with_config_arg() {
  local stage="$1"
  echo "==> livekit-wakeword ${stage} ${config}"
  "${script_dir}/docker-run.sh" "${stage}" "${config}" 2>&1 | tee "${log_dir}/${stage}.log"
}

run_setup() {
  echo "==> livekit-wakeword setup --config ${config}"
  "${script_dir}/docker-run.sh" setup --config "${config}" 2>&1 | tee "${log_dir}/setup.log"
}

if [[ "${WAKEWORD_SKIP_SETUP:-}" == "1" ]]; then
  echo "==> skipping setup because WAKEWORD_SKIP_SETUP=1"
else
  run_setup
fi
require_file \
  "${host_data_dir}/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
  "ACAV100M training features"
require_file \
  "${host_data_dir}/features/validation_set_features.npy" \
  "validation features"
run_stage_with_config_arg generate
run_stage_with_config_arg augment
run_stage_with_config_arg train
run_stage_with_config_arg export
run_stage_with_config_arg eval

classifier="${host_model_dir}/${model_name}.onnx"
eval_json="${host_model_dir}/${model_name}_eval.json"
require_file "${classifier}" "exported classifier"
require_file "${eval_json}" "eval metrics JSON"

cp "${classifier}" "${repo_root}/wakeword/artifacts/${model_name}.onnx"
cp "${eval_json}" "${repo_root}/wakeword/artifacts/${model_name}_eval.json"
WAKEWORD_ENTRYPOINT=python "${script_dir}/docker-run.sh" \
  wakeword/scripts/write-artifact-manifest.py "${config}"
