#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WAKEWORD_ENTRYPOINT=python "${script_dir}/docker-run.sh" - <<'PY'
from pathlib import Path
import json
import os

import onnxruntime as ort

artifact_dir = Path("wakeword/artifacts")
required = [
    artifact_dir / "melspectrogram.onnx",
    artifact_dir / "embedding_model.onnx",
]
require_classifier = os.environ.get("WAKEWORD_REQUIRE_CLASSIFIER") == "1"
if require_classifier:
    required.append(artifact_dir / "hey_assistant.onnx")
optional = [path for path in [artifact_dir / "hey_assistant.onnx"] if path not in required]

for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Required ONNX artifact missing or empty: {path}")

checked = []
for path in required + [p for p in optional if p.exists()]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    checked.append(
        {
            "path": str(path),
            "inputs": [(i.name, i.shape, i.type) for i in session.get_inputs()],
            "outputs": [(o.name, o.shape, o.type) for o in session.get_outputs()],
        }
    )

for item in checked:
    print(f"OK {item['path']}")
    print(f"  inputs: {item['inputs']}")
    print(f"  outputs: {item['outputs']}")

if require_classifier:
    eval_path = artifact_dir / "hey_assistant_eval.json"
    manifest_path = artifact_dir / "hey_assistant_manifest.json"
    for path, label in [(eval_path, "eval metrics JSON"), (manifest_path, "artifact manifest")]:
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Required {label} missing or empty: {path}")

    eval_metrics = json.loads(eval_path.read_text())
    required_eval_keys = {
        "recall",
        "fpph",
        "threshold",
        "n_positive",
        "n_negative",
        "validation_hours",
    }
    missing_eval_keys = required_eval_keys - set(eval_metrics)
    if missing_eval_keys:
        raise SystemExit(f"Eval metrics missing keys: {sorted(missing_eval_keys)}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("profile") == "initial" and manifest.get("productionQuality") is not False:
        raise SystemExit("Initial artifact manifest must mark productionQuality=false")
    if manifest.get("profile") == "initial" and "Reduced Docker-contained initial artifact" not in manifest.get("note", ""):
        raise SystemExit("Initial artifact manifest must include the reduced-subset caveat")
    manifest_artifacts = manifest.get("artifacts", {})
    for name in ["melspectrogram.onnx", "embedding_model.onnx", "hey_assistant.onnx", "hey_assistant_eval.json"]:
        if name not in manifest_artifacts:
            raise SystemExit(f"Manifest missing artifact entry: {name}")
    print(f"OK {eval_path}")
    print(
        "  metrics: "
        f"recall={eval_metrics['recall']} fpph={eval_metrics['fpph']} "
        f"threshold={eval_metrics['threshold']} "
        f"n_positive={eval_metrics['n_positive']} n_negative={eval_metrics['n_negative']} "
        f"validation_hours={eval_metrics['validation_hours']}"
    )
    print(f"OK {manifest_path}")
    print(
        "  provenance: "
        f"profile={manifest.get('profile')} productionQuality={manifest.get('productionQuality')} "
        f"revision={manifest.get('livekitWakewordRevision')} config={manifest.get('configPath')}"
    )
PY
