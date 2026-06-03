#!/usr/bin/env python3
"""Write provenance metadata for exported wake-word browser artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from livekit.wakeword.config import load_config

DEFAULT_LIVEKIT_WAKEWORD_REVISION = "5558b0a32c9316a0f22f2ba4e104658d3aa6238c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: write-artifact-manifest.py <wakeword-config.yaml>")

    config_path = Path(sys.argv[1])
    config = load_config(config_path)
    model_name = config.model_name
    artifact_dir = Path("/workspace/wakeword/artifacts")
    eval_path = artifact_dir / f"{model_name}_eval.json"
    model_path = artifact_dir / f"{model_name}.onnx"
    mel_path = artifact_dir / "melspectrogram.onnx"
    embedding_path = artifact_dir / "embedding_model.onnx"

    for path in [mel_path, embedding_path, model_path, eval_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"Required artifact missing or empty: {path}")

    eval_metrics = json.loads(eval_path.read_text())
    profile = "initial" if ".initial" in config_path.name else "production"
    manifest = {
        "modelName": model_name,
        "profile": profile,
        "productionQuality": profile == "production",
        "note": (
            "Reduced Docker-contained initial artifact for browser integration/manual testing; "
            "replace with a full production-profile artifact before production use."
            if profile == "initial"
            else "Production-profile artifact."
        ),
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "livekitWakewordRevision": os.environ.get(
            "LIVEKIT_WAKEWORD_REVISION", DEFAULT_LIVEKIT_WAKEWORD_REVISION
        ),
        "configPath": str(config_path),
        "config": {
            "targetPhrases": config.target_phrases,
            "customNegativePhrases": config.custom_negative_phrases,
            "nSamples": config.n_samples,
            "nSamplesVal": config.n_samples_val,
            "nBackgroundSamples": config.n_background_samples,
            "nBackgroundSamplesVal": config.n_background_samples_val,
            "dataDir": config.data_dir,
            "outputDir": config.output_dir,
            "augmentationRounds": config.augmentation.rounds,
            "modelType": config.model.model_type.value,
            "modelSize": config.model.model_size.value,
            "steps": config.steps,
            "targetFpPerHour": config.target_fp_per_hour,
        },
        "eval": eval_metrics,
        "artifacts": {
            "melspectrogram.onnx": {"sha256": sha256(mel_path), "bytes": mel_path.stat().st_size},
            "embedding_model.onnx": {
                "sha256": sha256(embedding_path),
                "bytes": embedding_path.stat().st_size,
            },
            f"{model_name}.onnx": {"sha256": sha256(model_path), "bytes": model_path.stat().st_size},
            f"{model_name}_eval.json": {
                "sha256": sha256(eval_path),
                "bytes": eval_path.stat().st_size,
            },
        },
    }

    manifest_path = artifact_dir / f"{model_name}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
