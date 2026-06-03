# LiveKit wake-word training

This directory contains the Docker-contained training setup for the MeetingAssistant `hey_assistant` wake-word classifier. Do not run `livekit-wakeword` directly on the host for this project; build and run the container so setup, generation, augmentation, training, export, and eval use the same Linux dependency set across developer machines.

## Source

The image uses the local `livekit-wakeword` clone as its build context by default:

```sh
LIVEKIT_WAKEWORD_SOURCE=/private/tmp/livekit-wakeword-inspect \
  wakeword/scripts/docker-build.sh
```

Override `LIVEKIT_WAKEWORD_SOURCE` only when inspecting a different clone of `github.com/livekit/livekit-wakeword`.
The reviewed source revision is pinned to `5558b0a32c9316a0f22f2ba4e104658d3aa6238c`; set `LIVEKIT_WAKEWORD_REVISION` when deliberately updating the reviewed source, or `WAKEWORD_ALLOW_UNPINNED_SOURCE=1` for a one-off local inspection build.

## Config

`wakeword/configs/hey_assistant.yaml` is the production training profile based on `livekit-wakeword/configs/prod.yaml` with:

- `model_name: hey_assistant`
- `target_phrases: ["hey assistant"]`
- adversarial custom negatives: `assistant`, `hey assistance`, `hey system`, `they assistant`, `hi assistant`
- Docker-mounted paths under `wakeword/data` and `wakeword/output`

`wakeword/configs/hey_assistant.initial.yaml` keeps the same phrase/negative contract but reduces samples, augmentation rounds, model size, and training steps for an initial Docker-contained browser-integration artifact on CPU-only Docker. It uses `wakeword/data-initial`, which should be prepared inside the container from deterministic small subsets of the downloaded feature arrays plus symlinks to the downloaded Piper/background/RIR assets. The resulting `hey_assistant.onnx` is not production-quality; use it only to verify browser model loading, wake-word plumbing, and manual QA until the full production profile above replaces it.

## Run

```sh
wakeword/scripts/docker-build.sh
wakeword/scripts/run-hey-assistant-pipeline.sh
WAKEWORD_REQUIRE_CLASSIFIER=1 wakeword/scripts/validate-artifacts.sh
```

The pipeline runs:

1. `setup --config wakeword/configs/hey_assistant.yaml`
2. `generate wakeword/configs/hey_assistant.yaml`
3. `augment wakeword/configs/hey_assistant.yaml`
4. `train wakeword/configs/hey_assistant.yaml`
5. `export wakeword/configs/hey_assistant.yaml`
6. `eval wakeword/configs/hey_assistant.yaml`

The custom script intentionally runs each CLI phase separately instead of `livekit-wakeword run` so each phase has its own log and the final artifacts can be copied and checked explicitly. Override `WAKEWORD_CONFIG` to run a different checked-in config, for example:

```sh
# Prepare the reduced data directory inside Docker first.
WAKEWORD_ENTRYPOINT=python wakeword/scripts/docker-run.sh wakeword/scripts/prepare-initial-data.py

WAKEWORD_CONFIG=wakeword/configs/hey_assistant.initial.yaml \
WAKEWORD_SKIP_SETUP=1 \
  wakeword/scripts/run-hey-assistant-pipeline.sh
```

Generated datasets, features, logs, and checkpoints are intentionally ignored by Git. Browser artifacts under `wakeword/artifacts/` are intended to be tracked, including `hey_assistant.onnx`, `hey_assistant_eval.json`, and `hey_assistant_manifest.json` after export/eval. The manifest records the livekit-wakeword source revision, config path, artifact hashes, and whether the artifact came from the reduced initial profile or the production profile.
