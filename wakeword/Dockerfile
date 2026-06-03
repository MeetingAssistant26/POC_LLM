# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      espeak-ng \
      ffmpeg \
      git \
      libsndfile1 \
      portaudio19-dev \
      sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/livekit-wakeword
COPY --from=livekit_wakeword . /opt/livekit-wakeword
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[train,eval,export]"

WORKDIR /workspace
ENTRYPOINT ["livekit-wakeword"]
