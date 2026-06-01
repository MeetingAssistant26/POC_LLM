# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Install ffmpeg and build deps required by soundfile / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    pkg-config \
    python3-dev \
    libffi-dev \
    build-essential \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --upgrade pip

# Install torch/torchaudio first from CPU index to avoid dependency resolution conflicts
# that can cause pip to select an older faster-whisper -> av without pre-built wheels
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (skip torch/torchaudio since they are already installed)
RUN grep -v -E '^(torch|torchaudio)' requirements.txt > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# Force CPU-only torch/torchaudio to avoid CUDA runtime deps and maintain pinned versions
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps

# Copy service code
COPY main.py .

# Expose the HTTP port
EXPOSE 8000

# Health check against the /healthz endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Run on CPU; int8 is set in code
ENV OMP_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
