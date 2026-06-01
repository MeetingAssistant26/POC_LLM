# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# Install Python deps first for layer caching
COPY services/llm/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY services/__init__.py ./services/__init__.py
COPY services/ai_debug.py ./services/ai_debug.py
COPY services/llm/main.py .

# Expose the HTTP port
EXPOSE 8000

# Health check against the /healthz endpoint
HEALTHCHECK --interval=10s --timeout=10s --start-period=45s --retries=12 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=5)" || exit 1

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
