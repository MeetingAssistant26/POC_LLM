# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY main.py .

# Expose the HTTP port (default 8000, overridden at runtime)
EXPOSE 8000

# Health check against the /healthz endpoint
HEALTHCHECK --interval=10s --timeout=10s --start-period=45s --retries=12 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=5)" || exit 1

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
