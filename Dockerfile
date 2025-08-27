FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose API port
EXPOSE 8000

# Default environment (can be overridden by .env / compose)
ENV WHISPER_MODEL=large-v3 \
    LANGUAGE=nl \
    COACH_THRESHOLD=0.72 \
    SAMPLE_RATE=16000 \
    VAD_FRAME_MS=30 \
    MIN_SEG_DUR=0.5 \
    MERGE_GAP=0.2 \
    OFFLINE_ONLY=true \
    WHISPER_LOCAL_DIR=app/store/models/faster-whisper \
    SB_ECAPA_LOCAL_DIR=app/store/models/spkrec-ecapa-voxceleb

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
