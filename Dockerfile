FROM python:3.11-slim

# yt-dlp needs ffmpeg; Chrome cookie fallback needs chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp as a system binary (kept separate from pip deps for easy upgrades)
RUN pip install --no-cache-dir yt-dlp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline_core.py pipeline.py chat.py api.py ./

# Databases are mounted as a volume — never baked into the image
RUN mkdir -p /data/databases
ENV DATABASES_DIR=/data/databases

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
