# ---------------------------------------------------------------------------
# Dockerfile
# WHAT THIS DOES: packages the app into a container so it can run anywhere
# Docker is installed, without needing to manually install Python/dependencies.
#
# HOW TO BUILD:
#   docker build -t supportai .
#
# HOW TO RUN (pass your .env file with secrets at runtime, not baked into the image):
#   docker run --env-file .env -p 8501:8501 supportai
#
# NOTE: the vector index is NOT pre-built into the image. After the first run,
# open the app, go to Admin, and click "Rebuild Index" once - or mount a
# volume for /app/vectorstore to persist it across container restarts:
#   docker run --env-file .env -p 8501:8501 -v $(pwd)/vectorstore:/app/vectorstore supportai
# ---------------------------------------------------------------------------

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by sentence-transformers/torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (better Docker layer caching - deps only
# reinstall if requirements.txt changes, not on every code edit).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
COPY . .

EXPOSE 8501

# Streamlit-specific settings for running inside a container.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
