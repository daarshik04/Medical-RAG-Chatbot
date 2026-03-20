## Parent image
## FIX 1: Changed python:3.10-slim to python:3.12-slim to match the project's
##         target Python version (3.12.3). Using 3.10 would cause subtle
##         incompatibilities with modern LangChain packages built for 3.12.
FROM python:3.12-slim

## Essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ## FIX 2: Prevents HuggingFace sentence-transformers from trying to write
    ##         to ~/.cache which may not be writable in some container environments.
    HF_HOME=/app/.cache/huggingface \
    ## FIX 3: Prevents pip from checking for newer versions on every install
    ##         (speeds up builds and avoids network calls at build time).
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

## Work directory inside the docker container
WORKDIR /app

## Installing system dependencies
## FIX 4: Added libgomp1 — required by faiss-cpu for OpenMP thread support.
##         Without it faiss silently falls back or crashes at runtime.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

## FIX 5: Copy and install dependencies BEFORE copying the full project.
##         Docker layer-caches the pip install step — if only your code changes
##         (not requirements.txt), Docker reuses the cached layer and skips
##         reinstalling all packages. This makes rebuilds much faster.
COPY requirements.txt .
COPY setup.py .
RUN pip install -e .

COPY . .

## FIX 6: Create cache and logs directories with correct permissions
##         so the app can write to them at runtime without errors.
RUN mkdir -p /app/.cache/huggingface /app/logs

## Expose only flask port
EXPOSE 5000

CMD ["python", "app/application.py"]
