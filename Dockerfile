FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LLAMA_MODEL_PATH=/app/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

WORKDIR /app

# Build/runtime dependencies for llama-cpp-python and scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && CMAKE_BUILD_PARALLEL_LEVEL=4 pip install -r requirements.txt

# Copy application code and non-secret runtime assets.
COPY . .

# The GGUF model is deliberately supplied as a read-only runtime volume rather
# than baked into the image or committed to source control.
RUN mkdir -p /app/models

EXPOSE 8502

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8502/_stcore/health', timeout=3)" || exit 1

CMD ["streamlit", "run", "Apziva_GUI_Authenticated.py", "--server.address=0.0.0.0", "--server.port=8502"]
