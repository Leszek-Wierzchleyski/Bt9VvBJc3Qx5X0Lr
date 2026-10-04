# Quantised Llama Docker patch

This patch replaces the Transformers FP32 Llama 3.1 8B loader with llama.cpp and a Q4_K_M GGUF model.

## 1. Copy the replacement files

Copy these files into the existing `Apziva_Authenticated_Security_Bundle_FINAL` folder, replacing the files with the same names:

- `Apziva_LLM_Interface_Authenticated.py`
- `Apziva_GUI_Authenticated.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

Do not copy the `models` directory from this patch; the model is downloaded separately.

## 2. Download the model

From the project root:

```bash
mkdir -p models
curl -L --fail --progress-bar \
  "https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true" \
  -o models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

The Q4_K_M file is about 4.92 GB.

## 3. Git safety

Add this to `.gitignore` if it is not already present:

```gitignore
models/
```

The GGUF model should not be committed to GitHub.

## 4. Build and run

```bash
docker compose down
docker compose up --build
```

Then open:

http://localhost:8502

The Docker Compose file mounts the model read-only at `/app/models` and sets `LLAMA_MODEL_PATH` accordingly.
