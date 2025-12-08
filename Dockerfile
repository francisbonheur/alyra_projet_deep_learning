FROM python:3.11-slim

WORKDIR /app

# (Optionnel mais recommandé si tu utilises scikit-learn ou autres libs compilées)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Installer uv
RUN pip install --no-cache-dir uv

# Copier le fichier de config des deps
COPY pyproject.toml .

# Installer les dépendances (sans extras dev)
RUN uv sync --no-dev

# Copier le code de l'application et les modèles
COPY main.py .
COPY app .
COPY models ../models

# (éventuels autres fichiers)
# COPY tests ./tests
# COPY README.md .

# Variables d'env pour FastAPI / uvicorn
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Lancement de l'API FastAPI avec uvicorn
# On suppose que dans app/app.py tu as: app = FastAPI(...)
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]