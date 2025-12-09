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
COPY app ./app
COPY models ./models

# (éventuels autres fichiers)
# COPY tests ./tests
# COPY README.md .

EXPOSE 8000

RUN cd /app
RUN mkdir -p ./preprocessing

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

