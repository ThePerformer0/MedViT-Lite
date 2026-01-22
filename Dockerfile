# Utilisation d'une image Python avec support CPU/GPU léger
FROM python:3.9-slim

# Dépendances système pour OpenCV et le traitement d'image
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'architecture complète
COPY models/ ./models/
COPY main.py .

# Commande par défaut
CMD ["python", "main.py"]