# 1. Utiliser une version légère de Python
FROM python:3.9-slim

# 2. Installer les dépendances système pour OpenCV (indispensable sous Linux/Docker)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Créer un dossier de travail
WORKDIR /app

# 4. Copier le fichier des dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le reste du code
COPY . .

# Commande par défaut (à adapter plus tard)
CMD ["python", "models/module1_frame_selection.py"]