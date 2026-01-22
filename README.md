# MedViT-Lite: Transformer Hiérarchique Adaptatif pour le Diagnostic Échographique

## 📌 Présentation du Projet
Ce projet s'inscrit dans le cadre du projet de fin d'année (Capstone Project). L'objectif est de concevoir, implémenter et évaluer une nouvelle architecture d'IA générative et agentique pour le diagnostic médical par ultrasons.

## 🚀 Innovations Majeures
* **Sparsification Dynamique :** Sélection intelligente des zones (patches) et mise en cache des trames pour une exécution en temps réel sur périphériques "edge".
* **Architecture Hiérarchique :** Utilisation de Transformers spatio-temporels pour capturer les détails anatomiques et les dépendances temporelles.
* **IA de Confiance :** Intégration de modules d'explicabilité (GradCAM++) et de gestion de l'incertitude (Bayésien).

## 🛠️ Objectifs de l'Architecture
1. **Raisonnement Temporel :** Améliorer la cohérence du diagnostic sur les flux vidéo.
2. **Efficacité :** Optimisation pour les contraintes de déploiement réel (latence, mémoire).
3. **Interprétabilité :** Fournir des justifications visuelles exploitables par les cliniciens.

## 📂 Structure du Dépôt
- `models/`: Définition des modules MedViT-Lite (Module 1 à 6).
- `src/`: Scripts de prétraitement et d'entraînement.
- `notebooks/`: Expérimentations sur Google Colab.
- `docker/`: Fichiers de configuration pour la reproductibilité.

## 🧪 Datasets de Référence
Le modèle est conçu pour être validé sur des bases de données telles que **EchoNet-Dynamic**, **MIMIC-CXR**, ou **EndoVis**.