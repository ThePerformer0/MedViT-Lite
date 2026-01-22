import torch
import numpy as np
import cv2

class AttentionVisualizer:
    """
    Module 5 : Explicabilité.
    Génère des cartes de chaleur montrant les zones d'intérêt du modèle.
    """
    def __init__(self, patch_size=16):
        self.patch_size = patch_size

    def generate_heatmap(self, attention_scores, img_size=(224, 224)):
        """
        attention_scores: [Heads, Nb_Patches, Nb_Patches]
        On fait la moyenne des têtes et on extrait l'attention globale.
        """
        # 1. Moyenne sur les têtes d'attention
        avg_attn = attention_scores.mean(dim=0)
        
        # 2. On prend l'attention que chaque patch porte à l'ensemble
        # (Simplification pour la visualisation)
        result = torch.mean(avg_attn, dim=0)
        
        # 3. Reshape vers la grille de patches (ex: 14x14)
        side = int(np.sqrt(result.size(0)))
        heatmap = result.reshape(side, side).detach().cpu().numpy()
        
        # 4. Upsample pour revenir à la taille de l'image (224x224)
        heatmap = cv2.resize(heatmap, img_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return np.uint8(255 * heatmap)