import torch
import torch.nn as nn
import cv2
import numpy as np
from models.frame_selector import FrameSelector
from models.spatial_selector import SpatialSelector
from models.transformer_block import PatchEmbedding, MedViTBlock
from models.temporal_module import TemporalMemory
from models.diagnostic_module import DiagnosticHead
from models.explicability import AttentionVisualizer

class MedViTLiteSystem:
    def __init__(self):
        # Initialisation de la pipeline selon ton schéma
        self.frame_selector = FrameSelector(threshold=0.96)
        self.spatial_selector = SpatialSelector(padding=10)
        
        # Hyperparamètres
        self.embed_dim = 128
        self.hidden_dim = 64
        
        # Modules Deep Learning
        self.patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=self.embed_dim)
        self.transformer = MedViTBlock(dim=self.embed_dim, num_heads=4)
        self.temporal_memory = TemporalMemory(dim=self.embed_dim, hidden_dim=self.hidden_dim)
        self.diagnostic_head = DiagnosticHead(input_dim=self.hidden_dim, num_classes=2)
        self.visualizer = AttentionVisualizer(patch_size=16)

    def process_video_stream(self, video_frames):
        """
        Traite une séquence de trames et retourne le diagnostic + incertitude.
        """
        memory_features = []
        last_attn_map = None
        last_roi = None

        for frame in video_frames:
            # 1. Sparsification Temporelle
            is_significant, _ = self.frame_selector.is_significant(frame)
            
            if is_significant:
                # 2. Sparsification Spatiale
                roi, _ = self.spatial_selector.extract_roi(frame)
                last_roi = cv2.resize(roi, (224, 224))
                
                # Préparation tenseur
                roi_t = torch.from_numpy(last_roi).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                
                # 3. Transformer Hiérarchique (Spatial)
                patches = self.patch_embed(roi_t)
                features, attn_scores = self.transformer(patches)
                
                # Pooling et stockage
                memory_features.append(features.mean(dim=1))
                last_attn_map = attn_scores
        
        if not memory_features:
            return None

        # 4. Raisonnement Temporel
        sequence = torch.stack(memory_features, dim=1)
        temporal_vector = self.temporal_memory(sequence)
        
        # 5. Diagnostic & Incertitude
        prediction, uncertainty = self.diagnostic_head.predict_with_uncertainty(temporal_vector)
        
        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "last_attn_map": last_attn_map,
            "last_roi": last_roi
        }

if __name__ == "__main__":
    print("Système MedViT-Lite initialisé. Prêt pour l'inférence.")
    # Le test réel se fera sur Colab ou CloudLab avec des données vidéo