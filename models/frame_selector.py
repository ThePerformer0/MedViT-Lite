import cv2
from skimage.metrics import structural_similarity as ssim

class FrameSelector:
    """
    Module 1 : Sélection de trames pour MedViT-Lite.
    Innovation : Mise en cache sélective pour l'efficacité (edge computing).
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.last_frame = None

    def is_significant(self, frame):
        # Si première trame, on traite obligatoirement
        if self.last_frame is None:
            self.last_frame = frame
            return True, 1.0

        # Prétraitement rapide pour le calcul de similarité
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcul du SSIM (Structural Similarity Index)
        score, _ = ssim(gray_last, gray_curr, full=True)

        if score < self.threshold:
            self.last_frame = frame
            return True, score # Traitement complet si changement significatif
        
        return False, score # "Non" -> Réutilisation du Cache