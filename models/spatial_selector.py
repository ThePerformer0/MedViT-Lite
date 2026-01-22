import cv2
import numpy as np

class SpatialSelector:
    """
    Module 2 : Sélection de zones (Spatial Sparsification).
    Objectif : Isoler la zone utile de l'échographie pour réduire les calculs du Transformer.
    """
    def __init__(self, padding=10):
        self.padding = padding

    def extract_roi(self, frame):
        """
        Détecte la boîte englobante de la zone active de l'échographie.
        """
        # 1. Conversion en gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Seuillage pour isoler les zones claires (l'organe) du fond noir
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # 3. Trouver les contours de la zone active
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame # Retourne l'image entière si rien n'est trouvé

        # 4. Prendre le plus grand contour (la zone d'examen)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # 5. Découper (Crop) avec un peu de marge
        roi = frame[max(0, y-self.padding):y+h+self.padding, 
                    max(0, x-self.padding):x+w+self.padding]
        
        return roi, (x, y, w, h)