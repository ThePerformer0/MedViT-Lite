from models.frame_selector import FrameSelector
from models.spatial_selector import SpatialSelector
import cv2

# Initialisation
f_selector = FrameSelector()
s_selector = SpatialSelector()

def process_pipeline(frame):
    # Étape 1 : Changement significatif ?
    significant, score = f_selector.is_significant(frame)
    
    if significant:
        # Étape 2 : Si oui, on extrait la zone utile
        roi, coords = s_selector.extract_roi(frame)
        return roi, "Traité"
    else:
        return None, "Réutilisation Cache"