import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagnosticHead(nn.Module):
    """
    Module 6 : Diagnostic Multi-Sorties & Incertitude.
    """
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(p=0.3) # Utilisé pour le Monte Carlo Dropout
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, mc_dropout=False):
        """
        x: vecteur issu de la mémoire temporelle (Module 4).
        mc_dropout: si True, active le dropout même en mode évaluation.
        """
        x = F.relu(self.fc1(x))
        
        # Force le dropout si on veut mesurer l'incertitude
        if mc_dropout:
            x = F.dropout(x, p=0.3, training=True)
        else:
            x = self.dropout(x)
            
        logits = self.classifier(x)
        return logits

    def predict_with_uncertainty(self, x, n_iter=10):
        """
        Exécute n_iter prédictions pour calculer la moyenne et la variance (incertitude).
        """
        results = torch.stack([torch.softmax(self.forward(x, mc_dropout=True), dim=-1) for _ in range(n_iter)])
        
        mean_prediction = results.mean(dim=0)
        uncertainty = results.std(dim=0) # L'écart-type représente l'incertitude
        
        return mean_prediction, uncertainty