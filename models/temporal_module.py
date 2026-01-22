import torch
import torch.nn as nn

class TemporalMemory(nn.Module):
    """
    Module 4 : Raisonnement Temporel.
    Objectif : Analyser la séquence de trames pour comprendre la dynamique.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # On utilise un GRU (Gated Recurrent Unit), plus léger qu'un LSTM
        # Idéal pour l'approche "Lite" de ton projet.
        self.gru = nn.GRU(dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1) # Pour un score de pathologie

    def forward(self, x):
        """
        x shape: [Batch, Time_Steps, Feature_Dim]
        """
        # Sortie : [Batch, Time_Steps, Hidden_Dim]
        # h_n : dernier état caché (résumé de toute la séquence)
        output, h_n = self.gru(x)
        
        # On prend le dernier état pour le diagnostic final
        last_state = h_n.squeeze(0)
        return last_state