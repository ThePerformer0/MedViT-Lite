import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Transforme une image en une séquence de vecteurs (patches).
    C'est la porte d'entrée de ton Transformer.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # On utilise une convolution pour découper et projeter en même temps
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        x = self.proj(x) # [B, Embed_Dim, H/P, W/P]
        x = x.flatten(2) # [B, Embed_Dim, Number_of_Patches]
        x = x.transpose(1, 2) # [B, Number_of_Patches, Embed_Dim]
        return x