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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections pour Query, Key, et Value
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape # Batch, Nb Patches, Embedding Dim
        
        # 1. Générer Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Calculer les scores d'attention (Q * K^T)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # Normalisation pour avoir des probabilités

        # 3. Appliquer l'attention aux valeurs (V)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 4. Projection finale
        x = self.proj(x)
        return x