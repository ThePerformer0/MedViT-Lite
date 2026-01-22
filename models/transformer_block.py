import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Module 3 : Transforme une image en une séquence de vecteurs (patches).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, Embed_Dim, H/P, W/P]
        x = x.flatten(2) # [B, Embed_Dim, N]
        x = x.transpose(1, 2) # [B, N, Embed_Dim]
        return x

class MultiHeadAttention(nn.Module):
    """
    Mécanisme de Self-Attention avec retour des scores pour l'explicabilité.
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # Voici les scores d'attention

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        # IMPORTANT : On retourne le résultat ET les scores d'attention
        return out, attn

class PatchMerging(nn.Module):
    """
    Réduction de résolution pour l'architecture hiérarchique.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class MedViTBlock(nn.Module):
    """
    Bloc principal combinant Normalisation, Attention et MLP.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Attention avec connexion résiduelle
        # On intercepte attn_scores ici
        res, attn_scores = self.attn(self.norm1(x))
        x = x + res
        
        # MLP avec connexion résiduelle
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_scores