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

class PatchMerging(nn.Module):
    """
    Module de transition : réduit la résolution spatiale et augmente la dimension.
    C'est ce qui crée la 'Hiérarchie' dans ton Module 3.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Réduction de dimension après fusion (4*dim -> 2*dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """
        x: [Batch, H*W, Dim] (Séquence de patches)
        H, W: Résolution actuelle de l'image
        """
        B, L, C = x.shape
        assert L == H * W, "La taille de la séquence ne correspond pas à H*W"

        x = x.view(B, H, W, C)

        # On prend 1 pixel sur 2 pour créer 4 sous-images
        x0 = x[:, 0::2, 0::2, :] # En haut à gauche
        x1 = x[:, 1::2, 0::2, :] # En bas à gauche
        x2 = x[:, 0::2, 1::2, :] # En haut à droite
        x3 = x[:, 1::2, 1::2, :] # En bas à droite
        
        # On les colle ensemble (concaténation)
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # Mise à plat : [B, (H/2)*(W/2), 4*C]

        # Normalisation et réduction de dimension
        x = self.norm(x)
        x = self.reduction(x)

        return x # Nouvelle dimension: [B, L/4, 2*C]
    
class MedViTBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # Petit réseau Feed-Forward (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(), # Activation moderne pour les Transformers
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        # Connexion résiduelle 1 (Attention)
        x = x + self.attn(self.norm1(x))
        # Connexion résiduelle 2 (MLP)
        x = x + self.mlp(self.norm2(x))
        return x