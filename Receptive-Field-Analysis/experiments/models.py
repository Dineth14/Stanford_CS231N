"""
models.py — Three 4-layer deep learning models for RF / frequency analysis.

Models
------
  SmallCNN    : 4-layer CNN, no downsampling (3→32→64→128→256)
  SmallViT    : 4-block Vision Transformer (patch_size=8, embed_dim=128)
  SmallMamba  : 4-layer Mamba/SSM (patch_size=4, d_model=64, state_dim=64)

All models accept (B, 3, 64, 64) input and are CPU-only compatible.
Call model(x, return_features=True) to get (logits, [feat_per_layer]).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. SMALL CNN
# ===========================================================================

class SmallCNN(nn.Module):
    """
    4 convolutional layers with NO spatial downsampling.
    Channel progression: 3 → 32 → 64 → 128 → 256
    Kernel=3, stride=1, padding=1 → spatial dims stay 64×64 throughout.
    Global Average Pooling → Linear classifier.

    Theoretical Receptive Field after L layers with k=3, s=1:
        RF_L = 1 + L*(k-1) = 1 + 4*2 = 9
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        ch = [in_channels, 32, 64, 128, 256]
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch[i], ch[i + 1],
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ch[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(4)
        ])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x)                        # (B, C, 64, 64)
        out = self.classifier(self.gap(x).flatten(1))
        return (out, features) if return_features else out


# ===========================================================================
# 2. VISION TRANSFORMER
# ===========================================================================

class MultiheadSelfAttention(nn.Module):
    """MHSA with stored attention weights for visualisation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor = None   # stored after each forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
                   .reshape(B, N, 3, self.num_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))          # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach().clone()   # (B, H, N, N)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiheadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """
    4-block Vision Transformer.
    patch_size=8  →  grid = 64/8 = 8  →  N = 64 patches
    embed_dim=128, heads=4, mlp_ratio=4, dropout=0.1
    CLS token + learnable positional embedding.

    Theoretical Receptive Field: global from layer 1 (all-pairs attention).
    """

    def __init__(self, img_size: int = 64, patch_size: int = 8,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 128, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 num_layers: int = 4):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size  = patch_size
        self.grid_size   = img_size // patch_size   # 8
        self.num_patches = self.grid_size ** 2      # 64
        self.embed_dim   = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def attention_maps(self):
        """List of attention weight tensors from each block."""
        return [b.attn.attn_weights for b in self.blocks]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)                       # (B, D, G, G)
        x = x.flatten(2).transpose(1, 2)              # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)              # (B, N+1, D)
        x   = self.pos_drop(x + self.pos_embed)

        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)                        # (B, N+1, D)

        x      = self.norm(x)
        logits = self.classifier(x[:, 0])             # CLS token
        return (logits, features) if return_features else logits


# ===========================================================================
# 3. MAMBA / STATE SPACE MODEL
# ===========================================================================

def selective_scan_cpu(A_bar: torch.Tensor, Bu: torch.Tensor) -> torch.Tensor:
    """
    Sequential selective scan (diagonal SSM, CPU-friendly).

    Recurrence:  h[t] = Ā[t] ⊙ h[t-1] + (B̄·u)[t],   h[-1] = 0

    Args
    ----
    A_bar : (B, L, N)  per-step diagonal state-transition values in (0, 1)
    Bu    : (B, L, N)  per-step input contribution  B̄ * u

    Returns
    -------
    h : (B, L, N)  all hidden states
    """
    B, L, N = A_bar.shape
    h  = A_bar.new_zeros(B, N)
    hs = []
    for t in range(L):
        h = A_bar[:, t] * h + Bu[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)           # (B, L, N)


class MambaBlock(nn.Module):
    """
    Simplified Mamba-style SSM block (no external mamba package required).

    Architecture
    ~~~~~~~~~~~~
    x ──► in_proj ──► split(x_in, z)
           x_in ──► depthwise_conv1d ──► SiLU
                ──► SSM (A,B,C,Δ selective) ──► D·x_in (skip)
                ──► ⊗ SiLU(z) ──► out_proj

    Discretised ZOH SSM
    ~~~~~~~~~~~~~~~~~~~
      Ā = exp(Δ·A),   B̄ = (Ā − I)/A · B
      h_t = Ā_t h_{t-1} + B̄_t x_t
      y_t = C_t · h_t + D x_t

    A is a *negative* diagonal matrix, initialised with log-spaced values
    (HiPPO-inspired). B, C, Δ are all input-dependent (selective scan).
    """

    def __init__(self, d_model: int, state_dim: int = 64, expand: int = 2):
        super().__init__()
        self.d_model   = d_model
        self.d_inner   = d_model * expand
        self.state_dim = state_dim
        dt_rank        = max(1, d_model // 16)
        self.dt_rank   = dt_rank

        # Input expansion + gating
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Depthwise conv1d for local mixing
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=3, padding=1,
                                  groups=self.d_inner, bias=True)
        # A: HiPPO-inspired diagonal, stored as log (so A = -exp(A_log))
        A_log = torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
        self.A_log    = nn.Parameter(A_log)             # (N,)
        # Selective B, C, Δ projections
        self.x_proj   = nn.Linear(self.d_inner,
                                  dt_rank + 2 * state_dim, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, self.d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, math.log(0.001))
        # Skip connection weight
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act      = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, d_model)  →  y : (B, L, d_model)"""
        B, L, _ = x.shape

        # 1. Split into inner state + gate
        xz   = self.in_proj(x)                          # (B, L, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)                   # (B, L, d_inner)

        # 2. Local context
        x_c = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_c = self.act(x_c)                             # (B, L, d_inner)

        # 3. Selective parameters
        dbc          = self.x_proj(x_c)                 # (B, L, rank+2N)
        dt_raw, B_s, C_s = torch.split(
            dbc, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))           # (B, L, d_inner) > 0

        # 4. Discretise via ZOH
        A       = -torch.exp(self.A_log)                # (N,), negative
        dt_mean = dt.mean(dim=-1, keepdim=True)         # (B, L, 1)
        A_bar   = torch.exp(dt_mean * A)                # (B, L, N), in (0,1)
        A_safe  = A.clamp(max=-1e-8)
        B_bar   = (A_bar - 1.0) / A_safe * B_s         # (B, L, N), positive

        # 5. Scan
        u      = x_c.mean(dim=-1, keepdim=True)         # (B, L, 1) scalar input
        Bu     = B_bar * u                              # (B, L, N)
        h      = selective_scan_cpu(A_bar, Bu)          # (B, L, N)

        # 6. Output + skip + gate
        y_ssm = (C_s * h).sum(dim=-1, keepdim=True)    # (B, L, 1)
        y     = y_ssm.expand(-1, -1, self.d_inner)     # (B, L, d_inner)
        y     = y + self.D * x_c                       # skip connection
        y     = y * self.act(z)                        # gating

        return self.out_proj(y)                         # (B, L, d_model)


class SmallMamba(nn.Module):
    """
    4-layer Mamba/SSM image classifier.

    Pipeline
    --------
    (B,3,64,64) → 4×4 patch embed → (B,256,d_model)
               → 4× [LayerNorm + MambaBlock + residual]
               → global avg pool → Linear classifier

    d_model=64, state_dim=64, expand=2
    Sequence length L = (64/4)² = 256 (fast sequential scan on CPU).

    Theoretical RF grows linearly with layers via SSM causality.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 patch_size: int = 4, img_size: int = 64,
                 d_model: int = 64, state_dim: int = 64,
                 expand: int = 2, num_layers: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.d_model    = d_model
        self.grid_h     = img_size // patch_size    # 16
        self.grid_w     = img_size // patch_size    # 16

        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size,
                                     stride=patch_size, bias=True)
        self.norm_pre = nn.LayerNorm(d_model)
        self.layers   = nn.ModuleList([
            nn.ModuleDict({
                'norm':  nn.LayerNorm(d_model),
                'block': MambaBlock(d_model, state_dim, expand),
            })
            for _ in range(num_layers)
        ])
        self.norm_post  = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        B, C, H, W = x.shape
        pH = H // self.patch_size
        pW = W // self.patch_size

        x = self.patch_embed(x)                      # (B, d_model, pH, pW)
        x = x.flatten(2).transpose(1, 2)             # (B, L, d_model)
        x = self.norm_pre(x)

        features = []
        for ld in self.layers:
            x = x + ld['block'](ld['norm'](x))       # pre-norm + residual
            feat = x.transpose(1, 2).reshape(B, self.d_model, pH, pW)
            features.append(feat)                    # (B, d_model, pH, pW)

        x      = self.norm_post(x)
        logits = self.classifier(x.mean(dim=1))      # GAP over sequence
        return (logits, features) if return_features else logits


# ===========================================================================
# Utility
# ===========================================================================

def get_all_models(num_classes: int = 10) -> dict:
    """Instantiate all three models and return as {name: model}."""
    return {
        'cnn':   SmallCNN(num_classes=num_classes),
        'vit':   SmallViT(num_classes=num_classes),
        'mamba': SmallMamba(num_classes=num_classes),
    }


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(2, 3, 64, 64)
    for name, model in get_all_models().items():
        model.eval()
        with torch.no_grad():
            out, feats = model(x, return_features=True)
        feat_shapes = [tuple(f.shape) for f in feats]
        print(f"{name:6s}: logits={tuple(out.shape)}, "
              f"features={feat_shapes}")
