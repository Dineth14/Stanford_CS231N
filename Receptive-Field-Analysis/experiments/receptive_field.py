"""
receptive_field.py — Effective & Theoretical Receptive Field analysis.

Methods
-------
1. Gradient-based Effective Receptive Field (ERF)
   • Backpropagate from the centre output neuron to the input.
   • |gradient| = ERF map.  Fit 2-D Gaussian to measure sigma.

2. Theoretical Receptive Field (formula-based)
   • CNN:   RF_L = 1 + Σ_i (k_i − 1) · Π_{j<i} s_j  = 9 for our config
   • ViT:   global from layer 1 (all-pairs attention)  = H × W = 4096
   • Mamba: causal linear growth ≈ layer × (4 × 4 patch) per-layer reach

Outputs
-------
  data/receptive_fields.json
"""

import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

# Make experiments/ importable when run directly
sys.path.insert(0, os.path.dirname(__file__))
from models import get_all_models


# ---------------------------------------------------------------------------
# Theoretical RF formulas
# ---------------------------------------------------------------------------

def theoretical_rf_cnn(num_layers: int = 4, kernel: int = 3, stride: int = 1) -> int:
    """RF_L = 1 + Σ_i (k-1) * Π_{j<i} s  (all strides=1 → RF = 1 + L*(k-1))."""
    rf = 1
    for i in range(num_layers):
        stride_prod = stride ** i          # product of strides before layer i
        rf += (kernel - 1) * stride_prod
    return rf                              # 9


def theoretical_rf_vit(img_size: int = 64) -> int:
    """ViT attends all-pairs from layer 1 → global RF = H*W."""
    return img_size * img_size             # 4096


def theoretical_rf_mamba(num_layers: int = 4, patch_size: int = 4,
                          img_size: int = 64) -> int:
    """
    Mamba processes tokens causally.  Within one layer each token can
    integrate the full causal history.  Across 4 layers the effective
    spatial reach grows linearly: roughly num_layers × patch_size² pixels.
    Upper-bound: full image (H×W).
    """
    per_layer = patch_size * patch_size        # pixels per patch = 16
    reach     = min(num_layers * per_layer * 4,
                    img_size * img_size)       # ~256 → 4096 cap
    return reach


# ---------------------------------------------------------------------------
# ERF helpers
# ---------------------------------------------------------------------------

def _center_output(model, model_name: str, x: torch.Tensor, img_size: int):
    """
    Forward pass and return the scalar output at the 'centre' of the
    last feature map.  Shape depends on architecture.
    """
    out, features = model(x, return_features=True)
    feat = features[-1]          # last layer features

    if model_name == 'cnn':
        # feat: (1, C, H, W)  — same spatial size as input
        ch, fh, fw = feat.shape[1:]
        cy, cx = fh // 2, fw // 2
        return feat[0, :, cy, cx].sum()

    elif model_name == 'vit':
        # feat: (1, N+1, D)  — tokens; CLS at 0, patches 1..N
        patch_size  = 8
        grid        = img_size // patch_size          # 8
        cy_patch    = grid // 2
        cx_patch    = grid // 2
        idx         = cy_patch * grid + cx_patch + 1  # +1 for CLS
        return feat[0, idx, :].sum()

    elif model_name == 'mamba':
        # feat: (1, d_model, pH, pW)
        ph, pw = feat.shape[2], feat.shape[3]
        cy, cx = ph // 2, pw // 2
        return feat[0, :, cy, cx].sum()

    raise ValueError(f"Unknown model: {model_name}")


def compute_erf(model, model_name: str,
                img_size: int = 64, device: str = 'cpu') -> np.ndarray:
    """
    Compute the Effective Receptive Field via input-gradient backprop.

    Returns
    -------
    erf : (img_size, img_size) float32 array, normalised to [0, 1].
    """
    model.eval()
    x = torch.randn(1, 3, img_size, img_size,
                    requires_grad=True, device=device)

    centre = _center_output(model, model_name, x, img_size)
    centre.backward()

    grad = x.grad.data                           # (1, 3, H, W)
    erf  = grad.abs().sum(dim=1).squeeze(0)     # (H, W)
    erf  = erf.cpu().numpy().astype(np.float32)

    # Smooth slightly to reduce noise
    erf = gaussian_filter(erf, sigma=0.5)

    # Normalise
    erf_min, erf_max = erf.min(), erf.max()
    if erf_max > erf_min:
        erf = (erf - erf_min) / (erf_max - erf_min)
    return erf


# ---------------------------------------------------------------------------
# Gaussian fit
# ---------------------------------------------------------------------------

def _gaussian_2d(xy, x0, y0, sigma, amplitude):
    x, y = xy
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


def fit_gaussian_sigma(erf: np.ndarray) -> float:
    """Fit a 2-D isotropic Gaussian to the ERF and return sigma (pixels)."""
    H, W = erf.shape
    y_grid, x_grid = np.mgrid[:H, :W]
    xy = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z  = erf.ravel().astype(np.float64)

    try:
        popt, _ = curve_fit(
            _gaussian_2d, xy, z,
            p0=[W / 2, H / 2, H / 8, z.max()],
            maxfev=8000,
        )
        return float(abs(popt[2]))
    except Exception:
        # Fall back to std of gradient-weighted coordinates
        w   = z / (z.sum() + 1e-12)
        x_m = (x_grid.ravel() * w).sum()
        y_m = (y_grid.ravel() * w).sum()
        var = (w * ((x_grid.ravel() - x_m) ** 2 +
                    (y_grid.ravel() - y_m) ** 2)).sum()
        return float(math.sqrt(max(var, 0.0)))


# ---------------------------------------------------------------------------
# ERF radius (95 % energy)
# ---------------------------------------------------------------------------

def erf_radius(erf: np.ndarray, threshold: float = 0.95) -> float:
    """Radius (pixels) of the circle containing `threshold` of gradient energy."""
    H, W = erf.shape
    cy, cx = H / 2.0, W / 2.0
    y_grid, x_grid = np.mgrid[:H, :W]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2).ravel()
    vals = erf.ravel()

    order = np.argsort(dist)
    cum   = np.cumsum(vals[order])
    total = cum[-1]
    if total < 1e-12:
        return 0.0
    idx = np.searchsorted(cum / total, threshold)
    return float(dist[order[min(idx, len(order) - 1)]])


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_receptive_field_analysis(img_size: int = 64,
                                  out_path: str = None) -> dict:
    """
    Run ERF + theoretical RF analysis for all three models.

    Returns the full results dict and writes it to out_path as JSON.
    """
    if out_path is None:
        here     = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, '..', 'data', 'receptive_fields.json')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    models = get_all_models(num_classes=10)
    results = {}

    theoretical = {
        'cnn':   theoretical_rf_cnn(),
        'vit':   theoretical_rf_vit(img_size),
        'mamba': theoretical_rf_mamba(img_size=img_size),
    }

    for name, model in models.items():
        print(f"  [{name.upper()}] computing ERF …", flush=True)
        torch.manual_seed(0)
        erf = compute_erf(model, name, img_size=img_size)

        sigma    = fit_gaussian_sigma(erf)
        radius   = erf_radius(erf, 0.95)
        th_rf    = theoretical[name]

        results[name] = {
            'theoretical_rf':     th_rf,
            'effective_rf_radius': round(radius, 4),
            'rf_gaussian_sigma':  round(sigma, 4),
            'erf_map':            erf.tolist(),      # 64×64 float list
        }
        print(f"         theoretical={th_rf}  "
              f"effective_r={radius:.2f}  sigma={sigma:.2f}")

    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"  Saved → {out_path}")
    return results


if __name__ == '__main__':
    run_receptive_field_analysis()
