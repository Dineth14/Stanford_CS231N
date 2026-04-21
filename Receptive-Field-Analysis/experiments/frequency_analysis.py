"""
frequency_analysis.py — Spatial-frequency response of CNN / ViT / Mamba.

Steps
-----
1. Frequency-Sensitivity Test
   • Generate sinusoidal gratings at f ∈ {1,2,4,8,16,32} cycles/image
     (both horizontal & vertical orientations).
   • Forward-pass through each model; extract per-layer feature maps.
   • Compute 2-D FFT of features → measure power at the input frequency
     relative to total power (= frequency-retention ratio).

2. Frequency-Response Heatmap
   • Feed a white-noise image; compute 2-D FFT of the LAST layer output.
   • Average over channels → 2-D spectral response map (64×64 in freq domain).

3. High vs Low Frequency Alignment
   • For 10 random test images, create:
       low-pass  = Gaussian blur (σ=8)
       high-pass = original − low-pass
   • Measure cosine similarity of final features between:
       original ↔ low-pass  →  low_freq_alignment
       original ↔ high-pass →  high_freq_alignment

Outputs
-------
  data/frequency_analysis.json
"""

import os
import sys
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import get_all_models


# ---------------------------------------------------------------------------
# Helpers — image generation
# ---------------------------------------------------------------------------

def sinusoidal_grating(freq: float, orientation: str = 'horizontal',
                        img_size: int = 64, amplitude: float = 0.5) -> torch.Tensor:
    """
    Generate a pure sinusoidal grating.

    Args
    ----
    freq        : cycles per image (integer or float)
    orientation : 'horizontal' | 'vertical'
    img_size    : spatial size (square)
    amplitude   : contrast amplitude

    Returns
    -------
    x : (1, 3, img_size, img_size) float tensor
    """
    coords = torch.linspace(0, 2 * math.pi * freq, img_size)
    if orientation == 'horizontal':
        wave = amplitude * torch.sin(coords).view(1, img_size)   # (1, W)
        grating = wave.expand(img_size, -1)                      # (H, W)
    else:
        wave    = amplitude * torch.sin(coords).view(img_size, 1)
        grating = wave.expand(-1, img_size)

    x = grating.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)  # (1,3,H,W)
    # Centre the grating (zero mean, range ~[-0.5, 0.5] → shift to [0.5,1.5])
    x = x + 1.0
    return x


def gaussian_blur_2d(x: torch.Tensor, sigma: float = 8.0) -> torch.Tensor:
    """Apply Gaussian blur with std=sigma to a (B,C,H,W) tensor (CPU)."""
    # Kernel half-size
    ks = int(4 * sigma) | 1           # make odd
    ks = max(ks, 3)
    # 1-D Gaussian kernel
    t    = torch.arange(ks, dtype=torch.float32) - ks // 2
    kern = torch.exp(-0.5 * (t / sigma) ** 2)
    kern = kern / kern.sum()
    # Separable 2-D via two 1-D convolutions
    B, C, H, W = x.shape
    k1d = kern.view(1, 1, 1, ks).expand(C, 1, 1, ks)
    k1d_v = kern.view(1, 1, ks, 1).expand(C, 1, ks, 1)
    pad = ks // 2
    out = F.conv2d(x, k1d,   padding=(0, pad), groups=C)
    out = F.conv2d(out, k1d_v, padding=(pad, 0), groups=C)
    return out


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def extract_feature_maps(model, model_name: str,
                          x: torch.Tensor, img_size: int = 64):
    """
    Run forward pass and return per-layer feature maps, all resized to
    (C, img_size, img_size) for uniform FFT analysis.
    """
    model.eval()
    with torch.no_grad():
        _, features = model(x, return_features=True)

    out_feats = []
    for feat in features:
        if feat.dim() == 3:
            # (B, L, D)  — ViT / Mamba token sequence; reshape to spatial
            B, L, D = feat.shape
            g = int(math.sqrt(L - 1)) if model_name == 'vit' else int(math.sqrt(L))
            if model_name == 'vit':
                # drop CLS token
                spatial = feat[:, 1:, :].transpose(1, 2).reshape(B, D, g, g)
            else:
                spatial = feat.transpose(1, 2).reshape(B, D, g, g)
        else:
            spatial = feat   # (B, C, H, W)

        # Resize to img_size × img_size for uniform comparison
        sp_resized = F.interpolate(spatial.float(),
                                   size=(img_size, img_size),
                                   mode='bilinear', align_corners=False)
        out_feats.append(sp_resized[0])   # (C, H, W)
    return out_feats


# ---------------------------------------------------------------------------
# Step 1 — Frequency sensitivity
# ---------------------------------------------------------------------------

FREQUENCIES = [1, 2, 4, 8, 16, 32]
ORIENTATIONS = ['horizontal', 'vertical']


def _power_at_freq(fft_mag: np.ndarray, freq: int, img_size: int) -> float:
    """Fraction of FFT power at the target spatial frequency bin."""
    H, W = fft_mag.shape
    cy, cx = H // 2, W // 2
    total = fft_mag.sum() + 1e-12
    # Expected frequency bin in shifted FFT
    # freq cycles / img_size pixels → bin index offset from centre
    bin_y = int(round(freq * H / img_size))
    bin_x = int(round(freq * W / img_size))
    # Sum a small neighbourhood (±1 bin) to be robust to rounding
    r = 1
    region = fft_mag[max(0, cy - bin_y - r):cy - bin_y + r + 1,
                     max(0, cx - bin_x - r):cx - bin_x + r + 1]
    power = region.sum()
    # Also check the symmetric side
    region2 = fft_mag[cy + bin_y - r:cy + bin_y + r + 1,
                      cx + bin_x - r:cx + bin_x + r + 1]
    power += region2.sum()
    return float(power / total)


def frequency_sensitivity(model, model_name: str, img_size: int = 64) -> dict:
    """
    For each frequency and orientation, compute the per-layer
    frequency-retention ratio and spectral centroid.
    """
    layer_results = []   # one entry per (freq, orient)

    for freq in FREQUENCIES:
        for orient in ORIENTATIONS:
            grating = sinusoidal_grating(freq, orient, img_size)
            feats   = extract_feature_maps(model, model_name, grating, img_size)

            per_layer = []
            for i, fm in enumerate(feats):
                # fm: (C, H, W)
                fft2 = torch.fft.fft2(fm.float())
                fft2 = torch.fft.fftshift(fft2, dim=(-2, -1))
                mag  = fft2.abs().mean(dim=0).numpy()   # (H, W)

                retention = _power_at_freq(mag, freq, img_size)

                # Spectral centroid: mean frequency weighted by power
                H2, W2 = mag.shape
                fy = (np.arange(H2) - H2 // 2).reshape(-1, 1)
                fx = (np.arange(W2) - W2 // 2).reshape(1, -1)
                fr = np.sqrt(fx ** 2 + fy ** 2)
                centroid = float((fr * mag).sum() / (mag.sum() + 1e-12))

                per_layer.append({
                    'layer': i,
                    'freq_retention': round(retention, 6),
                    'spectral_centroid': round(centroid, 4),
                })

            layer_results.append({
                'frequency':   freq,
                'orientation': orient,
                'layers':      per_layer,
            })

    return layer_results


# ---------------------------------------------------------------------------
# Step 2 — Frequency-response heatmap (white-noise input)
# ---------------------------------------------------------------------------

def frequency_response_heatmap(model, model_name: str,
                                img_size: int = 64, n_trials: int = 5) -> list:
    """
    Average 2-D spectral power map of the last-layer features
    when the input is white noise (averaged over n_trials for stability).

    Returns a (img_size × img_size) list of floats (shifted FFT power).
    """
    accumulated = None
    for _ in range(n_trials):
        noise = torch.randn(1, 3, img_size, img_size)
        feats = extract_feature_maps(model, model_name, noise, img_size)
        last  = feats[-1].float()               # (C, H, W)

        fft2  = torch.fft.fft2(last)
        fft2  = torch.fft.fftshift(fft2, dim=(-2, -1))
        mag   = fft2.abs().mean(dim=0).numpy()  # (H, W)

        accumulated = mag if accumulated is None else accumulated + mag

    hmap = accumulated / n_trials
    # Normalise to [0,1]
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-12)
    return hmap.tolist()


# ---------------------------------------------------------------------------
# Step 3 — High vs low frequency alignment
# ---------------------------------------------------------------------------

def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return float(F.cosine_similarity(a_flat.unsqueeze(0),
                                     b_flat.unsqueeze(0)).item())


def highlow_frequency_alignment(model, model_name: str,
                                 img_size: int = 64, n_images: int = 10,
                                 blur_sigma: float = 8.0) -> dict:
    """
    Measure how much each model's features align with low vs high frequencies.

    Returns dict with keys: low_freq_alignment, high_freq_alignment,
    frequency_bias ('low'|'high'|'balanced'), per_image_scores.
    """
    torch.manual_seed(42)
    low_scores  = []
    high_scores = []

    for _ in range(n_images):
        orig    = torch.randn(1, 3, img_size, img_size)
        low_p   = gaussian_blur_2d(orig, sigma=blur_sigma)
        high_p  = orig - low_p

        f_orig  = extract_feature_maps(model, model_name, orig,   img_size)[-1]
        f_low   = extract_feature_maps(model, model_name, low_p,  img_size)[-1]
        f_high  = extract_feature_maps(model, model_name, high_p, img_size)[-1]

        low_scores.append(_cos_sim(f_orig, f_low))
        high_scores.append(_cos_sim(f_orig, f_high))

    mean_low  = float(np.mean(low_scores))
    mean_high = float(np.mean(high_scores))

    diff = mean_low - mean_high
    if abs(diff) < 0.05:
        bias = 'balanced'
    elif diff > 0:
        bias = 'low'
    else:
        bias = 'high'

    return {
        'low_freq_alignment':  round(mean_low,  6),
        'high_freq_alignment': round(mean_high, 6),
        'frequency_bias':      bias,
        'per_image_low':       [round(s, 6) for s in low_scores],
        'per_image_high':      [round(s, 6) for s in high_scores],
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_frequency_analysis(img_size: int = 64, out_path: str = None) -> dict:
    """Run all frequency analyses and write data/frequency_analysis.json."""
    if out_path is None:
        here     = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, '..', 'data', 'frequency_analysis.json')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    models  = get_all_models(num_classes=10)
    results = {}

    for name, model in models.items():
        print(f"  [{name.upper()}] frequency sensitivity …", flush=True)
        sensitivity = frequency_sensitivity(model, name, img_size)

        print(f"  [{name.upper()}] frequency heatmap …", flush=True)
        heatmap = frequency_response_heatmap(model, name, img_size)

        print(f"  [{name.upper()}] high/low alignment …", flush=True)
        alignment = highlow_frequency_alignment(model, name, img_size)

        # Summarise retention per frequency (averaged over layers & orientations)
        freq_summary = {}
        for freq in FREQUENCIES:
            entries = [e for e in sensitivity if e['frequency'] == freq]
            all_ret = [layer['freq_retention']
                       for e in entries for layer in e['layers']]
            freq_summary[str(freq)] = round(float(np.mean(all_ret)), 6)

        results[name] = {
            'frequency_sensitivity':     sensitivity,
            'frequency_heatmap_64x64':   heatmap,
            'high_low_alignment':        alignment,
            'freq_retention_by_freq':    freq_summary,
        }

        print(f"         low_align={alignment['low_freq_alignment']:.4f}  "
              f"high_align={alignment['high_freq_alignment']:.4f}  "
              f"bias={alignment['frequency_bias']}")

    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"  Saved → {out_path}")
    return results


if __name__ == '__main__':
    run_frequency_analysis()
