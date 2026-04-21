"""
model_stats.py — Parameter counts, FLOPs, timing, memory, training curves.

For each model computes
-----------------------
  total_parameters       : int
  trainable_parameters   : int
  layer_wise_params      : [{layer_name, params}]
  flops_per_forward      : int  (via thop if available, else manual estimate)
  inference_time_ms      : float  (median over 100 CPU runs)
  memory_mb              : float  (peak RSS during forward pass)
  feature_map_sizes      : [[C, H, W] per layer]

Toy training experiment
-----------------------
  500 random images × 10 classes, Adam lr=1e-3, 20 epochs
  Records: train_loss, gradient_norm, weight_update_magnitude per epoch.

Outputs
-------
  data/model_stats.json
  data/training_curves.json
"""

import os
import sys
import json
import time
import tracemalloc
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from models import get_all_models, SmallCNN, SmallViT, SmallMamba


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layer_params = []
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            layer_params.append({'layer_name': name, 'params': params})

    return {
        'total_parameters':     total,
        'trainable_parameters': trainable,
        'layer_wise_params':    layer_params,
    }


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(model: nn.Module, input_size=(1, 3, 64, 64)) -> int:
    """
    Try thop first; fall back to a hand-crafted formula if unavailable.
    Returns FLOPs as integer.
    """
    try:
        from thop import profile
        x = torch.randn(*input_size)
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return int(flops)
    except ImportError:
        pass

    # Manual estimate
    x_dummy = torch.randn(*input_size)
    total_flops = 0

    def conv2d_flops(m, inp, out):
        nonlocal total_flops
        B, Cin, H, W = inp[0].shape
        Cout = out.shape[1]
        kH, kW = m.kernel_size
        total_flops += 2 * B * Cout * Cin * kH * kW * out.shape[2] * out.shape[3]

    def linear_flops(m, inp, out):
        nonlocal total_flops
        total_flops += 2 * inp[0].numel() * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_flops))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_flops))

    model.eval()
    with torch.no_grad():
        model(x_dummy)
    for h in hooks:
        h.remove()

    return total_flops


# ---------------------------------------------------------------------------
# Inference timing
# ---------------------------------------------------------------------------

def measure_inference_time(model: nn.Module, input_size=(1, 3, 64, 64),
                            n_runs: int = 50) -> float:
    """Median inference time in milliseconds (CPU, single image)."""
    model.eval()
    x = torch.randn(*input_size)
    times = []
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x)
    # Measure
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(statistics.median(times), 3)


# ---------------------------------------------------------------------------
# Peak memory (tracemalloc)
# ---------------------------------------------------------------------------

def measure_memory(model: nn.Module, input_size=(1, 3, 64, 64)) -> float:
    """Peak memory in MB during a single forward pass."""
    model.eval()
    x = torch.randn(*input_size)
    tracemalloc.start()
    with torch.no_grad():
        model(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(peak / 1e6, 3)


# ---------------------------------------------------------------------------
# Feature map shapes
# ---------------------------------------------------------------------------

def get_feature_map_sizes(model: nn.Module, model_name: str,
                           img_size: int = 64) -> list:
    """Return list of [C, H, W] for each layer's feature map."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        _, features = model(x, return_features=True)

    sizes = []
    for feat in features:
        if feat.dim() == 3:
            # token sequence (B, L, D) — treat as (D, sqrt(L), sqrt(L))
            import math
            B, L, D = feat.shape
            g = int(math.sqrt(L - 1)) if model_name == 'vit' else int(math.sqrt(L))
            sizes.append([D, g, g])
        else:
            sizes.append(list(feat.shape[1:]))   # [C, H, W]
    return sizes


# ---------------------------------------------------------------------------
# Attention maps (ViT only)
# ---------------------------------------------------------------------------

def get_attention_maps(model: nn.Module, model_name: str,
                       img_size: int = 64) -> list | None:
    if model_name != 'vit':
        return None
    model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        model(x)
    maps = []
    for attn in model.attention_maps:
        if attn is not None:
            # attn: (1, heads, N+1, N+1)
            maps.append(attn[0].cpu().numpy().tolist())
    return maps if maps else None


# ---------------------------------------------------------------------------
# Toy training experiment
# ---------------------------------------------------------------------------

def run_toy_training(model: nn.Module, model_name: str,
                     n_samples: int = 500, n_classes: int = 10,
                     epochs: int = 20, batch_size: int = 16,
                     lr: float = 1e-3, img_size: int = 32) -> dict:
    """
    Train on 500 random images with random labels for 20 epochs.
    Records loss, gradient norm, and weight-update magnitude per epoch.

    We use img_size=32 here for speed (this is just curve analysis,
    not accuracy benchmarking).
    """
    torch.manual_seed(42)

    # Generate dataset
    X = torch.randn(n_samples, 3, img_size, img_size)
    Y = torch.randint(0, n_classes, (n_samples,))

    # If model was built for img_size=64, we need to re-instantiate for 32
    if isinstance(model, SmallCNN):
        train_model = SmallCNN(num_classes=n_classes)
    elif isinstance(model, SmallViT):
        train_model = SmallViT(img_size=img_size, patch_size=4,
                               num_classes=n_classes, embed_dim=64,
                               num_heads=4)
    elif isinstance(model, SmallMamba):
        train_model = SmallMamba(img_size=img_size, patch_size=4,
                                 num_classes=n_classes, d_model=32)
    else:
        train_model = model

    train_model.train()
    optimizer = optim.Adam(train_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss':             [],
        'gradient_norm':          [],
        'weight_update_magnitude': [],
    }

    n_batches = math.ceil(n_samples / batch_size)

    # Snapshot weights before epoch for update magnitude
    prev_params = {n: p.data.clone()
                   for n, p in train_model.named_parameters() if p.requires_grad}

    for epoch in range(epochs):
        perm    = torch.randperm(n_samples)
        ep_loss = 0.0

        for b in range(n_batches):
            idx  = perm[b * batch_size: (b + 1) * batch_size]
            xb   = X[idx]
            yb   = Y[idx]

            optimizer.zero_grad()
            logits = train_model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        ep_loss /= n_batches

        # Gradient norm (last batch)
        grad_norm = 0.0
        for p in train_model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = math.sqrt(grad_norm)

        # Weight update magnitude
        upd_mag = 0.0
        n_param = 0
        for n, p in train_model.named_parameters():
            if p.requires_grad and n in prev_params:
                upd_mag += (p.data - prev_params[n]).norm(2).item()
                n_param += 1
        upd_mag /= max(n_param, 1)

        # Update prev
        prev_params = {n: p.data.clone()
                       for n, p in train_model.named_parameters() if p.requires_grad}

        history['train_loss'].append(round(ep_loss, 6))
        history['gradient_norm'].append(round(grad_norm, 6))
        history['weight_update_magnitude'].append(round(upd_mag, 6))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"         epoch {epoch+1:2d}/{epochs}  "
                  f"loss={ep_loss:.4f}  grad={grad_norm:.4f}  "
                  f"upd={upd_mag:.4f}")

    return history


import math  # needed for math.ceil inside run_toy_training


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model_stats(out_stats: str = None,
                    out_curves: str = None) -> tuple:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    if out_stats is None:
        out_stats  = os.path.join(data_dir, 'model_stats.json')
    if out_curves is None:
        out_curves = os.path.join(data_dir, 'training_curves.json')

    models  = get_all_models(num_classes=10)
    stats   = {}
    curves  = {}

    for name, model in models.items():
        print(f"  [{name.upper()}] parameters …", flush=True)
        param_info = count_parameters(model)

        print(f"  [{name.upper()}] FLOPs …", flush=True)
        flops = estimate_flops(model)

        print(f"  [{name.upper()}] inference time …", flush=True)
        inf_ms = measure_inference_time(model)

        print(f"  [{name.upper()}] memory …", flush=True)
        mem_mb = measure_memory(model)

        print(f"  [{name.upper()}] feature map sizes …", flush=True)
        fm_sizes = get_feature_map_sizes(model, name)

        print(f"  [{name.upper()}] attention maps …", flush=True)
        attn_maps = get_attention_maps(model, name)

        stats[name] = {
            **param_info,
            'flops_per_forward':  flops,
            'inference_time_ms':  inf_ms,
            'memory_mb':          mem_mb,
            'feature_map_sizes':  fm_sizes,
            'attention_maps':     attn_maps,
        }

        print(f"         params={param_info['total_parameters']:,}  "
              f"FLOPs={flops:,}  {inf_ms:.1f} ms  {mem_mb:.1f} MB")

        print(f"  [{name.upper()}] toy training …", flush=True)
        curve = run_toy_training(model, name)
        curves[name] = curve

    with open(out_stats, 'w') as f:
        json.dump(stats, f)
    with open(out_curves, 'w') as f:
        json.dump(curves, f)

    print(f"  Saved → {out_stats}")
    print(f"  Saved → {out_curves}")
    return stats, curves


if __name__ == '__main__':
    run_model_stats()
