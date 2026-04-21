"""
run_all.py — Master script that runs all experiments sequentially.

Usage
-----
  python experiments/run_all.py

Generates in data/
  receptive_fields.json
  frequency_analysis.json
  model_stats.json
  training_curves.json
  summary.json

Runtime estimate: ~4–8 minutes on a modern CPU laptop.
"""

import os
import sys
import json
import time

# Ensure experiments/ is in path regardless of cwd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _done(label: str, elapsed: float):
    print(f"  ✓  {label} completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    results     = {}

    # ── 1. Receptive Field Analysis ──────────────────────────────────────
    _section("1 / 4  Receptive Field Analysis")
    t0 = time.time()
    from receptive_field import run_receptive_field_analysis
    rf_data = run_receptive_field_analysis(
        img_size=64,
        out_path=os.path.join(DATA_DIR, 'receptive_fields.json'),
    )
    results['receptive_fields'] = rf_data
    _done("Receptive Field Analysis", time.time() - t0)

    # ── 2. Frequency Analysis ─────────────────────────────────────────────
    _section("2 / 4  Frequency Response Analysis")
    t0 = time.time()
    from frequency_analysis import run_frequency_analysis
    freq_data = run_frequency_analysis(
        img_size=64,
        out_path=os.path.join(DATA_DIR, 'frequency_analysis.json'),
    )
    results['frequency_analysis'] = freq_data
    _done("Frequency Analysis", time.time() - t0)

    # ── 3. Model Statistics + Training Curves ────────────────────────────
    _section("3 / 4  Model Statistics & Training Curves")
    t0 = time.time()
    from model_stats import run_model_stats
    stats_data, curves_data = run_model_stats(
        out_stats=os.path.join(DATA_DIR, 'model_stats.json'),
        out_curves=os.path.join(DATA_DIR, 'training_curves.json'),
    )
    results['model_stats']     = stats_data
    results['training_curves'] = curves_data
    _done("Model Statistics & Training", time.time() - t0)

    # ── 4. Summary JSON ───────────────────────────────────────────────────
    _section("4 / 4  Building Summary")
    t0 = time.time()
    summary = _build_summary(rf_data, freq_data, stats_data, curves_data)
    summary_path = os.path.join(DATA_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {summary_path}")
    _done("Summary", time.time() - t0)

    # ── Final report ──────────────────────────────────────────────────────
    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE  ({total/60:.1f} min)")
    print(f"  Output files in:  {DATA_DIR}")
    print(f"{'='*60}\n")

    _print_summary_table(summary)


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(rf: dict, freq: dict, stats: dict, curves: dict) -> dict:
    models = ['cnn', 'vit', 'mamba']
    summary = {'models': {}}

    for m in models:
        # Receptive field
        rf_m = rf.get(m, {})
        # Frequency
        fr_m   = freq.get(m, {})
        align  = fr_m.get('high_low_alignment', {})
        # Stats
        st_m   = stats.get(m, {})
        # Curves
        cv_m   = curves.get(m, {})
        loss   = cv_m.get('train_loss', [])
        gn     = cv_m.get('gradient_norm', [])

        summary['models'][m] = {
            'theoretical_rf':      rf_m.get('theoretical_rf'),
            'effective_rf_radius': rf_m.get('effective_rf_radius'),
            'rf_gaussian_sigma':   rf_m.get('rf_gaussian_sigma'),
            'total_parameters':    st_m.get('total_parameters'),
            'flops_per_forward':   st_m.get('flops_per_forward'),
            'inference_time_ms':   st_m.get('inference_time_ms'),
            'memory_mb':           st_m.get('memory_mb'),
            'low_freq_alignment':  align.get('low_freq_alignment'),
            'high_freq_alignment': align.get('high_freq_alignment'),
            'frequency_bias':      align.get('frequency_bias'),
            'final_train_loss':    loss[-1] if loss else None,
            'min_train_loss':      min(loss) if loss else None,
            'mean_grad_norm':      round(sum(gn) / len(gn), 6) if gn else None,
        }

    return summary


def _print_summary_table(summary: dict):
    models = ['cnn', 'vit', 'mamba']
    keys   = [
        ('theoretical_rf',     'Theoretical RF'),
        ('effective_rf_radius','Effective RF radius'),
        ('total_parameters',   'Parameters'),
        ('inference_time_ms',  'Inference (ms)'),
        ('frequency_bias',     'Freq bias'),
        ('final_train_loss',   'Final loss'),
    ]
    col_w = 16
    header = f"{'Metric':<28}" + "".join(f"{m.upper():>{col_w}}" for m in models)
    print(header)
    print("-" * (28 + col_w * len(models)))
    for key, label in keys:
        row = f"{label:<28}"
        for m in models:
            val = summary['models'].get(m, {}).get(key)
            if val is None:
                s = "N/A"
            elif isinstance(val, float):
                s = f"{val:.4f}"
            elif isinstance(val, int):
                s = f"{val:,}"
            else:
                s = str(val)
            row += f"{s:>{col_w}}"
        print(row)
    print()


if __name__ == '__main__':
    main()
