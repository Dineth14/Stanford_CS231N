# Receptive Field & Frequency Analysis

> Comparing CNN, Vision Transformer, and Mamba across receptive fields, frequency response, parameter efficiency, and training dynamics — with a full interactive research website.

---

## Architecture Overview

```
Receptive-Field-Analysis/
├── experiments/
│   ├── __init__.py
│   ├── models.py            # SmallCNN, SmallViT, SmallMamba model definitions
│   ├── receptive_field.py   # ERF computation + theoretical RF analysis
│   ├── frequency_analysis.py# Sinusoidal grating / FFT / high-low alignment
│   ├── model_stats.py       # Parameters, FLOPs, inference time, training curves
│   └── run_all.py           # Master orchestration script
├── data/                    # Generated JSON outputs (git-ignored)
│   ├── receptive_fields.json
│   ├── frequency_analysis.json
│   ├── model_stats.json
│   ├── training_curves.json
│   └── summary.json
├── website/
│   ├── package.json
│   ├── server.js            # Express dev server (port 3000)
│   ├── build.js             # Static site builder → dist/
│   └── public/
│       ├── index.html       # Homepage + summary table
│       ├── css/main.css     # Dark scientific design system
│       ├── js/
│       │   ├── main.js      # Navigation, data loading, scroll spy
│       │   └── charts.js    # All Chart.js / Plotly visualisations
│       └── pages/
│           ├── 01-introduction.html
│           ├── 02-architectures.html
│           ├── 03-receptive-fields.html
│           ├── 04-frequency-analysis.html
│           ├── 05-model-parameters.html
│           ├── 06-training-dynamics.html
│           ├── 07-remote-sensing.html
│           ├── 08-mathematical-theory.html
│           ├── 09-failure-modes.html
│           ├── 10-comparison.html
│           └── 11-conclusion.html
├── .github/workflows/deploy.yml  # GitHub Actions → GitHub Pages
├── requirements.txt
└── README.md
```

---

## Models

All three models are **4-layer, CPU-only** designs that complete all experiments in under 10 minutes on a laptop.

| Model | Description | Theoretical RF |
|---|---|---|
| `SmallCNN` | 4 × Conv(3,pad=1) + BN + ReLU, GlobalAvgPool → Linear | 9 px |
| `SmallViT` | Patch 8, embed 128, 4 × (MHSA + MLP), CLS token | Full image (global) |
| `SmallMamba` | Patch 4, d_model 64, state 64, 4 × MambaBlock | Causal full length |

---

## Setup

### Python

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Node.js

```bash
cd website
npm install
```

---

## Running Experiments

```bash
# From the project root
python experiments/run_all.py
```

This sequentially runs:
1. **Receptive field analysis** → `data/receptive_fields.json`
2. **Frequency analysis** → `data/frequency_analysis.json`
3. **Model statistics** → `data/model_stats.json`, `data/training_curves.json`
4. **Summary** → `data/summary.json`

Total runtime: ~5–10 minutes on a modern CPU.

---

## Running the Website (Development)

```bash
cd website
npm start          # or: node server.js
```

Open [http://localhost:3000](http://localhost:3000).

The server reads JSON files from `../data/` and serves them via:
- `GET /api/data` — all data merged
- `GET /api/data/:file` — single file (e.g. `/api/data/receptive_fields`)
- `GET /api/health` — health check

For hot-reloading during development:
```bash
npm run dev   # uses nodemon
```

---

## Building the Static Site

```bash
cd website
node build.js
```

This reads all JSON files, injects them as `window.__RESEARCH_DATA__` into every HTML page,
and copies the result to `website/dist/`. The dist folder is self-contained and can be
deployed to any static host.

---

## Deploying to GitHub Pages

Push to `main`. The [GitHub Actions workflow](.github/workflows/deploy.yml) will:
1. Install Python + run all experiments
2. Install Node + run `build.js`
3. Deploy `website/dist/` to GitHub Pages via the official Pages action

Enable GitHub Pages in **Settings → Pages → Source: GitHub Actions**.

---

## Output JSON Files

| File | Description |
|---|---|
| `data/receptive_fields.json` | Theoretical RF, ERF radius, Gaussian sigma, 64×64 ERF heatmap per model |
| `data/frequency_analysis.json` | Per-layer frequency retention, spectral centroid, 64×64 frequency response heatmaps, high/low alignment scores |
| `data/model_stats.json` | Total/trainable params, FLOPs, inference time (ms), memory (MB) per model |
| `data/training_curves.json` | Per-epoch train loss, gradient norm, weight update magnitude for 20-epoch toy training |
| `data/summary.json` | Key scalar metrics merged for quick reference |

---

## Mathematical Notation

| Symbol | Meaning |
|---|---|
| $\text{RF}_L$ | Theoretical receptive field after $L$ layers |
| $k$ | Convolutional kernel size |
| $s$ | Stride |
| $\bar{A}, \bar{B}$ | ZOH-discretised SSM matrices |
| $\Delta$ | SSM input-dependent timescale (selectivity) |
| $H(j\omega)$ | SSM frequency response (transfer function on imaginary axis) |
| $\kappa(q,k)$ | Softmax attention kernel |
| $\sigma_\text{ERF}$ | Gaussian sigma fitted to effective receptive field |

---

## Citations

```bibtex
@article{lecun1989backpropagation,
  title={Backpropagation applied to handwritten zip code recognition},
  author={LeCun, Yann and others}, journal={Neural computation}, year={1989}
}
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others}, booktitle={NeurIPS}, year={2017}
}
@inproceedings{dosovitskiy2021image,
  title={An image is worth 16x16 words},
  author={Dosovitskiy, Alexey and others}, booktitle={ICLR}, year={2021}
}
@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri}, journal={arXiv:2312.00752}, year={2023}
}
@inproceedings{luo2016understanding,
  title={Understanding the effective receptive field in deep CNNs},
  author={Luo, Wenjie and others}, booktitle={NeurIPS}, year={2016}
}
@inproceedings{park2022how,
  title={How do vision transformers work?},
  author={Park, Namuk and Kim, Songkuk}, booktitle={ICLR}, year={2022}
}
@inproceedings{gu2020hippo,
  title={HiPPO: Recurrent memory with optimal polynomial projections},
  author={Gu, Albert and others}, booktitle={NeurIPS}, year={2020}
}
```
