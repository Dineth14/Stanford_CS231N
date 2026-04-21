/**
 * charts.js — All Chart.js / D3 / Plotly visualisations.
 *
 * Each page calls window.initPageCharts(data) once data is loaded.
 * Charts degrade gracefully when data is missing (show placeholder text).
 */
'use strict';

/* ── Colour palette (matches CSS vars) ────────────────────────────────── */
const C = {
  bg:     '#0a0e1a',
  surface:'#111827',
  border: '#2d3a4f',
  white:  '#f0f4f8',
  muted:  '#8892a4',
  cyan:   '#00d4ff',
  violet: '#7c3aed',
  green:  '#10b981',
  amber:  '#f59e0b',
  pink:   '#ec4899',
};

const MODEL_COLORS = { cnn: C.green, vit: C.amber, mamba: C.pink };
const MODEL_LABELS = { cnn: 'CNN', vit: 'ViT', mamba: 'Mamba' };

/* ── Chart.js global defaults ─────────────────────────────────────────── */
if (typeof Chart !== 'undefined') {
  Chart.defaults.color          = C.muted;
  Chart.defaults.borderColor    = C.border;
  Chart.defaults.backgroundColor = C.surface;
  Chart.defaults.font.family    = "'IBM Plex Mono', monospace";
  Chart.defaults.font.size      = 12;
  Chart.defaults.plugins.legend.labels.color = C.white;
  Chart.defaults.plugins.tooltip.backgroundColor = C.surface;
  Chart.defaults.plugins.tooltip.borderColor     = C.border;
  Chart.defaults.plugins.tooltip.borderWidth     = 1;
}

/* ── Placeholder helper ───────────────────────────────────────────────── */
function showPlaceholder(el, msg = 'Run experiments to generate data.') {
  if (!el) return;
  el.innerHTML = `<div class="loading-placeholder">${msg}</div>`;
}

/* ── Helpers ──────────────────────────────────────────────────────────── */
function getCtx(id) {
  const el = document.getElementById(id);
  return el ? el.getContext('2d') : null;
}

function chartOpts(title, xLabel, yLabel, extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      title: {
        display: !!title,
        text:    title,
        color:   C.white,
        font:    { family: "'Space Grotesk', sans-serif", size: 14, weight: '600' },
        padding: { bottom: 12 },
      },
      legend: { position: 'top' },
    },
    scales: {
      x: {
        title: { display: !!xLabel, text: xLabel, color: C.muted },
        grid:  { color: C.border + '40' },
        ticks: { color: C.muted },
      },
      y: {
        title: { display: !!yLabel, text: yLabel, color: C.muted },
        grid:  { color: C.border + '40' },
        ticks: { color: C.muted },
      },
    },
    ...extra,
  };
}

/* =========================================================================
   PAGE 3 — Receptive Fields
   ========================================================================= */

function drawERFHeatmaps(rfData) {
  ['cnn', 'vit', 'mamba'].forEach(name => {
    const canvas = document.getElementById(`erf-canvas-${name}`);
    if (!canvas) return;

    const erf = rfData?.[name]?.erf_map;
    if (!erf) { showPlaceholder(canvas.parentElement); return; }

    const H = erf.length, W = erf[0].length;
    canvas.width  = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);

    const col = MODEL_COLORS[name];
    const r = parseInt(col.slice(1,3),16);
    const g = parseInt(col.slice(3,5),16);
    const b = parseInt(col.slice(5,7),16);

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v   = Math.min(1, Math.max(0, erf[y][x]));
        const idx = (y * W + x) * 4;
        img.data[idx]   = Math.round(r * v);
        img.data[idx+1] = Math.round(g * v);
        img.data[idx+2] = Math.round(b * v);
        img.data[idx+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  });
}

function drawRFComparisonTable(rfData) {
  const tbody = document.getElementById('rf-table-body');
  if (!tbody || !rfData) return;
  tbody.innerHTML = '';

  ['cnn', 'vit', 'mamba'].forEach(name => {
    const d = rfData[name] || {};
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="row-label"><span class="badge badge-${name}">${MODEL_LABELS[name]}</span></td>
      <td class="mono">${d.theoretical_rf ?? 'N/A'}</td>
      <td class="mono">${d.effective_rf_radius?.toFixed(2) ?? 'N/A'}</td>
      <td class="mono">${d.rf_gaussian_sigma?.toFixed(2) ?? 'N/A'}</td>
      <td class="mono">${d.effective_rf_radius
        ? ((d.effective_rf_radius / 32 * 100).toFixed(1) + '%')
        : 'N/A'}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* =========================================================================
   PAGE 4 — Frequency Analysis
   ========================================================================= */

function drawFrequencyRetentionChart(freqData) {
  const ctx = getCtx('freq-retention-chart');
  if (!ctx) return;
  if (!freqData) { showPlaceholder(ctx.canvas.parentElement); return; }

  const freqs = [1, 2, 4, 8, 16, 32];
  const datasets = ['cnn', 'vit', 'mamba'].map(name => {
    const ret = freqData[name]?.freq_retention_by_freq || {};
    return {
      label:       MODEL_LABELS[name],
      data:        freqs.map(f => (ret[String(f)] ?? 0) * 100),
      borderColor: MODEL_COLORS[name],
      backgroundColor: MODEL_COLORS[name] + '20',
      tension:     0.35,
      pointRadius: 5,
      fill:        true,
    };
  });

  new Chart(ctx, {
    type: 'line',
    data: { labels: freqs.map(f => `${f} c/img`), datasets },
    options: chartOpts(
      'Frequency Retention Ratio by Spatial Frequency',
      'Spatial Frequency (cycles/image)',
      'Retention (%)',
    ),
  });
}

function drawFreqHeatmapsPlotly(freqData) {
  ['cnn', 'vit', 'mamba'].forEach(name => {
    const divId = `freq-heatmap-${name}`;
    const div   = document.getElementById(divId);
    if (!div || typeof Plotly === 'undefined') return;

    const hmap = freqData?.[name]?.frequency_heatmap_64x64;
    if (!hmap) { showPlaceholder(div); return; }

    Plotly.newPlot(div, [{
      z:         hmap,
      type:      'heatmap',
      colorscale:'Viridis',
      showscale: false,
    }], {
      title:  { text: `${MODEL_LABELS[name]} Spectral Response`, font: { color: C.white, size: 13 } },
      paper_bgcolor: C.surface,
      plot_bgcolor:  C.surface,
      margin: { l: 30, r: 10, t: 40, b: 30 },
      xaxis:  { title: 'Freq X', color: C.muted, showgrid: false },
      yaxis:  { title: 'Freq Y', color: C.muted, showgrid: false, autorange: 'reversed' },
    }, { displayModeBar: false, responsive: true });
  });
}

function drawHighLowAlignmentChart(freqData) {
  const ctx = getCtx('high-low-chart');
  if (!ctx || !freqData) return;

  const models = ['cnn', 'vit', 'mamba'];
  const low  = models.map(m => freqData[m]?.high_low_alignment?.low_freq_alignment  ?? 0);
  const high = models.map(m => freqData[m]?.high_low_alignment?.high_freq_alignment ?? 0);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models.map(m => MODEL_LABELS[m]),
      datasets: [
        { label: 'Low-Freq Alignment',  data: low,  backgroundColor: C.cyan   + 'bb' },
        { label: 'High-Freq Alignment', data: high, backgroundColor: C.violet + 'bb' },
      ],
    },
    options: chartOpts(
      'High vs Low Frequency Feature Alignment',
      'Model', 'Cosine Similarity',
      { scales: { y: { min: -1, max: 1 } } },
    ),
  });
}

/* =========================================================================
   PAGE 5 — Model Parameters
   ========================================================================= */

function drawParamComparisonChart(statsData) {
  const ctx = getCtx('params-chart');
  if (!ctx || !statsData) return;

  const models = ['cnn', 'vit', 'mamba'];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models.map(m => MODEL_LABELS[m]),
      datasets: [{
        label: 'Total Parameters',
        data:  models.map(m => statsData[m]?.total_parameters ?? 0),
        backgroundColor: models.map(m => MODEL_COLORS[m] + 'cc'),
        borderColor:     models.map(m => MODEL_COLORS[m]),
        borderWidth: 2,
      }],
    },
    options: chartOpts('Total Trainable Parameters', 'Model', 'Parameters'),
  });
}

function drawFlopsComparisonChart(statsData) {
  const ctx = getCtx('flops-chart');
  if (!ctx || !statsData) return;

  const models = ['cnn', 'vit', 'mamba'];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models.map(m => MODEL_LABELS[m]),
      datasets: [{
        label: 'FLOPs per Forward Pass',
        data:  models.map(m => (statsData[m]?.flops_per_forward ?? 0) / 1e6),
        backgroundColor: models.map(m => MODEL_COLORS[m] + 'cc'),
        borderColor:     models.map(m => MODEL_COLORS[m]),
        borderWidth: 2,
      }],
    },
    options: chartOpts('FLOPs per Forward Pass', 'Model', 'MFLOPs (millions)'),
  });
}

function drawInferenceTimeChart(statsData) {
  const ctx = getCtx('inference-chart');
  if (!ctx || !statsData) return;

  const models = ['cnn', 'vit', 'mamba'];
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models.map(m => MODEL_LABELS[m]),
      datasets: [{
        label: 'Inference Time (ms)',
        data:  models.map(m => statsData[m]?.inference_time_ms ?? 0),
        backgroundColor: models.map(m => MODEL_COLORS[m] + 'cc'),
        borderColor:     models.map(m => MODEL_COLORS[m]),
        borderWidth: 2,
      }],
    },
    options: chartOpts('CPU Inference Time (median, 50 runs)', 'Model', 'Milliseconds'),
  });
}

function drawLayerParamsTable(statsData) {
  ['cnn', 'vit', 'mamba'].forEach(name => {
    const tbody = document.getElementById(`layer-params-${name}`);
    if (!tbody || !statsData?.[name]) return;

    const layers = statsData[name].layer_wise_params || [];
    tbody.innerHTML = '';
    layers.slice(0, 20).forEach(({ layer_name, params }) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono small">${layer_name}</td>
        <td class="mono small text-center">${params.toLocaleString()}</td>
      `;
      tbody.appendChild(tr);
    });
  });
}

/* =========================================================================
   PAGE 6 — Training Dynamics
   ========================================================================= */

function drawTrainingLossChart(curvesData) {
  const ctx = getCtx('loss-chart');
  if (!ctx || !curvesData) return;

  const epochs = Array.from({ length: 20 }, (_, i) => i + 1);
  const datasets = ['cnn', 'vit', 'mamba'].map(name => ({
    label:       MODEL_LABELS[name],
    data:        curvesData[name]?.train_loss || [],
    borderColor: MODEL_COLORS[name],
    backgroundColor: MODEL_COLORS[name] + '10',
    tension:     0.3,
    fill:        false,
    pointRadius: 3,
  }));

  new Chart(ctx, {
    type: 'line',
    data: { labels: epochs, datasets },
    options: chartOpts('Training Loss Curves', 'Epoch', 'Cross-Entropy Loss'),
  });
}

function drawGradientNormChart(curvesData) {
  const ctx = getCtx('grad-norm-chart');
  if (!ctx || !curvesData) return;

  const epochs = Array.from({ length: 20 }, (_, i) => i + 1);
  const datasets = ['cnn', 'vit', 'mamba'].map(name => ({
    label:       MODEL_LABELS[name],
    data:        curvesData[name]?.gradient_norm || [],
    borderColor: MODEL_COLORS[name],
    tension:     0.3,
    fill:        false,
    pointRadius: 3,
  }));

  new Chart(ctx, {
    type: 'line',
    data: { labels: epochs, datasets },
    options: chartOpts('Gradient L2 Norm per Epoch', 'Epoch', '‖∇‖₂'),
  });
}

function drawWeightUpdateChart(curvesData) {
  const ctx = getCtx('weight-update-chart');
  if (!ctx || !curvesData) return;

  const epochs = Array.from({ length: 20 }, (_, i) => i + 1);
  const datasets = ['cnn', 'vit', 'mamba'].map(name => ({
    label:       MODEL_LABELS[name],
    data:        curvesData[name]?.weight_update_magnitude || [],
    borderColor: MODEL_COLORS[name],
    tension:     0.3,
    fill:        false,
    pointRadius: 3,
  }));

  new Chart(ctx, {
    type: 'line',
    data: { labels: epochs, datasets },
    options: chartOpts('Weight Update Magnitude per Epoch', 'Epoch', 'Mean ‖Δw‖₂'),
  });
}

/* =========================================================================
   PAGE 10 — Radar / Spider Chart
   ========================================================================= */

function drawSpiderChart(data) {
  const ctx = getCtx('spider-chart');
  if (!ctx) return;

  const summary = data?.summary?.models || {};
  const rf      = data?.receptive_fields || {};

  // Normalise a value to [0, 10] scale for radar
  function norm(val, min, max) {
    if (val == null) return 0;
    return Math.max(0, Math.min(10, ((val - min) / (max - min)) * 10));
  }

  // Labels and raw values
  const labels = [
    'Receptive\nField',
    'Data\nEfficiency',
    'Parameter\nEfficiency',
    'Inference\nSpeed',
    'Frequency\nFlexibility',
    'Scalability',
    'RS\nSuitability',
    'Training\nStability',
  ];

  // Heuristic scores per model on each dimension [0-10]
  const scores = {
    cnn: [
      norm(rf.cnn?.theoretical_rf, 0, 4096),         // RF (low for CNN)
      8,   // Data efficiency (inductive bias helps)
      7,   // Param efficiency
      9,   // Speed (fast conv)
      5,   // Freq flexibility (bandpass bias)
      5,   // Scalability (linear in params)
      6,   // RS suitability (good for local texture)
      8,   // Training stability
    ],
    vit: [
      10,  // RF (global from layer 1)
      3,   // Data efficiency (needs large pretraining)
      5,   // Param efficiency (many params)
      4,   // Speed (quadratic attention)
      8,   // Freq flexibility (learns any)
      4,   // Scalability (quadratic cost)
      7,   // RS suitability (global context)
      5,   // Training stability (needs warmup)
    ],
    mamba: [
      7,   // RF (causal linear growth)
      6,   // Data efficiency (no strong inductive bias)
      8,   // Param efficiency (linear complexity)
      7,   // Speed (linear scan)
      7,   // Freq flexibility (IIR filter)
      9,   // Scalability (linear in sequence length)
      8,   // RS suitability (long sequences)
      7,   // Training stability (stable eigenvalues)
    ],
  };

  const datasets = ['cnn', 'vit', 'mamba'].map(name => ({
    label:           MODEL_LABELS[name],
    data:            scores[name],
    borderColor:     MODEL_COLORS[name],
    backgroundColor: MODEL_COLORS[name] + '25',
    pointBackgroundColor: MODEL_COLORS[name],
    borderWidth: 2,
  }));

  new Chart(ctx, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0, max: 10,
          ticks:    { color: C.muted, stepSize: 2 },
          grid:     { color: C.border + '60' },
          angleLines:{ color: C.border + '60' },
          pointLabels: {
            color:     C.white,
            font:      { family: "'Space Grotesk', sans-serif", size: 11 },
          },
        },
      },
      plugins: {
        legend: { position: 'top' },
        title: {
          display: true,
          text: 'Model Capability Comparison (0–10)',
          color: C.white,
          font: { family: "'Space Grotesk', sans-serif", size: 14, weight: '600' },
        },
      },
    },
  });
}

/* =========================================================================
   Router — each page exports its chart init via window.initPageCharts
   ========================================================================= */

window.initPageCharts = function initPageCharts(data) {
  const page = window.location.pathname.split('/').pop() || 'index.html';

  // ── Index / Overview ────────────────────────────────────────────────────
  if (page === 'index.html' || page === '') {
    // summary table is populated by main.js → populateSummaryTable
  }

  // ── Page 03: Receptive Fields ───────────────────────────────────────────
  if (page.includes('03-receptive')) {
    drawERFHeatmaps(data.receptive_fields);
    drawRFComparisonTable(data.receptive_fields);
  }

  // ── Page 04: Frequency Analysis ─────────────────────────────────────────
  if (page.includes('04-frequency')) {
    drawFrequencyRetentionChart(data.frequency_analysis);
    drawFreqHeatmapsPlotly(data.frequency_analysis);
    drawHighLowAlignmentChart(data.frequency_analysis);
  }

  // ── Page 05: Model Parameters ───────────────────────────────────────────
  if (page.includes('05-model')) {
    drawParamComparisonChart(data.model_stats);
    drawFlopsComparisonChart(data.model_stats);
    drawInferenceTimeChart(data.model_stats);
    drawLayerParamsTable(data.model_stats);
  }

  // ── Page 06: Training Dynamics ──────────────────────────────────────────
  if (page.includes('06-training')) {
    drawTrainingLossChart(data.training_curves);
    drawGradientNormChart(data.training_curves);
    drawWeightUpdateChart(data.training_curves);
  }

  // ── Page 10: Comparison ─────────────────────────────────────────────────
  if (page.includes('10-comparison')) {
    drawSpiderChart(data);
  }
};
