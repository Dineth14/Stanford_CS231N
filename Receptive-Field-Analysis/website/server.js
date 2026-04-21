/**
 * server.js — Express server for the RF & Frequency Analysis research website.
 *
 * Routes
 * ------
 *   GET /              → serves public/index.html
 *   GET /pages/:page   → serves public/pages/<page>.html
 *   GET /api/data      → returns all JSON data as one merged object
 *   GET /api/data/:file → returns a specific JSON file
 *   Static             → serves everything in public/
 */

'use strict';

const path        = require('path');
const fs          = require('fs');
const express     = require('express');
const compression = require('compression');

const app = express();
const PORT = process.env.PORT || 3000;

// Directories
const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR   = path.join(__dirname, '..', 'data');

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(compression());
app.use(express.static(PUBLIC_DIR));

// ── API: all data ──────────────────────────────────────────────────────────
const DATA_FILES = [
  'receptive_fields',
  'frequency_analysis',
  'model_stats',
  'training_curves',
  'summary',
];

function loadJson(name) {
  const p = path.join(DATA_DIR, `${name}.json`);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch (e) {
    console.error(`Failed to parse ${p}:`, e.message);
    return null;
  }
}

app.get('/api/data', (req, res) => {
  const merged = {};
  for (const name of DATA_FILES) {
    const data = loadJson(name);
    if (data) merged[name] = data;
  }
  res.json(merged);
});

app.get('/api/data/:file', (req, res) => {
  const name = path.basename(req.params.file, '.json');
  if (!DATA_FILES.includes(name)) {
    return res.status(404).json({ error: 'Unknown data file' });
  }
  const data = loadJson(name);
  if (!data) {
    return res.status(404).json({
      error: 'Data not found. Run: python experiments/run_all.py first.',
    });
  }
  res.json(data);
});

// ── Health check ───────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  const available = DATA_FILES.filter(n => {
    return fs.existsSync(path.join(DATA_DIR, `${n}.json`));
  });
  res.json({
    status:          'ok',
    data_available:  available,
    data_missing:    DATA_FILES.filter(n => !available.includes(n)),
  });
});

// ── SPA fallback (serve index.html for unknown routes) ────────────────────
app.get('*', (req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  RF & Frequency Analysis  —  http://localhost:${PORT}\n`);
  const missing = DATA_FILES.filter(
    n => !fs.existsSync(path.join(DATA_DIR, `${n}.json`)),
  );
  if (missing.length > 0) {
    console.warn(`  ⚠  Missing data files: ${missing.join(', ')}`);
    console.warn(`     Run:  python experiments/run_all.py\n`);
  } else {
    console.log(`  ✓  All data files present.\n`);
  }
});
