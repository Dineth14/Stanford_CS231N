/**
 * build.js — Static site builder for GitHub Pages deployment.
 *
 * Steps
 * -----
 * 1. Read all JSON from ../data/
 * 2. Inject as window.__RESEARCH_DATA__ = {...} into each HTML page
 * 3. Copy all assets to dist/
 * 4. No server required — works as a fully static site.
 *
 * Usage
 * -----
 *   node website/build.js
 */

'use strict';

const path = require('path');
const fs   = require('fs');

const ROOT       = path.join(__dirname, '..');
const PUBLIC_DIR = path.join(__dirname, 'public');
const DATA_DIR   = path.join(ROOT, 'data');
const DIST_DIR   = path.join(__dirname, 'dist');

// ---------------------------------------------------------------------------
// Load all JSON data
// ---------------------------------------------------------------------------

const DATA_FILES = [
  'receptive_fields',
  'frequency_analysis',
  'model_stats',
  'training_curves',
  'summary',
];

function loadAllData() {
  const merged = {};
  for (const name of DATA_FILES) {
    const p = path.join(DATA_DIR, `${name}.json`);
    if (fs.existsSync(p)) {
      try {
        merged[name] = JSON.parse(fs.readFileSync(p, 'utf8'));
        console.log(`  ✓  Loaded ${name}.json`);
      } catch (e) {
        console.warn(`  ⚠  Could not parse ${name}.json:`, e.message);
      }
    } else {
      console.warn(`  ⚠  Missing ${name}.json — using empty placeholder.`);
      merged[name] = null;
    }
  }
  return merged;
}

// ---------------------------------------------------------------------------
// Inject data script into HTML
// ---------------------------------------------------------------------------

function injectData(html, data) {
  const script = `<script>
window.__RESEARCH_DATA__ = ${JSON.stringify(data)};
</script>`;
  // Insert just before </head>
  if (html.includes('</head>')) {
    return html.replace('</head>', `${script}\n</head>`);
  }
  return script + '\n' + html;
}

// ---------------------------------------------------------------------------
// Copy directory recursively
// ---------------------------------------------------------------------------

function copyDir(src, dest) {
  if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}

// ---------------------------------------------------------------------------
// Process HTML files
// ---------------------------------------------------------------------------

function processHtml(src, dest, data) {
  const html       = fs.readFileSync(src, 'utf8');
  const injected   = injectData(html, data);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, injected, 'utf8');
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

function build() {
  console.log('\n  Building static site → dist/\n');

  // Clean dist
  if (fs.existsSync(DIST_DIR)) {
    fs.rmSync(DIST_DIR, { recursive: true });
  }
  fs.mkdirSync(DIST_DIR, { recursive: true });

  // Load data
  const data = loadAllData();
  console.log();

  // Walk public/, inject data into HTML, copy everything else
  function walk(srcDir, destDir) {
    for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
      const s = path.join(srcDir, entry.name);
      const d = path.join(destDir, entry.name);
      if (entry.isDirectory()) {
        fs.mkdirSync(d, { recursive: true });
        walk(s, d);
      } else if (entry.name.endsWith('.html')) {
        processHtml(s, d, data);
        console.log(`  ✓  ${path.relative(PUBLIC_DIR, s)}`);
      } else {
        fs.mkdirSync(path.dirname(d), { recursive: true });
        fs.copyFileSync(s, d);
      }
    }
  }

  walk(PUBLIC_DIR, DIST_DIR);

  // Write a _nojekyll file so GitHub Pages doesn't ignore _ files
  fs.writeFileSync(path.join(DIST_DIR, '.nojekyll'), '');

  console.log('\n  ✓  Build complete →', DIST_DIR, '\n');
}

build();
