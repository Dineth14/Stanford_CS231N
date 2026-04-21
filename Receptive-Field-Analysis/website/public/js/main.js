/**
 * main.js — Navigation, data loading, progress bar, sidebar toggle.
 */
'use strict';

// ── Global data store ──────────────────────────────────────────────────────
window.RFData = window.__RESEARCH_DATA__ || null;  // injected by build.js

// ── Data loader (fetches from Express API when RFData not injected) ────────
async function loadData() {
  if (window.RFData) return window.RFData;
  try {
    const res = await fetch('/api/data');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    window.RFData = await res.json();
    return window.RFData;
  } catch (e) {
    console.warn('Could not load data from API:', e.message);
    return {};
  }
}

// ── Reading progress bar ───────────────────────────────────────────────────
function initProgressBar() {
  const bar = document.getElementById('progress-bar');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const total    = document.documentElement.scrollHeight - window.innerHeight;
    const scrolled = window.scrollY;
    bar.style.width = total > 0 ? `${(scrolled / total) * 100}%` : '0%';
  }, { passive: true });
}

// ── Active nav link ────────────────────────────────────────────────────────
function setActiveNavLink() {
  const current = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href') || '';
    const page = href.split('/').pop();
    if (page === current || (current === 'index.html' && href === '/') ||
        (current === '' && href === '/')) {
      a.classList.add('active');
    } else {
      a.classList.remove('active');
    }
  });
}

// ── Mobile sidebar toggle ──────────────────────────────────────────────────
function initSidebar() {
  const sidebar  = document.querySelector('.sidebar');
  const overlay  = document.querySelector('.sidebar-overlay');
  const hamburger = document.querySelector('.hamburger');
  if (!sidebar || !hamburger) return;

  function open()  { sidebar.classList.add('open'); overlay?.classList.add('open'); }
  function close() { sidebar.classList.remove('open'); overlay?.classList.remove('open'); }

  hamburger.addEventListener('click', () =>
    sidebar.classList.contains('open') ? close() : open());
  overlay?.addEventListener('click', close);

  // Close on nav link click (mobile)
  sidebar.querySelectorAll('.nav-link').forEach(a => {
    a.addEventListener('click', close);
  });
}

// ── Intersection Observer — highlight nav section in view ─────────────────
function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  if (!sections.length) return;

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('.nav-link').forEach(a => {
          a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`);
        });
      }
    });
  }, { rootMargin: '-30% 0px -60% 0px' });

  sections.forEach(s => obs.observe(s));
}

// ── KaTeX auto-render ──────────────────────────────────────────────────────
function renderMath() {
  if (typeof renderMathInElement === 'undefined') return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true  },
      { left: '$',  right: '$',  display: false },
    ],
    throwOnError: false,
  });
}

// ── Utility: format numbers ────────────────────────────────────────────────
function fmtInt(n)   { return n == null ? 'N/A' : parseInt(n).toLocaleString(); }
function fmtFloat(n, d=4) { return n == null ? 'N/A' : parseFloat(n).toFixed(d); }
function fmtMs(n)    { return n == null ? 'N/A' : `${parseFloat(n).toFixed(1)} ms`; }
function fmtMB(n)    { return n == null ? 'N/A' : `${parseFloat(n).toFixed(1)} MB`; }

window.RFUtils = { fmtInt, fmtFloat, fmtMs, fmtMB };

// ── Populate summary comparison table (used on index & comparison pages) ───
function populateSummaryTable(data) {
  const summary = data?.summary?.models;
  if (!summary) return;

  const fields = {
    'td-cnn-rf':      fmtInt(summary.cnn?.theoretical_rf),
    'td-vit-rf':      fmtInt(summary.vit?.theoretical_rf),
    'td-mamba-rf':    fmtInt(summary.mamba?.theoretical_rf),
    'td-cnn-erf':     fmtFloat(summary.cnn?.effective_rf_radius, 1),
    'td-vit-erf':     fmtFloat(summary.vit?.effective_rf_radius, 1),
    'td-mamba-erf':   fmtFloat(summary.mamba?.effective_rf_radius, 1),
    'td-cnn-params':  fmtInt(summary.cnn?.total_parameters),
    'td-vit-params':  fmtInt(summary.vit?.total_parameters),
    'td-mamba-params':fmtInt(summary.mamba?.total_parameters),
    'td-cnn-ms':      fmtMs(summary.cnn?.inference_time_ms),
    'td-vit-ms':      fmtMs(summary.vit?.inference_time_ms),
    'td-mamba-ms':    fmtMs(summary.mamba?.inference_time_ms),
    'td-cnn-bias':    summary.cnn?.frequency_bias  ?? 'N/A',
    'td-vit-bias':    summary.vit?.frequency_bias  ?? 'N/A',
    'td-mamba-bias':  summary.mamba?.frequency_bias ?? 'N/A',
    'td-cnn-loss':    fmtFloat(summary.cnn?.final_train_loss, 4),
    'td-vit-loss':    fmtFloat(summary.vit?.final_train_loss, 4),
    'td-mamba-loss':  fmtFloat(summary.mamba?.final_train_loss, 4),
  };

  for (const [id, val] of Object.entries(fields)) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
}

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  initProgressBar();
  initSidebar();
  setActiveNavLink();
  initScrollSpy();
  renderMath();

  // Load data then trigger page-specific chart init (defined in charts.js)
  const data = await loadData();
  populateSummaryTable(data);

  if (typeof window.initPageCharts === 'function') {
    window.initPageCharts(data);
  }
});
