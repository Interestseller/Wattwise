/**
 * WattWise — main.js
 * Shared utilities, Plotly theme config, and fetch helpers.
 */

// ── Plotly dark theme config ──────────────────────────────────────────────
const WW = {
  layout(extras = {}) {
    return Object.assign({
      paper_bgcolor: "transparent",
      plot_bgcolor:  "transparent",
      font: { family: "Inter, system-ui, sans-serif", color: "#A5A5B0", size: 11 },
      xaxis: {
        gridcolor:     "#2a2a2a",
        linecolor:     "#303030",
        tickcolor:     "#303030",
        zerolinecolor: "#303030",
        automargin:    true,
      },
      yaxis: {
        gridcolor:     "#2a2a2a",
        linecolor:     "#303030",
        tickcolor:     "#303030",
        zerolinecolor: "#303030",
        automargin:    true,
      },
      margin: { t: 10, r: 14, b: 40, l: 50 },
      legend: {
        bgcolor:     "transparent",
        borderwidth: 0,
        font:        { color: "#A5A5B0", size: 11 },
        orientation: "h",
        xanchor:     "left",
        x: 0,
        y: -0.18,
      },
      hoverlabel: {
        bgcolor:    "#222222",
        bordercolor:"#303030",
        font: { color: "#F5F5F7", size: 12, family: "Inter, sans-serif" },
      },
      hovermode: "x unified",
      modebar:   { bgcolor: "transparent", color: "#A5A5B0" },
      showlegend: true,
    }, extras);
  },

  config() {
    return {
      responsive:              true,
      displayModeBar:          "hover",
      modeBarButtonsToRemove:  ["select2d", "lasso2d", "autoScale2d"],
      displaylogo:             false,
    };
  },

  // Colour constants
  PURPLE:  "#9B87F5",
  PURPLE2: "#B4A1FF",
  GREEN:   "#34d399",
  AMBER:   "#fbbf24",
  RED:     "#f87171",
  GREY:    "#4a4a55",
  GREY2:   "#6e6e78",
};

// ── Safe Plotly render helpers ────────────────────────────────────────────
// Always clear the container first so there's no leftover spinner HTML
// that Plotly would try to work around.

function plotlyReact(idOrEl, traces, layout, config) {
  const el = typeof idOrEl === "string" ? document.getElementById(idOrEl) : idOrEl;
  if (!el) return;
  el.innerHTML = "";                         // wipe any loading spinner
  Plotly.react(el, traces, layout, config);
}

// ── API fetch helpers ─────────────────────────────────────────────────────

async function apiFetch(url, opts = {}) {
  try {
    const r    = await fetch(url, opts);
    const data = await r.json();
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { error: e.message } };
  }
}

async function apiGet(url)       { return apiFetch(url); }
async function apiPost(url, body) {
  return apiFetch(url, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(body),
  });
}

// ── Demo banner ───────────────────────────────────────────────────────────

function showDemoBanner(isDemo) {
  const banner = document.getElementById("demo-banner");
  if (banner && isDemo) banner.style.display = "flex";
}

// ── Loading / error state helpers ─────────────────────────────────────────
// These write into the element directly.
// For chart containers, the spinner sits inside the div until Plotly replaces it.
// plotlyReact() always calls el.innerHTML = "" first, so no conflict occurs.

function setLoading(el, show = true) {
  if (!el) return;
  if (show) {
    // If a Plotly chart already lives here, don't replace its DOM — just
    // reduce opacity so the user sees something is happening.
    if (el._fullLayout) {
      el.style.opacity = "0.4";
      el.style.pointerEvents = "none";
    } else {
      el.innerHTML = `<div class="loading-spinner">
        <div class="spinner"></div><span>Loading…</span>
      </div>`;
    }
  } else {
    el.style.opacity = "";
    el.style.pointerEvents = "";
  }
}

function setError(el, msg) {
  if (!el) return;
  el.innerHTML = `<div class="error-state">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    <span>${msg}</span>
  </div>`;
}

// ── Number formatting ─────────────────────────────────────────────────────

function fmt(n, dec = 2) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toFixed(dec);
}

function fmtK(n) {
  if (n == null) return "—";
  if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + "k";
  return Number(n).toFixed(2);
}

function fmtPct(n) {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return sign + Number(n).toFixed(1) + "%";
}

// ── Patch Plotly.react to clear spinner markup before first render ─────────
// Only wipe innerHTML when the container holds a loading spinner (no chart
// yet). Once Plotly has rendered, el.data is set — leave it alone so that
// Plotly.react can update the chart in-place without losing its DOM state.
(function patchPlotly() {
  if (typeof Plotly === "undefined") {
    document.addEventListener("DOMContentLoaded", patchPlotly);
    return;
  }
  const _orig = Plotly.react.bind(Plotly);
  Plotly.react = function (divOrId, traces, layout, config) {
    const el = typeof divOrId === "string"
      ? document.getElementById(divOrId)
      : divOrId;
    // Clear only when there is no existing Plotly chart in the element
    // (i.e. it still contains a loading spinner or other placeholder HTML).
    if (el && !el._fullLayout) el.innerHTML = "";
    const result = _orig(el || divOrId, traces, layout, config);
    // Restore opacity in case setLoading() dimmed it during a re-fetch.
    if (el) { el.style.opacity = ""; el.style.pointerEvents = ""; }
    return result;
  };
}());

