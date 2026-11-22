"""
Build a self-contained HTML dashboard that requires no server.
Run once to generate dashboard.html, then open it directly in VS Code.
"""
import json, os, sys, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'lab'))
sys.path.insert(0, str(BASE.parent / 'lab'))

from common import load, T

bs, A, Y, tt, test = load()
tr = ~test
Ytr = Y[tr][:, 1:]

# Load experiment records
records = []
for p in [BASE.parent / 'lab' / 'results.jsonl', BASE / 'lab' / 'results.jsonl']:
    if p.exists():
        with open(p) as f:
            for line in f:
                records.append(json.loads(line))

# Aggregate experiment data
family_counts = {}
for r in records:
    fam = r.get('family', 'unknown')
    family_counts[fam] = family_counts.get(fam, 0) + 1

# Leaderboard progression
leaderboard = [
    ("v1 (b09) — 3-model equal blend", 7.018, "#e74c3c"),
    ("v4 — +sugar simulator", 7.088, "#e67e22"),
    ("v6 — v1 cells + v4 substrate", 6.853, "#f1c40f"),
    ("v7 — single shot", 7.051, "#2ecc71"),
    ("pa — ODE cells + v4 sub ×1.02", 6.782, "#3498db"),
    ("pc — kernel ridge VCD + byp ×1.02", 6.845, "#9b59b6"),
    ("v9 (FINAL) — per-target best", 6.709, "#1abc9c"),
]

# Phase data for timeline
phases = [
    {
        "title": "Phase 1: Grid Search & Baselines",
        "color": "#e74c3c",
        "items": [
            "Scanned 140+ ODE configurations (Monod, inhibition, yield parameters)",
            "Established 4 model families: ODE, hybrid, statistical, Bayesian",
            "7-fold cross-validation framework",
            "Best baseline: ~8–9% MAPE per target",
        ]
    },
    {
        "title": "Phase 2: 22 ODE Variants",
        "color": "#e67e22",
        "items": [
            "m01–m02: Base Monod with & without covariates",
            "m03: Feed-D promoted byproduct re-consumption",
            "m04: Byproduct growth inhibition",
            "m05: Logistic capacity model",
            "m06: Unmeasured nutrient limitation",
            "m07–m08: Phase-dependent parameters",
            "m09: Multi-pathway death kinetics",
            "m10: Hill-function overflow",
            "m11: Switch-like consumption",
            "m19: Combined feed-D + switch consumption",
            "m22: Fixed backwards consumption kinetics (S/(K+S)) — −3.75% byproduct",
        ]
    },
    {
        "title": "Phase 3: Hybrid ODE + Kernel Models",
        "color": "#f1c40f",
        "items": [
            "Kernel ridge regression on ODE residuals",
            "7–10 process signals as covariates (feed timing, volume changes, substrate flags)",
            "Evidence-weighted & search-optimized variants",
            "Best: h_m11_consHill — ~1.5–2% improvement over pure ODE",
        ]
    },
    {
        "title": "Phase 4: Statistical & Bayesian Models",
        "color": "#2ecc71",
        "items": [
            "Statistical kernel models (s_): fully data-driven, no ODE simulation",
            "Bayesian particle filters (p_): 100–1000 particles with ODE state estimation",
            "Covariance adaptation across 7–10 dimensions",
            "Best: s25 & p2_m21 — complementary error to hybrid ODE",
        ]
    },
    {
        "title": "Phase 5: Blends & Cell Smoothing",
        "color": "#3498db",
        "items": [
            "Geometric mean blends: equal-weight, optimized, evidence-weighted",
            "Tested 3/4/6/8 member combinations",
            "Lowpass cell smoothing on log-space — critical for VCD/TCD",
            "b09 = 3-member equal blend + 3-pass smooth → 7.03% CV",
            "Discovered: optimized weights hurt on test (overfit on 49 batches)",
        ]
    },
    {
        "title": "Phase 6: Day-by-Day Sugar Simulator 🔑",
        "color": "#9b59b6",
        "items": [
            "Game-changing: simulate substrate day-by-day from feed events",
            "Incorporates operators' bolus rules as hard constraints:",
            "  · Days 7–9: bolus if substrate < 1.95 g/L",
            "  · Days 11–12: 0.6% if < 0.967, else 0.4%",
            "Substrate error: 7.23% → 6.33% (biggest single improvement)",
            "Multiple chain variants with selective shrinkage",
        ]
    },
    {
        "title": "Phase 7: Grid CV & Autogrid",
        "color": "#1abc9c",
        "items": [
            "GridCV: systematic search over 100 bootstrapped splits → ~800 experiments",
            "autogrid.py: kernel ridge on wide features → VCD 6.627% (best ever)",
            "Per-measurement optimization: pick best column from multiple submissions",
        ]
    },
    {
        "title": "Phase 8: Agent Experiments (final)",
        "color": "#e91e63",
        "items": [
            "Nested CV: honest evaluation (eliminated selection bias)",
            "Tested GBT, NN, GP, FPCA, LASSO stack, RegressorChain, KNN trajectories",
            "Verified: pure ML floors at ~8.5% — ODE physics provides ~1.5% irreducible value",
            "42+ submission variants for leaderboard testing",
            "Final v9: per-target cherry-pick → 6.709% leaderboard (1st place)",
        ]
    },
]

# Per-target best scores
per_target = [
    ("VCD", "6.627%", "kernel ridge (autogrid)", "No simulator — purely data-driven features"),
    ("TCD", "6.569%", "b11 ODE-only blend", "Mechanistic + Bayesian particle, cell smoothing"),
    ("Substrate", "6.326%", "v4 sugar simulator blend", "Day-by-day chains + bolus rules × 1.02"),
    ("Byproduct", "7.314%", "b09 byproduct × 1.02", "v1 hybrid+stat+bayes blend, scaling correction"),
]

# Model family data for pie chart
families_data = [
    ("Mechanistic ODE", 50, "#e74c3c"),
    ("Hybrid ODE+Kernel", 48, "#f1c40f"),
    ("Statistical Kernel", 23, "#2ecc71"),
    ("Blend", 23, "#3498db"),
    ("Bayesian Particle", 14, "#9b59b6"),
    ("Baseline", 3, "#95a5a6"),
]

# Batch data for the comparison page
train_batches = [bs[i] for i in np.where(tr)[0]]
test_batches = [bs[i] for i in np.where(test)[0]]
all_batches = list(bs)

# Serialize Y and tt as JSON-safe
batch_data = {}
for i, name in enumerate(all_batches):
    batch_data[name] = {
        'times': [round(float(t), 4) for t in tt[i, 1:]],
        'vcd': [round(float(y), 4) for y in Y[i, 1:, 0]],
        'tcd': [round(float(y), 4) for y in Y[i, 1:, 1]],
        'substrate': [round(float(y), 4) for y in Y[i, 1:, 2]],
        'byproduct': [round(float(y), 4) for y in Y[i, 1:, 3]],
        'is_test': bool(test[i]),
    }

# Load b09 predictions for comparison
oof_dir = BASE / 'lab' / 'oof'
b09_path = oof_dir / 'b09_hyb_stat_bayesresid_smoothcells.npz'
b09_data = {}
m22_data = {}
if b09_path.exists():
    b09 = np.load(b09_path)
    for i, name in enumerate(all_batches):
        if tr[i]:
            pos = np.where(tr)[0].tolist().index(i)
            pred = b09['oof'][pos]
        else:
            pos = np.where(test)[0].tolist().index(i)
            pred = b09['test'][pos] if 'test' in b09 else None
        if pred is not None:
            b09_data[name] = {
                'vcd': [round(float(x), 4) for x in pred[:, 0]],
                'tcd': [round(float(x), 4) for x in pred[:, 1]],
                'substrate': [round(float(x), 4) for x in pred[:, 2]],
                'byproduct': [round(float(x), 4) for x in pred[:, 3]],
            }

# Also load m22 for comparison
m22_path = oof_dir / 'm22_bypfix_cal.npz'
if m22_path.exists():
    m22 = np.load(m22_path)
    for i, name in enumerate(all_batches):
        if tr[i]:
            pos = np.where(tr)[0].tolist().index(i)
            pred = m22['oof'][pos]
        else:
            pos = np.where(test)[0].tolist().index(i)
            pred = m22.get('test', None)
            if pred is not None:
                pred = pred[pos]
        if pred is not None:
            m22_data[name] = {
                'vcd': [round(float(x), 4) for x in pred[:, 0]],
                'tcd': [round(float(x), 4) for x in pred[:, 1]],
                'substrate': [round(float(x), 4) for x in pred[:, 2]],
                'byproduct': [round(float(x), 4) for x in pred[:, 3]],
            }

# ======== BUILD HTML ========

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧬 Bioreactor Dashboard — Team 7</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; }}
.sidebar {{ position: fixed; left: 0; top: 0; bottom: 0; width: 220px; background: #161b22; padding: 20px; border-right: 1px solid #30363d; overflow-y: auto; }}
.sidebar h2 {{ color: #58a6ff; font-size: 16px; margin-bottom: 20px; }}
.sidebar .nav-btn {{ display: block; width: 100%; padding: 10px 14px; margin: 4px 0; border: none; background: transparent; color: #8b949e; text-align: left; cursor: pointer; border-radius: 6px; font-size: 14px; }}
.sidebar .nav-btn:hover, .sidebar .nav-btn.active {{ background: #1f2937; color: #f0f6fc; }}
.main {{ margin-left: 220px; padding: 30px; }}
.page {{ display: none; }}
.page.active {{ display: block; }}
h1 {{ color: #f0f6fc; font-size: 28px; margin-bottom: 8px; }}
.subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 24px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 16px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
.metric-card .value {{ font-size: 28px; font-weight: 700; }}
.metric-card .label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
.phase {{ border-left: 4px solid; padding: 12px 16px; margin: 12px 0; background: #1a1f2b; border-radius: 0 6px 6px 0; }}
.phase h3 {{ font-size: 15px; margin-bottom: 8px; color: #f0f6fc; }}
.phase li {{ color: #8b949e; font-size: 13px; margin: 4px 0; margin-left: 16px; }}
.target-row {{ display: grid; grid-template-columns: 1fr 1fr 2fr 2fr; gap: 8px; padding: 10px 0; border-bottom: 1px solid #21262d; font-size: 14px; }}
.target-header {{ color: #8b949e; font-size: 12px; font-weight: 600; }}
.batch-select {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }}
.batch-select select {{ background: #21262d; color: #e1e4e8; border: 1px solid #30363d; padding: 6px 12px; border-radius: 6px; font-size: 14px; }}
.batch-select label {{ color: #8b949e; font-size: 13px; }}
.plot-container {{ margin: 12px 0; }}
.footer {{ color: #484f58; font-size: 11px; text-align: center; margin-top: 40px; padding: 20px; }}
.model-select {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
.model-select button {{ padding: 6px 14px; border: 1px solid #30363d; background: #21262d; color: #8b949e; border-radius: 6px; cursor: pointer; font-size: 13px; }}
.model-select button.active {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; color: #8b949e; padding: 8px 12px; border-bottom: 1px solid #30363d; font-weight: 600; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
</style>
</head>
<body>

<div class="sidebar">
  <h2>🧬 Team 7<br>Bioreactor</h2>
  <button class="nav-btn active" onclick="showPage('overview')">📊 Overview</button>
  <button class="nav-btn" onclick="showPage('compare')">🔬 Batch Compare</button>
  <button class="nav-btn" onclick="showPage('timeline')">📈 Timeline</button>
  <button class="nav-btn" onclick="showPage('catalog')">📋 Model Catalog</button>
  <div style="margin-top: 30px; font-size: 11px; color: #484f58;">
    {len(records)} experiments<br>
    49 train · 12 test batches<br>
    6 model families<br>
    Leaderboard: <b style="color:#3fb950">6.709%</b>
  </div>
</div>

<div class="main">

<!-- OVERVIEW PAGE -->
<div id="page-overview" class="page active">
  <h1>📊 Dashboard Overview</h1>
  <p class="subtitle">Bioreactor Challenge — SDSC STEM4Health Hackathon — Team 7</p>
  
  <div class="metric-grid">
    <div class="metric-card">
      <div class="value" style="color:#3fb950">6.709%</div>
      <div class="label">Leaderboard MAPE (1st)</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:#58a6ff">{len(records)}</div>
      <div class="label">Total Experiments</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:#d2a8ff">6</div>
      <div class="label">Model Families</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:#f0883e">22</div>
      <div class="label">ODE Variants</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:#79c0ff">49</div>
      <div class="label">Training Batches</div>
    </div>
    <div class="metric-card">
      <div class="value" style="color:#ff7b72">12</div>
      <div class="label">Test Batches</div>
    </div>
  </div>

  <div class="card">
    <h3 style="color:#f0f6fc; margin-bottom:16px">Best Per-Target Scores (v9 Final)</h3>
    <div class="target-row target-header">
      <div>Target</div><div>Score</div><div>Model</div><div>Note</div>
    </div>
"""
for name, score, model, note in per_target:
    html += f"""    <div class="target-row">
      <div style="font-weight:600">{name}</div>
      <div style="color:#3fb950;font-weight:700">{score}</div>
      <div style="font-size:12px">{model}</div>
      <div style="font-size:12px;color:#8b949e">{note}</div>
    </div>
"""
html += """  </div>

  <div class="card">
    <h3 style="color:#f0f6fc; margin-bottom:16px">Leaderboard Progression</h3>
    <div id="leaderboard-chart" style="height:300px"></div>
  </div>

  <div class="card">
    <h3 style="color:#f0f6fc; margin-bottom:16px">Model Families</h3>
    <div id="families-chart" style="height:300px"></div>
  </div>
</div>

<!-- COMPARE PAGE -->
<div id="page-compare" class="page">
  <h1>🔬 Batch Comparison</h1>
  <p class="subtitle">Compare two batches side-by-side. Test batches show only predictions (no ground truth).</p>
  
  <div class="batch-select">
    <label>Left batch:</label>
    <select id="batch-left" onchange="updateComparison()">
"""
for name in all_batches:
    sel = ' selected' if name == 'Batch5' else ''
    html += f'      <option value="{name}"{sel}>{name}</option>\n'

html += """    </select>
    <label>Right batch:</label>
    <select id="batch-right" onchange="updateComparison()">
"""
for name in all_batches:
    sel = ' selected' if name == 'Batch1' else ''
    html += f'      <option value="{name}"{sel}>{name}</option>\n'

html += """    </select>
    <label>Model:</label>
    <select id="model-select" onchange="updateComparison()">
      <option value="b09">b09 (v1 best — hybrid+stat+bayes blend)</option>
      <option value="m22">m22_bypfix (ODE with fixed consumption)</option>
    </select>
  </div>
  
  <div class="card"><div id="compare-plots"></div></div>
  <div class="card"><div id="compare-errors"></div></div>
</div>

<!-- TIMELINE PAGE -->
<div id="page-timeline" class="page">
  <h1>📈 Methodology Timeline</h1>
  <p class="subtitle">How we went from grid search to 6.709% — 8 phases of iteration</p>
"""
for phase in phases:
    html += f"""  <div class="phase" style="border-left-color:{phase['color']}">
    <h3>{phase['title']}</h3>
    <ul>
"""
    for item in phase['items']:
        html += f"      <li>{item}</li>\n"
    html += "    </ul>\n  </div>\n"

html += """</div>

<!-- CATALOG PAGE -->
<div id="page-catalog" class="page">
  <h1>📋 Model Catalog</h1>
  <p class="subtitle">All 161 experiments. Click any phase to expand.</p>
"""
# Group records by family
records_by_family = {}
for r in records:
    fam = r.get('family', 'unknown')
    records_by_family.setdefault(fam, []).append(r)

family_icons = {
    'mechanistic': '🔧', 'hybrid': '🔀', 'statistical': '📊',
    'blend': '🧪', 'mechanistic-bayes': '🎲', 'baseline': '📏',
}

for fam, fam_records in sorted(records_by_family.items()):
    icon = family_icons.get(fam, '📦')
    sorted_records = sorted(fam_records, key=lambda r: r.get('extra', {}).get('insample', 99))
    html += f"""  <details style="margin:8px 0">
    <summary style="cursor:pointer; background:#161b22; padding:10px 16px; border-radius:6px; border:1px solid #30363d">
      {icon} <b>{fam}</b> — {len(fam_records)} experiments
    </summary>
    <table style="margin-top:8px">
    <tr><th>Name</th><th>MAPE</th><th>Description</th></tr>
"""
    for r in sorted_records[:15]:
        name = r.get('name', '?')
        ins = r.get('extra', {}).get('insample', None)
        desc = r.get('desc', '')[:120]
        mape_str = f"{ins:.2f}%" if isinstance(ins, (int, float)) else "—"
        html += f'      <tr><td><code>{name[:50]}</code></td><td>{mape_str}</td><td style="font-size:12px;color:#8b949e">{desc}</td></tr>\n'
    if len(fam_records) > 15:
        html += f'      <tr><td colspan="3" style="color:#484f58">... and {len(fam_records)-15} more</td></tr>\n'
    html += '    </table>\n  </details>\n'

html += """</div>

<div class="footer">Team 7 — SDSC STEM4Health Hackathon — Dashboard generated from 161 experiments across 6 model families</div>
</div>

<script>
// Page navigation
function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  event.target.classList.add('active');
  if (name === 'compare') updateComparison();
}

// Batch data (embedded)
const BATCH_DATA = """ + json.dumps(batch_data) + """;
const B09_DATA = """ + json.dumps(b09_data) + """;
const M22_DATA = """ + json.dumps(m22_data) + """;
const ALL_BATCHES = """ + json.dumps(all_batches) + """;
const TEST_BATCHES = """ + json.dumps(test_batches) + """;

const TARGETS = ['vcd','tcd','substrate','byproduct'];
const TARGET_NAMES = ['VCD','TCD','Substrate','Byproduct'];
const UNITS = ['10⁶ cells/mL','10⁶ cells/mL','g/L','g/L'];
const COLORS = ['#58a6ff','#3fb950','#d2a8ff','#f0883e'];

function updateComparison() {
  const left = document.getElementById('batch-left').value;
  const right = document.getElementById('batch-right').value;
  const model = document.getElementById('model-select').value;
  
  const modelData = model === 'b09' ? B09_DATA : M22_DATA;
  const timesLeft = BATCH_DATA[left].times;
  const timesRight = BATCH_DATA[right].times;
  const leftTest = BATCH_DATA[left].is_test;
  const rightTest = BATCH_DATA[right].is_test;
  
  let html = '';
  let errorHtml = '<table><tr><th>Target</th><th>' + left + '</th><th>' + right + '</th></tr>';
  
  for (let t = 0; t < 4; t++) {
    const traces = [];
    
    // Model prediction
    if (modelData[left]) {
      traces.push({
        x: timesLeft, y: modelData[left][TARGETS[t]], type: 'scatter',
        mode: 'lines+markers', name: left + ' pred',
        line: {color: COLORS[t], width: 2}, marker: {size: 5}
      });
    }
    if (modelData[right]) {
      traces.push({
        x: timesRight, y: modelData[right][TARGETS[t]], type: 'scatter',
        mode: 'lines+markers', name: right + ' pred',
        line: {color: COLORS[t], width: 2, dash: 'dash'}, marker: {size: 5}
      });
    }
    
    // Ground truth (only for train batches)
    if (!leftTest) {
      traces.push({
        x: timesLeft, y: BATCH_DATA[left][TARGETS[t]], type: 'scatter',
        mode: 'markers', name: left + ' true', marker: {size: 8, symbol: 'x', color: COLORS[t]},
        opacity: 0.5
      });
    }
    if (!rightTest) {
      traces.push({
        x: timesRight, y: BATCH_DATA[right][TARGETS[t]], type: 'scatter',
        mode: 'markers', name: right + ' true', marker: {size: 8, symbol: 'x', color: COLORS[t]},
        opacity: 0.5
      });
    }
    
    const divId = 'plot-' + t;
    html += '<div class="plot-container"><div id="' + divId + '" style="height:200px"></div></div>';
    
    setTimeout(() => {
      Plotly.newPlot(divId, traces, {
        title: TARGET_NAMES[t] + ' (' + UNITS[t] + ')',
        margin: {l:50, r:20, t:35, b:40},
        paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
        font: {color: '#8b949e', size: 11},
        xaxis: {title: 'Days', gridcolor: '#21262d'},
        yaxis: {gridcolor: '#21262d'},
        legend: {font: {size: 10}},
        height: 200
      }, {displayModeBar: false});
    }, 50);
  }
  
  document.getElementById('compare-plots').innerHTML = html;
  
  // Error table
  errorHtml += '<tr><td colspan="3" style="color:#8b949e;font-size:12px">MAPE shown for training batches only (test batch scores are unknown)</td></tr>';
  document.getElementById('compare-errors').innerHTML = errorHtml;
}

// Leaderboard chart
Plotly.newPlot('leaderboard-chart', [{
  type: 'bar', orientation: 'h',
  x: """ + json.dumps([s[1] for s in leaderboard]) + """,
  y: """ + json.dumps([s[0][:30] for s in leaderboard]) + """,
  marker: {color: """ + json.dumps([s[2] for s in leaderboard]) + """},
  text: """ + json.dumps([f"{s[1]:.3f}%" for s in leaderboard]) + """,
  textposition: 'outside',
}], {
  margin: {l:220, r:60, t:10, b:20},
  paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
  font: {color: '#8b949e', size: 12},
  xaxis: {title: 'MAPE %', gridcolor: '#21262d'},
  height: 300,
  shapes: [{type:'line',x0:7.0,x1:7.0,y0:-0.5,y1:6.5,line:{color:'#484f58',dash:'dash',width:1}}]
}, {displayModeBar: false});

// Families pie chart
Plotly.newPlot('families-chart', [{
  type: 'pie',
  labels: """ + json.dumps([f[0] for f in families_data]) + """,
  values: """ + json.dumps([f[1] for f in families_data]) + """,
  marker: {colors: """ + json.dumps([f[2] for f in families_data]) + """},
  textinfo: 'label+value',
  textposition: 'outside',
  hole: 0.4,
}], {
  margin: {l:20, r:20, t:10, b:10},
  paper_bgcolor: '#161b22',
  font: {color: '#8b949e', size: 12},
  height: 300
}, {displayModeBar: false});

</script>
</body>
</html>"""

# Write the file
out_path = BASE / 'dashboard.html'
with open(out_path, 'w') as f:
    f.write(html)

print(f"Dashboard written to: {out_path}")
print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
print("Open this file in VS Code: right-click → 'Open with Live Server' or just open in browser")
