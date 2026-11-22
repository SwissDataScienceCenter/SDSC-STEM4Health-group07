"""
Bioreactor Challenge — Model Evolution Dashboard
Streamlit app showing side-by-side batch predictions, experiment history, and methodology timeline.
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'lab'))
sys.path.insert(0, str(BASE.parent / 'lab'))

from common import load, T, folds7, mape, feature_groups

st.set_page_config(
    page_title="Bioreactor Model Dashboard — Team 7",
    page_icon="🧬",
    layout="wide",
)

# Load data once
@st.cache_data
def load_all():
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:, 1:]
    return bs, A, Y, tt, test, tr, Ytr

@st.cache_data
def load_oof(name):
    """Load a prediction from new_best_technique oof cache, fall back to main oof."""
    for oof_dir in [BASE / 'lab' / 'oof', BASE.parent / 'lab' / 'oof']:
        path = oof_dir / f'{name}.npz'
        if path.exists():
            data = np.load(path)
            return data['oof'], data.get('test', None)
    return None, None

@st.cache_data
def load_experiment_history():
    """Load all experiment records from both results.jsonl files."""
    records = []
    for results_path in [BASE.parent / 'lab' / 'results.jsonl', BASE / 'lab' / 'results.jsonl']:
        if results_path.exists():
            with open(results_path) as f:
                for line in f:
                    records.append(json.loads(line))
    return records

bs, A, Y, tt, test, tr, Ytr = load_all()
records = load_experiment_history()
train_batches = [bs[i] for i in np.where(tr)[0]]
test_batches = [bs[i] for i in np.where(test)[0]]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🧬 Team 7 Bioreactor Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Batch Comparison", "📈 Experiment History", "🔬 Methodology Timeline", "📋 Model Catalog"],
)

# Shared batch selector
all_batches = list(bs)
default_left = 'Batch5'  # First test batch
default_right = 'Batch1'  # First train batch

if page == "📊 Batch Comparison":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        left_batch = st.selectbox("Left Batch", all_batches, index=all_batches.index(default_left))
    with col2:
        right_batch = st.selectbox("Right Batch", all_batches, index=all_batches.index(default_right))
    
    model_options = [
        'b09_hyb_stat_bayesresid_smoothcells (v1 best)',
        'm01_base_cal (pure ODE)',
        'm22_bypfix_cal (ODE + byproduct fix)',
        'b15_v4 (v4 best)',
    ]
    selected_model = st.sidebar.selectbox("Model to display", model_options)
    show_true = st.sidebar.checkbox("Show ground truth (train batches only)", True)

# ============================================================
# PAGE 1: BATCH COMPARISON
# ============================================================
if page == "📊 Batch Comparison":
    st.title("📊 Batch Comparison — Side by Side")
    st.markdown("Compare two batches across elapsed time. Select batches and a model from the sidebar.")
    
    model_key = selected_model.split(' ')[0]
    oof, test_preds = load_oof(model_key)
    
    if oof is None:
        st.error(f"Model {model_key} not found in OOF cache.")
    else:
        left_idx = list(bs).index(left_batch)
        right_idx = list(bs).index(right_batch)
        
        # Get predictions
        left_data = {}
        right_data = {}
        
        for name, idx in [('Left', left_idx), ('Right', right_idx)]:
            is_test_batch = test[idx]
            if is_test_batch and test_preds is not None:
                # Use test predictions
                test_pos = np.where(test)[0].tolist().index(idx)
                pred = test_preds[test_pos]  # (13, 4)
            else:
                # Use OOF predictions
                train_pos = np.where(tr)[0].tolist().index(idx) if tr[idx] else None
                if train_pos is not None:
                    pred = oof[train_pos]
                else:
                    pred = None
            
            true = Y[idx, 1:, :] if not test[idx] else None
            
            data = {'pred': pred, 'true': true, 'is_test': test[idx]}
            if name == 'Left':
                left_data = data
            else:
                right_data = data
        
        # Plotting
        fig_cols = st.columns(4)
        target_names = ['VCD', 'TCD', 'Substrate', 'Byproduct']
        units = ['10⁶ cells/mL', '10⁶ cells/mL', 'g/L', 'g/L']
        colors_left = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        colors_right = ['#17becf', '#bcbd22', '#98df8a', '#ff9896']
        
        time_left = tt[left_idx, 1:]
        time_right = tt[right_idx, 1:]
        
        for i, (tname, unit, cl, cr) in enumerate(zip(target_names, units, colors_left, colors_right)):
            with fig_cols[i]:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(4, 3))
                
                # Left batch
                if left_data['pred'] is not None:
                    ax.plot(time_left, left_data['pred'][:, i], 'o-', color=cl, 
                            label=f'{left_batch} pred', markersize=4, linewidth=1.5)
                if show_true and left_data['true'] is not None:
                    ax.plot(time_left, left_data['true'][:, i], 's--', color=cl, 
                            label=f'{left_batch} true', markersize=3, alpha=0.5)
                
                # Right batch
                if right_data['pred'] is not None:
                    ax.plot(time_right, right_data['pred'][:, i], 'o-', color=cr,
                            label=f'{right_batch} pred', markersize=4, linewidth=1.5)
                if show_true and right_data['true'] is not None:
                    ax.plot(time_right, right_data['true'][:, i], 's--', color=cr,
                            label=f'{right_batch} true', markersize=3, alpha=0.5)
                
                ax.set_title(f"{tname} ({unit})")
                ax.set_xlabel("Days")
                ax.legend(fontsize=6, loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
        
        # Error table
        st.markdown("---")
        st.subheader("Prediction Error (MAPE %)")
        
        err_rows = []
        for side, data, bname in [('Left', left_data, left_batch), ('Right', right_data, right_batch)]:
            if data['true'] is not None and data['pred'] is not None:
                err = np.abs(data['pred'] - data['true']) / data['true'] * 100
                row = {'Batch': bname, 'Type': 'Train' if not data['is_test'] else 'Test'}
                for i, t in enumerate(T):
                    row[t] = f"{err[:, i].mean():.2f}%"
                row['Overall'] = f"{err.mean():.2f}%"
                err_rows.append(row)
            elif data['is_test']:
                err_rows.append({'Batch': bname, 'Type': 'Test (hidden)', 
                               'vcd': '?', 'tcd': '?', 'substrate': '?', 'byproduct': '?', 'Overall': '?'})
        
        if err_rows:
            st.dataframe(pd.DataFrame(err_rows), use_container_width=True)

# ============================================================
# PAGE 2: EXPERIMENT HISTORY
# ============================================================
elif page == "📈 Experiment History":
    st.title("📈 Experiment History")
    st.markdown(f"**{len(records)} total experiments** run across the project lifecycle.")
    
    # Summary cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    family_counts = {}
    for r in records:
        fam = r.get('family', 'unknown')
        family_counts[fam] = family_counts.get(fam, 0) + 1
    
    family_icons = {
        'mechanistic': '🔧',
        'hybrid': '🔀',
        'statistical': '📊',
        'blend': '🧪',
        'mechanistic-bayes': '🎲',
        'baseline': '📏',
    }
    
    for i, (fam, count) in enumerate(sorted(family_counts.items())):
        cols = [col1, col2, col3, col4, col5, col6]
        with cols[i]:
            icon = family_icons.get(fam, '📦')
            st.metric(f"{icon} {fam}", count)
    
    st.markdown("---")
    
    # Timeline
    st.subheader("Experiment Timeline")
    
    # Parse experiment names to extract groups
    phases = {
        'Grid Search & Baselines (Start)': [],
        'ODE Variants (m01-m22)': [],
        'Hybrid ODE + Kernel (h_)': [],
        'Statistical Models (s_)': [],
        'Bayesian Particle (p/p2_)': [],
        'Blends & Ensembles (b_)': [],
        'Final Optimization (v4-v9)': [],
        'Agents Experiments (improvements)': [],
    }
    
    import re
    for r in records:
        name = r.get('name', '')
        if name.startswith('m0') or name.startswith('m1') or name.startswith('m2'):
            phases['ODE Variants (m01-m22)'].append(r)
        elif name.startswith('h_') or name.startswith('hf_') or name.startswith('hm_'):
            phases['Hybrid ODE + Kernel (h_)'].append(r)
        elif name.startswith('s'):
            phases['Statistical Models (s_)'].append(r)
        elif name.startswith('p_') or name.startswith('p2_'):
            phases['Bayesian Particle (p/p2_)'].append(r)
        elif name.startswith('b'):
            phases['Blends & Ensembles (b_)'].append(r)
        elif name.startswith('v') or name.startswith('x_'):
            phases['Final Optimization (v4-v9)'].append(r)
        elif 'pure' in name or 'cal' in name:
            phases['ODE Variants (m01-m22)'].append(r)
        else:
            phases['Grid Search & Baselines (Start)'].append(r)
    
    for phase_name, phase_records in phases.items():
        if phase_records:
            with st.expander(f"{phase_name} ({len(phase_records)} experiments)", expanded=len(phase_records) < 15):
                # Show best from this phase
                phase_records.sort(key=lambda r: r.get('extra', {}).get('insample', 99))
                cols = st.columns([3, 1, 1, 1, 1, 1])
                cols[0].markdown("**Name**")
                cols[1].markdown("**VCD**")
                cols[2].markdown("**TCD**")
                cols[3].markdown("**Sub**")
                cols[4].markdown("**Byp**")
                cols[5].markdown("**Overall**")
                
                for r in phase_records[:10]:  # Show top 10
                    cols = st.columns([3, 1, 1, 1, 1, 1])
                    name = r.get('name', '?')
                    extra = r.get('extra', {})
                    ins = extra.get('insample', None)
                    
                    cols[0].markdown(f"`{name[:45]}`")
                    if isinstance(ins, (int, float)):
                        cols[5].markdown(f"{ins:.2f}%")
                    else:
                        cols[5].markdown("—")
                
                if len(phase_records) > 10:
                    st.caption(f"... and {len(phase_records) - 10} more")

# ============================================================
# PAGE 3: METHODOLOGY TIMELINE
# ============================================================
elif page == "🔬 Methodology Timeline":
    st.title("🔬 Methodology Timeline")
    st.markdown("How the modeling approach evolved from grid search to the winning submission.")
    
    timeline = [
        {
            "phase": "Phase 1: Grid Search & Baselines",
            "color": "#e74c3c",
            "details": [
                "Scanned 140+ ODE configurations (Monod parameters, inhibition constants, yield factors)",
                "Evaluated 4 model families: mechanistic ODE, hybrid ODE+kernel, statistical kernel, Bayesian particle",
                "Established 7-fold cross-validation framework",
                "Best baseline: ~8-9% MAPE per target",
            ]
        },
        {
            "phase": "Phase 2: ODE Variants (22 variants)",
            "color": "#e67e22",
            "details": [
                "Tested 22 ODE variants with different kinetic hypotheses:",
                "• m01-m02: Base Monod with/without covariates",
                "• m03: Feed-D promoted byproduct re-consumption",
                "• m04: Byproduct inhibition of growth",
                "• m05: Logistic capacity instead of IVCD inhibition",
                "• m06: Unmeasured nutrient limitation",
                "• m07-m08: Phase-dependent (time-varying) parameters",
                "• m09: Multi-pathway death (substrate + byproduct)",
                "• m10: Hill-function overflow kinetics",
                "• m11: Switch-like byproduct re-consumption",
                "• m19: Combined feed-D + switch consumption",
                "• m22: **Fixed consumption kinetics** (S/(K+S) instead of backwards Kg/(Kg+S))",
                "Best ODE: m22_bypfix improved byproduct by 3.75% vs base",
            ]
        },
        {
            "phase": "Phase 3: Hybrid Models",
            "color": "#f1c40f",
            "details": [
                "Kernel ridge regression on ODE residual errors",
                "7-10 process signals as covariates (feed timing, volume changes, substrate flags)",
                "Evidence-weighted vs search-optimized variants",
                "Best: h_m11_consHill (hybrid with switch-like consumption)",
                "Hybrid improvements: ~1.5-2% MAPE over pure ODE",
            ]
        },
        {
            "phase": "Phase 4: Statistical & Bayesian Models",
            "color": "#2ecc71",
            "details": [
                "Statistical kernel models (s_): purely data-driven, no ODE",
                "Bayesian particle filters (p_): 100-1000 particles estimating state from ODE + noise",
                "Covariance matrix adaptation across 7-10 dimensions",
                "Best: s25 & p2_m21 — complementary error patterns to hybrid ODE",
            ]
        },
        {
            "phase": "Phase 5: Blends & Smoothing",
            "color": "#3498db",
            "details": [
                "Geometric mean blends: equal-weight, optimized, evidence-weighted",
                "Tested 3/4/6/8 member combinations",
                "Cell trajectory smoothing (lowpass filter on log-space) — critical for VCD/TCD",
                "Best: b09 — 3-member equal-weight blend + 3-pass cell smoothing → 7.03%",
                "Discovered: optimizing blend weights in CV *worsens* leaderboard score",
            ]
        },
        {
            "phase": "Phase 6: Day-by-Day Sugar Simulator",
            "color": "#9b59b6",
            "details": [
                "Game-changing innovation: simulate substrate day-by-day from feed events",
                "Incorporates operators' bolus rules as hard constraints:",
                "• Days 7-9: bolus if substrate < 1.95",
                "• Days 11-12: 0.6% if < 0.967, else 0.4%",
                "Substrate error dropped from 7.23% → 6.33%",
                "Multiple chain variants with selective shrinkage",
            ]
        },
        {
            "phase": "Phase 7: Grid CV & Kernel Ridge (v4-v8)",
            "color": "#1abc9c",
            "details": [
                "GridCV: systematic search over 100 bootstrapped splits → ~800 experiments",
                "Discovered: small statistical add-ons for cells hurt on test (overfit on 49 batches)",
                "autogrid.py: kernel ridge on wide features → VCD 6.627% (best ever)",
                "Per-measurement optimization: pick best column from multiple submissions",
            ]
        },
        {
            "phase": "Phase 8: Agent Experiments (v9)",
            "color": "#e91e63",
            "details": [
                "Nested cross-validation framework: honest evaluation (eliminated selection bias)",
                "Tested fundamentally different approaches: GBT, NN, GP, FPCA, LASSO stack",
                "Verified: pure ML (no ODE) floors at ~8.5% — physics provides ~1.5% irreducible value",
                "Tested RegressorChain, KNN trajectory matching, decomposed targets, log-space per-day LGB",
                "Generated 42+ submission variants for leaderboard testing",
                "Final v9: per-target cherry-pick → 6.709% leaderboard (1st place)",
            ]
        },
    ]
    
    for phase in timeline:
        with st.expander(f"**{phase['phase']}**", expanded=False):
            st.markdown(f"<div style='border-left: 4px solid {phase['color']}; padding-left: 16px;'>", unsafe_allow_html=True)
            for detail in phase['details']:
                st.markdown(detail)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Leaderboard progression
    st.markdown("---")
    st.subheader("Leaderboard Score Progression")
    
    scores_data = [
        ("v1 (b09)", 7.018, "#e74c3c"),
        ("v4", 7.088, "#e67e22"),
        ("v6 (mix)", 6.853, "#f1c40f"),
        ("v7 (single shot)", 7.051, "#2ecc71"),
        ("pa (ODE cells)", 6.782, "#3498db"),
        ("pc (kernel ridge)", 6.845, "#9b59b6"),
        ("v9 (final)", 6.709, "#1abc9c"),
    ]
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    names = [s[0] for s in scores_data]
    values = [s[1] for s in scores_data]
    colors = [s[2] for s in scores_data]
    
    ax.barh(names, values, color=colors, height=0.5)
    ax.set_xlabel('MAPE %')
    ax.axvline(x=7.0, color='gray', linestyle='--', alpha=0.5, label='7.0% baseline')
    ax.legend()
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f'{v:.3f}%', va='center', fontweight='bold')
    st.pyplot(fig)

# ============================================================
# PAGE 4: MODEL CATALOG
# ============================================================
elif page == "📋 Model Catalog":
    st.title("📋 Model Catalog")
    st.markdown(f"**{len(records)} models** across 6 families. Click to see details.")
    
    search = st.text_input("Search models", placeholder="e.g., b09, m22, h_m11...")
    
    filtered = records
    if search:
        filtered = [r for r in records if search.lower() in r.get('name', '').lower()]
    
    family_filter = st.multiselect(
        "Filter by family",
        sorted(set(r.get('family', '?') for r in records)),
        default=[]
    )
    if family_filter:
        filtered = [r for r in filtered if r.get('family') in family_filter]
    
    st.markdown(f"Showing {len(filtered)} of {len(records)} models")
    
    for r in sorted(filtered, key=lambda x: x.get('extra', {}).get('insample', 99))[:30]:
        name = r.get('name', '?')
        fam = r.get('family', '?')
        extra = r.get('extra', {})
        ins = extra.get('insample', None)
        
        with st.expander(f"`{name}` — {fam} — {ins:.2f}%" if isinstance(ins, float) else f"`{name}` — {fam}"):
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Family:** {fam}")
                if isinstance(ins, float):
                    st.markdown(f"**In-sample MAPE:** {ins:.2f}%")
                st.markdown(f"**Description:** {r.get('desc', 'N/A')[:200]}")
            with cols[1]:
                if 'extra' in r:
                    st.json(r['extra'], expanded=False)

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption(f"Total experiments: {len(records)}")
st.sidebar.caption(f"OOF predictions: 145 (main) + 255 (v9)")
st.sidebar.caption(f"Training batches: 49 | Test batches: 12")
st.sidebar.caption("Team 7 — SDSC STEM4Health Hackathon")
