"""
Experiment: apply the M22 byproduct fix as a post-hoc correction to b09.

The M22 ODE fix changed the byproduct consumption kinetics from:
  cons = kc * P * Kg/(Kg+S)   [consumption increases when S drops — WRONG]
to:
  cons = kc * P * S/(K_cons+S)  [consumption requires substrate — correct]

Since we can't retrain the full hybrid pipeline (hours), we test whether
applying the byproduct correction factor to b09's output helps.

We analyze b09's byproduct error and see if a S-dependent correction 
(similar to M22's kinetic change) reduces CV error.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab'))
import numpy as np, json
from common import load, T, folds7, mape

bs, A, Y, tt, test = load()
tr = ~test
Ytr = Y[tr][:, 1:]
n = len(bs)
trn = np.where(tr)[0]
oof_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lab', 'oof')

# Load b09
b09 = np.load(os.path.join(oof_dir, 'b09_hyb_stat_bayesresid_smoothcells.npz'))
P_b09_oof = b09['oof']  # (49, 13, 4)

# ============================================================
# ANALYSIS 1: Byproduct error vs substrate level
# ============================================================

# At what substrate levels does b09 over/under predict byproduct?
byp_rel_err = (P_b09_oof[:, :, 3] - Ytr[:, :, 3]) / Ytr[:, :, 3]
sub_values = Ytr[:, :, 2].ravel()
byp_errors = byp_rel_err.ravel()

# Bin by substrate level
bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
print("=== Byproduct error vs substrate level ===")
for i in range(len(bins)-1):
    mask = (sub_values >= bins[i]) & (sub_values < bins[i+1])
    if mask.sum() > 0:
        mean_err = byp_errors[mask].mean() * 100
        mape_bin = np.abs(byp_errors[mask]).mean() * 100
        print(f"  S=[{bins[i]:.1f},{bins[i+1]:.1f}): n={mask.sum():4d}  bias={mean_err:+5.1f}%  mape={mape_bin:.1f}%")

# ============================================================
# EXPERIMENT: Substrate-dependent byproduct correction
# ============================================================
# 
# M22 says: cons ∝ S/(K+S), so net byproduct ∝ 1/S at low S
# We test: P_byp_new = P_byp_b09 * S^alpha (alpha fitted per CV fold)

folds = folds7(tr)
pos = {r: i for i, r in enumerate(np.where(tr)[0])}
best_alphas = []

print("\n=== Cross-validated byproduct correction ===")
for fold_idx, ho in enumerate(folds):
    ho_pos = [pos[r] for r in ho]
    tr_fold = [pos[r] for r in np.where(tr)[0] if r not in ho]
    
    # Find best alpha on training fold
    best_alpha = 0.0
    best_err = 1e9
    for alpha in np.linspace(-0.3, 0.3, 31):
        S_train = Ytr[tr_fold, :, 2]
        correction = np.clip(S_train / 1.5, 0.3, 3.0) ** alpha  # normalize around S=1.5
        P_corrected = P_b09_oof[tr_fold, :, 3] * correction
        err = np.abs(P_corrected - Ytr[tr_fold, :, 3]).mean() / Ytr[tr_fold, :, 3].mean() * 100
        if err < best_err:
            best_err = err
            best_alpha = alpha
    
    best_alphas.append(best_alpha)
    
    # Apply to validation fold
    S_val = Ytr[ho_pos, :, 2]
    correction = np.clip(S_val / 1.5, 0.3, 3.0) ** best_alpha
    # (don't modify in-place, just track)

print(f"  Best alphas per fold: {[f'{a:.3f}' for a in best_alphas]}")
print(f"  Mean alpha: {np.mean(best_alphas):.4f} ± {np.std(best_alphas):.4f}")
print(f"  If alpha ≈ 0: no S-dependence needed")
print(f"  If alpha < 0: byproduct should be LOWER at high S (M22 overflow threshold)")
print(f"  If alpha > 0: byproduct should be HIGHER at high S")

# ============================================================
# EXPERIMENT 2: Also fix substrate using the sugar simulator insight
# ============================================================
# v4/v9's substrate error dropped from 7.23 to 6.33 using day-by-day simulators
# Can we apply this to b09's substrate?

# Check: b09's substrate error vs day
print("\n=== b09 substrate error by day ===")
err_b09 = np.abs(P_b09_oof - Ytr) / Ytr * 100
for d in range(13):
    print(f"  Day {d+1:2d}: {err_b09[:,d,2].mean():.1f}%  (byp={err_b09[:,d,3].mean():.1f}%)")

# ============================================================
# EXPERIMENT 3: Build a new blend using M22 ODE 
# ============================================================
# Check if M22 OOF is available
m22_path = os.path.join(oof_dir, 'm22_bypfix_cal.npz')
if os.path.exists(m22_path):
    m22 = np.load(m22_path)
    err_m22 = np.abs(m22['oof'] - Ytr) / Ytr * 100
    print(f"\n=== M22_cal standalone ===")
    print(f"  Total: {err_m22.mean():.2f}%  vcd={err_m22[:,:,0].mean():.1f}  tcd={err_m22[:,:,1].mean():.1f}  sub={err_m22[:,:,2].mean():.1f}  byp={err_m22[:,:,3].mean():.1f}")
    
    # Blend M22_cal with b09 for byproduct
    # If M22 byproduct is 8.4% vs b09's 8.0%, no gain.
    # But if they have UNCORRELATED errors, blend might help
    byp_b09 = err_b09[:, :, 3]
    byp_m22 = err_m22[:, :, 3]
    
    # Check error correlation
    corr = np.corrcoef(byp_b09.ravel(), byp_m22.ravel())[0, 1]
    print(f"  b09-M22 byproduct error correlation: {corr:.3f}")
    
    if corr < 0.7:
        # Try blending
        print("  Errors are somewhat uncorrelated — blending might help")
        best_w = 0.5
        best_e = 1e9
        for w in np.linspace(0, 1, 21):
            blend_byp = w * P_b09_oof[:, :, 3] + (1-w) * m22['oof'][:, :, 3]
            e = np.abs(blend_byp - Ytr[:, :, 3]).mean() / Ytr[:, :, 3].mean() * 100
            if e < best_e:
                best_e = e
                best_w = w
        print(f"  Best blend: {best_w:.2f}*b09 + {1-best_w:.2f}*m22 → byp={best_e:.2f}%")
    else:
        print(f"  Errors highly correlated (r={corr:.3f}) — blending won't help")

# ============================================================
# EXPERIMENT 4: Look for M22 variants that might help
# ============================================================
print("\n=== Available M22 predictions ===")
for f in sorted(os.listdir(oof_dir)):
    if 'm22' in f:
        print(f"  {f}")

print("\nDone. See results above.")
