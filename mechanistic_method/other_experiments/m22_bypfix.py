"""
New ODE variant: m22_bypfix — fixes the byproduct consumption kinetics.

Changes from base:
1. cons() = kc * P * S/(K_cons + S) instead of kc * P * Kg/(Kg+S)
   - Consumption requires substrate (energy source)
   - When S is low, consumption stops → byproduct stays high (matches data)
   
2. ov() uses a substrate threshold So:
   - ov = qov * (S-So)+ / (Kov + (S-So)+) / (1 + (I/KIo)^no)
   - Prevents overflow at very low substrate levels
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lab'))
import numpy as np
from ode import *
from common import *

# ============================================================
# M22: Fixed consumption + thresholded overflow
# ============================================================
class M22_bypfix(Model):
    name = 'm22_bypfix'
    desc = 'm02 + substrate-dependent byproduct consumption (S/(K+S)) + substrate-thresholded overflow'
    zset = 'z7'
    parent = 'm02_cov7'
    
    def spec(self):
        s = dict(BASE_SPEC)
        # Replace Kg with K_cons (substrate affinity for consumption)
        # Add So (substrate threshold for overflow)
        # Add Kov (overflow half-saturation - was hardcoded 0.05)
        s['K_cons'] = (0.5, 0.01, 20, 'log')
        s['So'] = (0.1, 0.001, 2.0, 'log')
        s['Kov'] = (0.2, 0.01, 5, 'log')
        return s
    
    def ov(self, p, s):
        """Overflow with substrate threshold.
        No overflow below So: cells use all glucose for growth.
        """
        Seff = np.maximum(s['S'] - p['So'], 0.0)
        return p['qov'] * Seff / (p['Kov'] + Seff) / (1 + (np.maximum(s['I'], 1e-9) / p['KIo']) ** p['no'])
    
    def cons(self, p, s):
        """Substrate-dependent byproduct consumption.
        Cells can only re-consume byproduct when glucose is available.
        """
        return p['kc'] * s['P'] * s['S'] / (p['K_cons'] + s['S'])


# ============================================================
# Quick test: compare M22 against data without refitting
# ============================================================
if __name__ == '__main__':
    bs, A, Y, tt, test = load()
    tr = ~test
    Ytr = Y[tr][:,1:,:]
    
    # We can't re-fit from scratch here, but we can do a diagnostic:
    # Run M22 with the same parameters as M01 but using the new kinetics
    # The initial values matter less than the dynamics
    
    m01_params = json.load(open('../lab/ode/m01_base.json'))
    print("M01 params:", m01_params['p'])
    
    # The key insight: regardless of refitting, switching cons() from
    # kc*P*Kg/(Kg+S) to kc*P*S/(K+S) will:
    # - REDUCE consumption at early days (when S is high but P is low)
    # - REDUCE consumption at late days (when S is low)
    # - Net effect: higher byproduct at late days (fixes underprediction)
    # - Net effect: lower consumption of early byproduct (could overshoot)
    
    # The thresholded overflow:
    # - REDUCES overflow at very low substrate
    # - Mainly affects early phase (S drops below So quickly?)
    # - Could reduce early byproduct peak
    
    print("\n=== M22 analysis complete ===")
    print("To test this properly, run:")
    print("  cd ../lab && python3 variants.py m22_bypfix")
    print("But this requires adding M22 to variants.py first.")
    print("\nThe fix targets the two identified problems:")
    print("  1. Overflow too sensitive at early stage → add So threshold")
    print("  2. Consumption too aggressive at late stage → use S/(K+S) kinetics")
