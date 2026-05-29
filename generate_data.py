"""
generate_data.py
Generates realistic synthetic semiconductor fab process data.
Simulates 5 measurement types across 25 subgroups of size 5.
Intentionally injects out-of-control signals for SPC detection practice.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SUBGROUPS = 25
SUBGROUP_SIZE = 5
N = N_SUBGROUPS * SUBGROUP_SIZE  # 125 total observations

subgroup_ids = np.repeat(np.arange(1, N_SUBGROUPS + 1), SUBGROUP_SIZE)
sample_ids   = np.tile(np.arange(1, SUBGROUP_SIZE + 1), N_SUBGROUPS)

# ── 1. Temperature (°C) — furnace oxidation process, target 1000°C ──────────
# Inject a mean shift at subgroup 18+ (process drift)
temp_base = np.random.normal(loc=1000.0, scale=1.5, size=N)
temp_base[17 * SUBGROUP_SIZE:] += 4.0   # +4°C shift starting subgroup 18
temperature = np.round(temp_base, 2)

# ── 2. Pressure (mTorr) — etch chamber, target 50 mTorr ─────────────────────
# Inject increased variability at subgroups 10–14 (unstable chuck)
pressure_base = np.random.normal(loc=50.0, scale=0.8, size=N)
pressure_base[9*SUBGROUP_SIZE : 14*SUBGROUP_SIZE] += np.random.normal(0, 2.5, 5*SUBGROUP_SIZE)
pressure = np.round(pressure_base, 2)

# ── 3. Oxide thickness (Å) — thermal oxidation, target 500 Å ────────────────
# Normal, in-control process — good baseline for students to see clean charts
thickness_base = np.random.normal(loc=500.0, scale=3.0, size=N)
thickness = np.round(thickness_base, 1)

# ── 4. Leakage current (nA) — MOS capacitor gate leakage, target <2 nA ──────
# Inject one outlier subgroup (subgroup 22) — single bad wafer lot
leak_base = np.abs(np.random.normal(loc=1.2, scale=0.3, size=N))
leak_base[21*SUBGROUP_SIZE : 22*SUBGROUP_SIZE] += 3.5   # spike in subgroup 22
leakage_current = np.round(leak_base, 3)

# ── 5. Defect count (per wafer) — visual inspection, Poisson process ─────────
# Inject upward trend from subgroup 15 onward (gradual contamination)
defect_lambda = np.where(
    subgroup_ids < 15,
    2.0,
    2.0 + (subgroup_ids - 14) * 0.4   # linear drift upward
)
defect_counts = np.random.poisson(lam=defect_lambda)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "subgroup":         subgroup_ids,
    "sample":           sample_ids,
    "temp_C":           temperature,
    "pressure_mTorr":   pressure,
    "thickness_A":      thickness,
    "leakage_nA":       leakage_current,
    "defects_per_wafer": defect_counts
})

df.to_csv("fab_process_data.csv", index=False)
print(f"Generated {len(df)} rows across {N_SUBGROUPS} subgroups.")
print(df.head(10).to_string(index=False))
print("\nColumn stats:")
print(df.describe().round(3).to_string())
