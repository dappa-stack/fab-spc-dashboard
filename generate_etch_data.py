"""
generate_etch_data.py — Plasma Etch Tool Sensor Data Generator
==============================================================
Generates realistic synthetic sensor data from a plasma etch tool.
Simulates normal operation and 4 fault conditions used for ML training.

Sensors simulated (based on real Lam Research / Applied Materials etch tools):
  • RF Power (W)         — main plasma power source
  • Chamber Pressure (mTorr) — process pressure
  • Gas Flow (sccm)      — etch gas (Cl2/HBr) flow rate
  • Chuck Temperature (°C)   — electrostatic chuck temp control
  • Bias Voltage (V)     — substrate bias for ion energy control
  • Reflected Power (W)  — RF power reflected back (matching network)
  • DC Bias (V)          — self-bias voltage indicator of plasma density

Fault classes:
  0 — Normal operation
  1 — Pressure drift      (slow leak or pressure controller fault)
  2 — RF instability      (matching network or generator fault)
  3 — Gas flow anomaly    (MFC fault or gas line issue)
  4 — Chuck temp excursion (cooling system or ESC fault)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ── Normal operating parameters (center values) ───────────────────────────────
NORMAL_PARAMS = {
    "rf_power_W":       {"mean": 500.0,  "std": 5.0},
    "pressure_mTorr":   {"mean": 30.0,   "std": 0.5},
    "gas_flow_sccm":    {"mean": 80.0,   "std": 1.0},
    "chuck_temp_C":     {"mean": 20.0,   "std": 0.3},
    "bias_voltage_V":   {"mean": 120.0,  "std": 2.0},
    "reflected_power_W":{"mean": 5.0,    "std": 1.0},
    "dc_bias_V":        {"mean": 200.0,  "std": 4.0},
}

SENSORS = list(NORMAL_PARAMS.keys())
N_SENSORS = len(SENSORS)

# ── Fault class definitions ───────────────────────────────────────────────────
FAULT_CLASSES = {
    0: "Normal",
    1: "Pressure Drift",
    2: "RF Instability",
    3: "Gas Flow Anomaly",
    4: "Chuck Temp Excursion",
}

N_SAMPLES_PER_CLASS = 300   # samples per fault class
WINDOW_SIZE         = 20    # samples per time window for feature extraction
N_WINDOWS_PER_SAMPLE = 1


def generate_normal(n):
    """Generate n samples of normal tool operation."""
    data = {}
    for sensor, p in NORMAL_PARAMS.items():
        data[sensor] = np.random.normal(p["mean"], p["std"], n)
    return pd.DataFrame(data)


def generate_pressure_drift(n):
    """
    Fault Class 1: Pressure Drift
    Simulates a slow chamber leak or pressure controller fault.
    Pressure rises gradually; DC bias and reflected power respond.
    """
    data = {}
    t = np.linspace(0, 1, n)   # normalized time

    for sensor, p in NORMAL_PARAMS.items():
        base = np.random.normal(p["mean"], p["std"], n)
        if sensor == "pressure_mTorr":
            # Gradual upward drift +8 mTorr over the window
            base += t * 8.0 + np.random.normal(0, 0.8, n)
        elif sensor == "dc_bias_V":
            # DC bias drops as pressure rises (plasma density changes)
            base -= t * 15.0 + np.random.normal(0, 2.0, n)
        elif sensor == "reflected_power_W":
            # Slight impedance mismatch as pressure changes
            base += t * 3.0 + np.random.normal(0, 0.5, n)
        data[sensor] = base

    return pd.DataFrame(data)


def generate_rf_instability(n):
    """
    Fault Class 2: RF Instability
    Simulates matching network fault or RF generator issue.
    RF power oscillates; reflected power spikes; DC bias fluctuates.
    """
    data = {}
    t = np.linspace(0, 4 * np.pi, n)

    for sensor, p in NORMAL_PARAMS.items():
        base = np.random.normal(p["mean"], p["std"], n)
        if sensor == "rf_power_W":
            # Oscillating RF power ±40W
            base += 40 * np.sin(t) + np.random.normal(0, 8.0, n)
        elif sensor == "reflected_power_W":
            # High reflected power with spikes
            base += 25 * np.abs(np.sin(t)) + np.random.normal(0, 3.0, n)
        elif sensor == "dc_bias_V":
            # DC bias follows RF power oscillations
            base += 30 * np.sin(t + 0.3) + np.random.normal(0, 5.0, n)
        elif sensor == "bias_voltage_V":
            base += 15 * np.sin(t) + np.random.normal(0, 3.0, n)
        data[sensor] = base

    return pd.DataFrame(data)


def generate_gas_flow_anomaly(n):
    """
    Fault Class 3: Gas Flow Anomaly
    Simulates MFC (mass flow controller) fault or gas line blockage.
    Gas flow drops suddenly; pressure drops; plasma chemistry changes.
    """
    data = {}
    fault_start = n // 3   # fault begins 1/3 into the window

    for sensor, p in NORMAL_PARAMS.items():
        base = np.random.normal(p["mean"], p["std"], n)
        fault_mask = np.zeros(n)
        fault_mask[fault_start:] = 1.0

        if sensor == "gas_flow_sccm":
            # Gas flow drops by 40% after fault
            base -= fault_mask * 32.0 + np.random.normal(0, 2.0, n)
        elif sensor == "pressure_mTorr":
            # Pressure drops with gas flow
            base -= fault_mask * 8.0 + np.random.normal(0, 0.8, n)
        elif sensor == "dc_bias_V":
            # Plasma density changes affect DC bias
            base -= fault_mask * 25.0 + np.random.normal(0, 4.0, n)
        elif sensor == "rf_power_W":
            # Control system tries to compensate
            base += fault_mask * 20.0 + np.random.normal(0, 5.0, n)
        data[sensor] = base

    return pd.DataFrame(data)


def generate_chuck_temp_excursion(n):
    """
    Fault Class 4: Chuck Temperature Excursion
    Simulates cooling system failure or ESC (electrostatic chuck) fault.
    Chuck temperature rises; affects etch rate uniformity signals.
    """
    data = {}
    t = np.linspace(0, 1, n)

    for sensor, p in NORMAL_PARAMS.items():
        base = np.random.normal(p["mean"], p["std"], n)
        if sensor == "chuck_temp_C":
            # Temperature rises exponentially (cooling loss)
            base += 15 * (np.exp(t) - 1) / (np.e - 1) + np.random.normal(0, 0.5, n)
        elif sensor == "bias_voltage_V":
            # Slight bias shift from thermal expansion effects
            base += t * 8.0 + np.random.normal(0, 1.5, n)
        elif sensor == "dc_bias_V":
            # DC bias affected by wafer temperature
            base += t * 12.0 + np.random.normal(0, 3.0, n)
        data[sensor] = base

    return pd.DataFrame(data)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(df_window):
    """
    Extract statistical features from a sensor window.
    This is how real FDC systems work — models train on features,
    not raw time-series data.

    Features per sensor:
      mean, std, min, max, range, rate of change (slope), RMS
    """
    features = {}
    for sensor in SENSORS:
        vals = df_window[sensor].values
        features[f"{sensor}_mean"]  = np.mean(vals)
        features[f"{sensor}_std"]   = np.std(vals)
        features[f"{sensor}_min"]   = np.min(vals)
        features[f"{sensor}_max"]   = np.max(vals)
        features[f"{sensor}_range"] = np.max(vals) - np.min(vals)
        features[f"{sensor}_slope"] = np.polyfit(np.arange(len(vals)), vals, 1)[0]
        features[f"{sensor}_rms"]   = np.sqrt(np.mean(vals**2))
    return features


def build_dataset():
    """
    Generate full dataset with all fault classes.
    Each sample = one WINDOW_SIZE time window with extracted features.
    Returns feature matrix X and label vector y.
    """
    generators = {
        0: generate_normal,
        1: generate_pressure_drift,
        2: generate_rf_instability,
        3: generate_gas_flow_anomaly,
        4: generate_chuck_temp_excursion,
    }

    all_features = []
    all_labels   = []
    all_raw      = []

    for fault_class, gen_fn in generators.items():
        for sample_idx in range(N_SAMPLES_PER_CLASS):
            # Generate one window of raw sensor data
            raw = gen_fn(WINDOW_SIZE)

            # Extract features from window
            feats = extract_features(raw)
            feats["fault_class"] = fault_class
            feats["fault_label"] = FAULT_CLASSES[fault_class]
            feats["sample_idx"]  = sample_idx

            all_features.append(feats)

            # Store raw sensor trace for visualization
            raw["fault_class"] = fault_class
            raw["fault_label"] = FAULT_CLASSES[fault_class]
            raw["sample_idx"]  = sample_idx
            raw["time_step"]   = np.arange(WINDOW_SIZE)
            all_raw.append(raw)

    df_features = pd.DataFrame(all_features)
    df_raw      = pd.concat(all_raw, ignore_index=True)

    return df_features, df_raw


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating plasma etch tool sensor data...")

    df_features, df_raw = build_dataset()

    # Save both datasets
    df_features.to_csv("etch_features.csv", index=False)
    df_raw.to_csv("etch_raw_sensors.csv",   index=False)

    print(f"\nFeature dataset: {df_features.shape[0]} samples × "
          f"{df_features.shape[1]-3} features")
    print(f"Raw sensor data: {df_raw.shape[0]} rows")

    print("\nClass distribution:")
    for cls, label in FAULT_CLASSES.items():
        n = (df_features["fault_class"] == cls).sum()
        print(f"  Class {cls} — {label:<25} {n} samples")

    print("\nFeature columns (first 10):")
    feat_cols = [c for c in df_features.columns
                 if c not in ["fault_class", "fault_label", "sample_idx"]]
    for col in feat_cols[:10]:
        print(f"  {col}")
    print(f"  ... and {len(feat_cols)-10} more")

    print("\nFiles saved:")
    print("  etch_features.csv   — feature matrix for ML training")
    print("  etch_raw_sensors.csv — raw sensor traces for visualization")
