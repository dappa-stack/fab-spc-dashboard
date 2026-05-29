"""
spc_tool.py  —  Command-Line Statistical Process Control Analyzer
=================================================================
Analyzes semiconductor fab process data and generates SPC control charts.

Charts produced:
  • Xbar-R chart  — for continuous measurements (temp, pressure, thickness, leakage)
  • C chart       — for defect counts per unit (Poisson data)

Western Electric (WE) run rules applied to every chart:
  Rule 1 — 1 point beyond 3σ control limits
  Rule 2 — 9 consecutive points same side of centerline
  Rule 3 — 6 consecutive points trending up or down
  Rule 4 — 14 consecutive points alternating up/down

Usage:
  python spc_tool.py                        # runs full analysis on fab_process_data.csv
  python spc_tool.py --file mydata.csv      # specify a different data file
  python spc_tool.py --chart temp           # analyze one measurement only
  python spc_tool.py --list                 # list available measurements
  python spc_tool.py --save                 # save charts to PNG files
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for CLI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Control chart constants (from ASTM / Montgomery) ─────────────────────────
# A2, D3, D4 factors indexed by subgroup size n=2..10
_A2 = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
_D3 = {2:0,     3:0,     4:0,     5:0,     6:0,     7:0.076, 8:0.136, 9:0.184, 10:0.223}
_D4 = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}

# ── Measurement catalog ───────────────────────────────────────────────────────
MEASUREMENTS = {
    "temp": {
        "column":    "temp_C",
        "label":     "Furnace Temperature",
        "unit":      "°C",
        "chart":     "xbar_r",
        "target":    1000.0,
        "ucl_hint":  None,   # auto-calculated
    },
    "pressure": {
        "column":    "pressure_mTorr",
        "label":     "Etch Chamber Pressure",
        "unit":      "mTorr",
        "chart":     "xbar_r",
        "target":    50.0,
        "ucl_hint":  None,
    },
    "thickness": {
        "column":    "thickness_A",
        "label":     "Oxide Thickness",
        "unit":      "Å",
        "chart":     "xbar_r",
        "target":    500.0,
        "ucl_hint":  None,
    },
    "leakage": {
        "column":    "leakage_nA",
        "label":     "Gate Leakage Current",
        "unit":      "nA",
        "chart":     "xbar_r",
        "target":    1.2,
        "ucl_hint":  None,
    },
    "defects": {
        "column":    "defects_per_wafer",
        "label":     "Defects per Wafer",
        "unit":      "count",
        "chart":     "c_chart",
        "target":    None,
        "ucl_hint":  None,
    },
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_CL   = "#1F3A6B"   # centerline (dark blue)
C_UCL  = "#C0392B"   # upper control limit (red)
C_LCL  = "#C0392B"   # lower control limit (red)
C_WARN = "#E67E22"   # warning zone (orange) — 2σ
C_DATA = "#2C3E50"   # data line
C_OOC  = "#E74C3C"   # out-of-control point
C_TARGET = "#27AE60" # target / nominal (green)
C_BG   = "#FAFAFA"
C_GRID = "#E0E0E0"


# ═══════════════════════════════════════════════════════════════════════════════
# WESTERN ELECTRIC RUN RULES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_we_rules(values, cl, ucl, lcl):
    """
    Apply Western Electric run rules to a series of values.
    Returns a dict of {rule_name: [list of violating indices]}.
    """
    n = len(values)
    sigma = (ucl - cl) / 3.0 if ucl != cl else 1e-9
    z = (np.array(values) - cl) / sigma   # standardized values

    violations = {
        "Rule 1 — Beyond 3σ":               [],
        "Rule 2 — 9 pts same side":          [],
        "Rule 3 — 6 pts trending":           [],
        "Rule 4 — 14 pts alternating":       [],
    }

    for i in range(n):
        # Rule 1: point outside ±3σ
        if abs(z[i]) > 3.0:
            violations["Rule 1 — Beyond 3σ"].append(i)

    for i in range(8, n):
        # Rule 2: 9 consecutive points on same side of CL
        window = z[i-8 : i+1]
        if all(w > 0 for w in window) or all(w < 0 for w in window):
            violations["Rule 2 — 9 pts same side"].append(i)

    for i in range(5, n):
        # Rule 3: 6 consecutive points strictly increasing or decreasing
        window = values[i-5 : i+1]
        diffs = np.diff(window)
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            violations["Rule 3 — 6 pts trending"].append(i)

    for i in range(13, n):
        # Rule 4: 14 consecutive points alternating up/down
        window = values[i-13 : i+1]
        diffs = np.diff(window)
        alternating = all(diffs[j] * diffs[j+1] < 0 for j in range(len(diffs)-1))
        if alternating:
            violations["Rule 4 — 14 pts alternating"].append(i)

    return violations


def all_ooc_indices(violations):
    """Flatten all violation indices into a sorted unique set."""
    idx = set()
    for v in violations.values():
        idx.update(v)
    return sorted(idx)


# ═══════════════════════════════════════════════════════════════════════════════
# XBAR-R CHART
# ═══════════════════════════════════════════════════════════════════════════════

def compute_xbar_r(data, col, n):
    """
    Compute subgroup means (Xbar) and ranges (R) from raw data.
    Returns DataFrames with control limits.
    """
    grouped = data.groupby("subgroup")[col]
    xbar = grouped.mean()
    R    = grouped.max() - grouped.min()
    subgroups = xbar.index.values

    A2, D3, D4 = _A2[n], _D3[n], _D4[n]

    xbar_cl  = xbar.mean()
    R_cl     = R.mean()

    xbar_ucl = xbar_cl + A2 * R_cl
    xbar_lcl = xbar_cl - A2 * R_cl
    R_ucl    = D4 * R_cl
    R_lcl    = D3 * R_cl  # 0 for n<=6

    return {
        "subgroups": subgroups,
        "xbar":  xbar.values,
        "R":     R.values,
        "xbar_cl":  xbar_cl,
        "xbar_ucl": xbar_ucl,
        "xbar_lcl": xbar_lcl,
        "R_cl":  R_cl,
        "R_ucl": R_ucl,
        "R_lcl": R_lcl,
        "sigma_est": R_cl / 2.326,   # d2 factor for n=5
    }


def plot_xbar_r(stats, meta, save_path=None):
    """Plot Xbar and R charts stacked, return (fig, xbar_violations, R_violations)."""
    sg  = stats["subgroups"]
    label = meta["label"]
    unit  = meta["unit"]
    target = meta["target"]

    fig = plt.figure(figsize=(13, 7), facecolor=C_BG)
    fig.suptitle(f"Xbar-R Control Chart — {label} ({unit})",
                 fontsize=14, fontweight="bold", color=C_CL, y=0.98)
    gs = GridSpec(2, 1, figure=fig, hspace=0.45)

    # ── Xbar chart ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C_BG)
    ax1.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")

    xbar_viol = apply_we_rules(stats["xbar"], stats["xbar_cl"],
                                stats["xbar_ucl"], stats["xbar_lcl"])
    ooc = all_ooc_indices(xbar_viol)

    ax1.plot(sg, stats["xbar"], color=C_DATA, linewidth=1.4,
             marker="o", markersize=5, zorder=3, label="Subgroup mean")

    # Highlight OOC points
    if ooc:
        ax1.scatter(sg[ooc], stats["xbar"][ooc], color=C_OOC,
                    s=80, zorder=5, label="Out of control")

    # Control limits
    for val, color, ls, lw, lbl in [
        (stats["xbar_ucl"], C_UCL,    "--", 1.5, f"UCL = {stats['xbar_ucl']:.3f}"),
        (stats["xbar_cl"],  C_CL,     "-",  1.8, f"CL  = {stats['xbar_cl']:.3f}"),
        (stats["xbar_lcl"], C_LCL,    "--", 1.5, f"LCL = {stats['xbar_lcl']:.3f}"),
    ]:
        ax1.axhline(val, color=color, linestyle=ls, linewidth=lw, label=lbl)

    # 2σ warning zones (shaded)
    sigma = (stats["xbar_ucl"] - stats["xbar_cl"]) / 3
    ax1.axhspan(stats["xbar_cl"] + 2*sigma, stats["xbar_ucl"],
                alpha=0.06, color=C_UCL, label="±2σ–3σ zone")
    ax1.axhspan(stats["xbar_lcl"], stats["xbar_cl"] - 2*sigma,
                alpha=0.06, color=C_LCL)

    # Target line (if specified)
    if target is not None:
        ax1.axhline(target, color=C_TARGET, linestyle=":", linewidth=1.2,
                    label=f"Target = {target}")

    ax1.set_ylabel(f"Subgroup Mean ({unit})", fontsize=10)
    ax1.set_title("Xbar Chart (subgroup means)", fontsize=11, pad=4)
    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.85, ncol=3)
    ax1.set_xlim(sg[0] - 0.5, sg[-1] + 0.5)
    ax1.set_xticks(sg)

    # ── R chart ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C_BG)
    ax2.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")

    R_viol = apply_we_rules(stats["R"], stats["R_cl"],
                             stats["R_ucl"], max(stats["R_lcl"], 0))
    ooc_r = all_ooc_indices(R_viol)

    ax2.plot(sg, stats["R"], color=C_DATA, linewidth=1.4,
             marker="s", markersize=5, zorder=3, label="Subgroup range")

    if ooc_r:
        ax2.scatter(sg[ooc_r], stats["R"][ooc_r], color=C_OOC,
                    s=80, zorder=5, label="Out of control")

    for val, color, ls, lw, lbl in [
        (stats["R_ucl"], C_UCL, "--", 1.5, f"UCL = {stats['R_ucl']:.3f}"),
        (stats["R_cl"],  C_CL,  "-",  1.8, f"CL  = {stats['R_cl']:.3f}"),
    ]:
        ax2.axhline(val, color=color, linestyle=ls, linewidth=lw, label=lbl)

    if stats["R_lcl"] > 0:
        ax2.axhline(stats["R_lcl"], color=C_LCL, linestyle="--", linewidth=1.5,
                    label=f"LCL = {stats['R_lcl']:.3f}")

    ax2.set_ylabel(f"Subgroup Range ({unit})", fontsize=10)
    ax2.set_xlabel("Subgroup Number", fontsize=10)
    ax2.set_title("R Chart (subgroup ranges)", fontsize=11, pad=4)
    ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.85, ncol=3)
    ax2.set_xlim(sg[0] - 0.5, sg[-1] + 0.5)
    ax2.set_xticks(sg)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Chart saved → {save_path}")

    return fig, xbar_viol, R_viol


# ═══════════════════════════════════════════════════════════════════════════════
# C CHART (defect counts)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_c_chart(data, col):
    """Compute C chart (average defects per unit, Poisson model)."""
    grouped = data.groupby("subgroup")[col]
    c_vals  = grouped.mean()          # mean defects per subgroup
    subgroups = c_vals.index.values

    c_bar = c_vals.mean()
    ucl   = c_bar + 3 * np.sqrt(c_bar)
    lcl   = max(0, c_bar - 3 * np.sqrt(c_bar))

    return {
        "subgroups": subgroups,
        "c":   c_vals.values,
        "cl":  c_bar,
        "ucl": ucl,
        "lcl": lcl,
    }


def plot_c_chart(stats, meta, save_path=None):
    """Plot C chart, return (fig, violations)."""
    sg    = stats["subgroups"]
    label = meta["label"]

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
    fig.suptitle(f"C Chart — {label}", fontsize=14, fontweight="bold",
                 color=C_CL, y=1.01)

    viol = apply_we_rules(stats["c"], stats["cl"], stats["ucl"], stats["lcl"])
    ooc  = all_ooc_indices(viol)

    ax.plot(sg, stats["c"], color=C_DATA, linewidth=1.4,
            marker="o", markersize=5, zorder=3, label="Mean defects/subgroup")

    if ooc:
        ax.scatter(sg[ooc], stats["c"][ooc], color=C_OOC,
                   s=80, zorder=5, label="Out of control")

    for val, color, ls, lw, lbl in [
        (stats["ucl"], C_UCL, "--", 1.5, f"UCL = {stats['ucl']:.3f}"),
        (stats["cl"],  C_CL,  "-",  1.8, f"CL  = {stats['cl']:.3f}"),
        (stats["lcl"], C_LCL, "--", 1.5, f"LCL = {stats['lcl']:.3f}"),
    ]:
        ax.axhline(val, color=color, linestyle=ls, linewidth=lw, label=lbl)

    sigma = np.sqrt(stats["cl"])
    ax.axhspan(stats["cl"] + 2*sigma, stats["ucl"],
               alpha=0.06, color=C_UCL)
    ax.axhspan(max(0, stats["lcl"]), max(0, stats["cl"] - 2*sigma),
               alpha=0.06, color=C_LCL)

    ax.set_ylabel("Mean Defects per Wafer", fontsize=10)
    ax.set_xlabel("Subgroup Number", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, ncol=3)
    ax.set_xlim(sg[0] - 0.5, sg[-1] + 0.5)
    ax.set_xticks(sg)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Chart saved → {save_path}")

    return fig, viol


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS CAPABILITY (Cp, Cpk)
# ═══════════════════════════════════════════════════════════════════════════════

def process_capability(stats, col, lsl, usl):
    """Compute Cp and Cpk from Xbar-R stats."""
    sigma = stats["sigma_est"]
    xbar  = stats["xbar_cl"]
    cp    = (usl - lsl) / (6 * sigma)
    cpu   = (usl - xbar) / (3 * sigma)
    cpl   = (xbar - lsl) / (3 * sigma)
    cpk   = min(cpu, cpl)
    return {"Cp": cp, "Cpk": cpk, "Cpu": cpu, "Cpl": cpl, "sigma_est": sigma}


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ═══════════════════════════════════════════════════════════════════════════════

SEPARATOR = "=" * 68

def print_header(text):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)

def print_section(text):
    print(f"\n  ── {text}")

def summarize_violations(violations, chart_name):
    total = sum(len(v) for v in violations.values())
    if total == 0:
        print(f"  ✓  {chart_name}: IN CONTROL — no violations detected.")
        return
    print(f"  ✗  {chart_name}: {total} violation(s) detected:")
    for rule, indices in violations.items():
        if indices:
            sg_nums = [i + 1 for i in indices]   # 1-indexed subgroup
            print(f"       {rule}: subgroups {sg_nums}")

def print_capability(cap, label):
    print_section(f"Process Capability — {label}")
    print(f"     σ̂  (estimated) : {cap['sigma_est']:.4f}")
    print(f"     Cp             : {cap['Cp']:.3f}  {'✓ Capable' if cap['Cp'] >= 1.33 else '✗ Not capable'}")
    print(f"     Cpk            : {cap['Cpk']:.3f}  {'✓ Centered & capable' if cap['Cpk'] >= 1.33 else '✗ Needs attention'}")
    print(f"     Cpu / Cpl      : {cap['Cpu']:.3f} / {cap['Cpl']:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

# Spec limits for each measurement (LSL, USL) — realistic fab tolerances
SPEC_LIMITS = {
    "temp":      (994.0,  1006.0),   # ±6°C on 1000°C furnace
    "pressure":  (47.0,   53.0),     # ±3 mTorr on 50 mTorr etch
    "thickness": (490.0,  510.0),    # ±10 Å on 500 Å oxide
    "leakage":   (0.0,    2.5),      # 0–2.5 nA gate leakage spec
}


def run_analysis(data, key, save=False, output_dir="."):
    meta = MEASUREMENTS[key]
    col  = meta["column"]

    print_header(f"{meta['label'].upper()} — {meta['unit']}")

    if meta["chart"] == "xbar_r":
        n = data.groupby("subgroup")[col].count().iloc[0]
        stats = compute_xbar_r(data, col, n)

        print_section("Control Limits (Xbar Chart)")
        print(f"     UCL = {stats['xbar_ucl']:.4f}  |  CL = {stats['xbar_cl']:.4f}  |  LCL = {stats['xbar_lcl']:.4f}")
        print_section("Control Limits (R Chart)")
        print(f"     UCL = {stats['R_ucl']:.4f}  |  CL = {stats['R_cl']:.4f}  |  LCL = {stats['R_lcl']:.4f}")

        save_path = os.path.join(output_dir, f"chart_{key}.png") if save else None
        _, xv, rv = plot_xbar_r(stats, meta, save_path=save_path)

        print_section("Western Electric Run Rule Results")
        summarize_violations(xv, "Xbar Chart")
        summarize_violations(rv, "R Chart")

        if key in SPEC_LIMITS:
            lsl, usl = SPEC_LIMITS[key]
            cap = process_capability(stats, col, lsl, usl)
            print_capability(cap, meta["label"])
            print(f"     Spec limits: LSL = {lsl}  |  USL = {usl}")

    elif meta["chart"] == "c_chart":
        stats = compute_c_chart(data, col)

        print_section("Control Limits (C Chart)")
        print(f"     UCL = {stats['ucl']:.4f}  |  CL = {stats['cl']:.4f}  |  LCL = {stats['lcl']:.4f}")

        save_path = os.path.join(output_dir, f"chart_{key}.png") if save else None
        _, viol = plot_c_chart(stats, meta, save_path=save_path)

        print_section("Western Electric Run Rule Results")
        summarize_violations(viol, "C Chart")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Command-line SPC analyzer for fab process data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--file",  default="fab_process_data.csv",
                        help="Path to process data CSV (default: fab_process_data.csv)")
    parser.add_argument("--chart", default="all",
                        help="Which measurement to analyze: temp | pressure | thickness | leakage | defects | all")
    parser.add_argument("--save",  action="store_true",
                        help="Save charts as PNG files")
    parser.add_argument("--outdir", default=".",
                        help="Directory to save chart PNGs (default: current dir)")
    parser.add_argument("--list",  action="store_true",
                        help="List available measurement keys and exit")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable measurement keys:")
        for k, v in MEASUREMENTS.items():
            print(f"  {k:<12} — {v['label']} ({v['unit']}, {v['chart']})")
        sys.exit(0)

    # Load data
    if not os.path.exists(args.file):
        print(f"Error: data file '{args.file}' not found.")
        print("Run generate_data.py first to create sample data.")
        sys.exit(1)

    data = pd.read_csv(args.file)
    print(f"\nLoaded {len(data)} rows from '{args.file}'")
    print(f"Subgroups: {data['subgroup'].nunique()}  |  Subgroup size: {data.groupby('subgroup').size().iloc[0]}")

    keys = list(MEASUREMENTS.keys()) if args.chart == "all" else [args.chart]

    if args.chart != "all" and args.chart not in MEASUREMENTS:
        print(f"Error: unknown chart key '{args.chart}'. Run with --list to see options.")
        sys.exit(1)

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)

    for key in keys:
        run_analysis(data, key, save=args.save, output_dir=args.outdir)

    print(SEPARATOR)
    print("  Analysis complete.")
    if args.save:
        print(f"  Charts saved to: {os.path.abspath(args.outdir)}/")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
