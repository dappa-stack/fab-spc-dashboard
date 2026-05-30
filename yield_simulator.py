"""
yield_simulator.py — Semiconductor Wafer Yield Map Simulator
=============================================================
Simulates wafer-level defect distribution and die yield using
industry-standard models used at Intel, TSMC, Samsung, and other fabs.

Models implemented:
  • Poisson defect distribution   — random particle contamination
  • Murphy's yield model          — industry standard die yield formula
  • Edge exclusion zone           — realistic wafer edge die removal
  • Defect clustering (optional)  — simulates localized contamination

Wafer sizes supported:
  • 200mm (8-inch)  — older fabs, ON Semi, Microchip, GlobalFoundries
  • 300mm (12-inch) — modern fabs, Intel, TSMC, Samsung, TSMC

Usage:
  python yield_simulator.py                        # default 300mm wafer
  python yield_simulator.py --wafer 200            # 200mm wafer
  python yield_simulator.py --defect-density 0.5   # set defects/cm²
  python yield_simulator.py --die-size 100         # die size in mm²
  python yield_simulator.py --cluster              # add defect cluster
  python yield_simulator.py --save                 # save wafer map PNG
  python yield_simulator.py --wafer both           # compare 200 vs 300mm
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# ── Wafer specifications ──────────────────────────────────────────────────────
WAFER_SPECS = {
    200: {
        "diameter_mm":    200,
        "radius_mm":      100,
        "edge_excl_mm":   3,      # edge exclusion zone (standard)
        "notch_mm":       1,      # flat/notch size
        "label":          "200mm (8-inch)",
        "fabs":           "ON Semi, Microchip, GlobalFoundries",
    },
    300: {
        "diameter_mm":    300,
        "radius_mm":      150,
        "edge_excl_mm":   3,
        "notch_mm":       1,
        "label":          "300mm (12-inch)",
        "fabs":           "Intel, TSMC, Samsung, TSMC Arizona",
    },
}

# ── Colors ────────────────────────────────────────────────────────────────────
C_PASS    = "#27AE60"   # green — passing die
C_FAIL    = "#E74C3C"   # red — failing die
C_EDGE    = "#95A5A6"   # gray — edge exclusion / outside wafer
C_BG      = "#1A1A2E"   # dark background
C_WAFER   = "#2C3E50"   # wafer body color
C_BLUE    = "#1F3A6B"
C_GOLD    = "#F39C12"
C_WHITE   = "#FAFAFA"

SEPARATOR = "=" * 68


# ══════════════════════════════════════════════════════════════════════════════
# WAFER GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def build_wafer_grid(wafer_mm, die_size_mm2, edge_excl_mm):
    """
    Tile the wafer with dies and classify each as:
      'pass'  — inside wafer, outside edge exclusion, no defect
      'fail'  — inside wafer, outside edge exclusion, has defect
      'edge'  — inside wafer radius but in edge exclusion zone
      'out'   — outside wafer radius
    
    Returns:
      grid      : 2D array of die status strings
      die_xy    : list of (col, row, cx, cy) for dies inside wafer
      die_info  : dict of wafer-level stats
    """
    radius_mm    = wafer_mm / 2
    active_r     = radius_mm - edge_excl_mm

    # Die dimensions (assume square dies)
    die_side_mm  = np.sqrt(die_size_mm2)

    # Grid dimensions — center the grid on wafer
    n_cols = int(np.ceil(wafer_mm / die_side_mm)) + 2
    n_rows = int(np.ceil(wafer_mm / die_side_mm)) + 2

    # Offset so grid is centered
    x_start = -(n_cols * die_side_mm) / 2
    y_start = -(n_rows * die_side_mm) / 2

    grid     = []
    die_xy   = []   # (col, row, center_x, center_y) for active dies
    edge_xy  = []

    total_dies    = 0
    active_dies   = 0
    edge_dies     = 0

    for row in range(n_rows):
        grid_row = []
        for col in range(n_cols):
            cx = x_start + col * die_side_mm + die_side_mm / 2
            cy = y_start + row * die_side_mm + die_side_mm / 2
            dist = np.sqrt(cx**2 + cy**2)

            if dist + die_side_mm / 2 > radius_mm:
                grid_row.append("out")
            elif dist + die_side_mm / 2 > active_r:
                grid_row.append("edge")
                edge_dies += 1
                edge_xy.append((col, row, cx, cy))
                total_dies += 1
            else:
                grid_row.append("active")
                active_dies += 1
                die_xy.append((col, row, cx, cy))
                total_dies += 1
        grid.append(grid_row)

    die_info = {
        "total_dies":   total_dies,
        "active_dies":  active_dies,
        "edge_dies":    edge_dies,
        "die_side_mm":  die_side_mm,
        "die_size_mm2": die_size_mm2,
        "n_cols":       n_cols,
        "n_rows":       n_rows,
        "x_start":      x_start,
        "y_start":      y_start,
    }

    return grid, die_xy, edge_xy, die_info


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT & YIELD MODELS
# ══════════════════════════════════════════════════════════════════════════════

def murphy_yield(defect_density, die_size_mm2):
    """
    Murphy's yield model — industry standard.
    
    Y = [(1 - exp(-D0 * A)) / (D0 * A)]^2
    
    Where:
      D0 = defect density (defects/cm²)
      A  = die area (cm²)
    
    Murphy's model accounts for the fact that defects are not
    perfectly random — they tend to cluster. This makes it more
    accurate than the simple Poisson model for real fab data.
    """
    A_cm2 = die_size_mm2 / 100.0   # convert mm² to cm²
    x     = defect_density * A_cm2
    if x < 1e-10:
        return 1.0
    y = ((1 - np.exp(-x)) / x) ** 2
    return float(np.clip(y, 0, 1))


def poisson_yield(defect_density, die_size_mm2):
    """
    Simple Poisson yield model.
    Y = exp(-D0 * A)
    Less accurate than Murphy's but useful for comparison.
    """
    A_cm2 = die_size_mm2 / 100.0
    return float(np.exp(-defect_density * A_cm2))


def generate_defects(wafer_mm, defect_density, add_cluster=False,
                     cluster_center=None, cluster_strength=3.0):
    """
    Generate random defect locations on the wafer using Poisson statistics.
    Optionally adds a Gaussian defect cluster (simulates localized contamination).
    
    Returns array of (x, y) defect coordinates in mm.
    """
    radius_mm = wafer_mm / 2
    wafer_area_cm2 = np.pi * (radius_mm / 10) ** 2   # mm → cm
    n_defects = np.random.poisson(defect_density * wafer_area_cm2)

    defects = []
    while len(defects) < n_defects:
        # Rejection sampling inside circle
        x = np.random.uniform(-radius_mm, radius_mm)
        y = np.random.uniform(-radius_mm, radius_mm)
        if np.sqrt(x**2 + y**2) <= radius_mm:
            defects.append((x, y))

    # Optional: add a Gaussian defect cluster
    if add_cluster:
        if cluster_center is None:
            # Random cluster location inside active area
            angle = np.random.uniform(0, 2 * np.pi)
            r     = np.random.uniform(0, radius_mm * 0.6)
            cluster_center = (r * np.cos(angle), r * np.sin(angle))

        cx, cy   = cluster_center
        n_extra  = int(defect_density * np.pi * (radius_mm / 10)**2 * cluster_strength)
        sigma    = wafer_mm * 0.08   # cluster spread ~8% of wafer diameter

        for _ in range(n_extra):
            x = np.random.normal(cx, sigma)
            y = np.random.normal(cy, sigma)
            if np.sqrt(x**2 + y**2) <= radius_mm:
                defects.append((x, y))

    return np.array(defects) if defects else np.zeros((0, 2))


def assign_die_results(die_xy, defects, die_side_mm):
    """
    Check each active die for defects. A die fails if any defect
    lands within its boundaries.
    
    Returns dict: {(col, row): 'pass'|'fail'}
    """
    results = {}
    half = die_side_mm / 2

    for col, row, cx, cy in die_xy:
        xmin, xmax = cx - half, cx + half
        ymin, ymax = cy - half, cy + half

        failed = False
        if len(defects) > 0:
            in_die = ((defects[:, 0] >= xmin) & (defects[:, 0] < xmax) &
                      (defects[:, 1] >= ymin) & (defects[:, 1] < ymax))
            failed = in_die.any()

        results[(col, row)] = "fail" if failed else "pass"

    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_wafer_map(grid, die_xy, edge_xy, die_results, defects,
                   die_info, wafer_mm, params, save_path=None):
    """Plot a single wafer yield map."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    ax.set_aspect("equal")

    radius_mm  = wafer_mm / 2
    die_side   = die_info["die_side_mm"]
    x_start    = die_info["x_start"]
    y_start    = die_info["y_start"]

    # Draw wafer circle background
    wafer_circle = plt.Circle((0, 0), radius_mm, color=C_WAFER,
                               zorder=1, linewidth=2,
                               edgecolor="#4A5568")
    ax.add_patch(wafer_circle)

    # Draw edge exclusion ring
    edge_excl = params.get("edge_excl_mm", 3)
    excl_circle = plt.Circle((0, 0), radius_mm - edge_excl,
                              fill=False, edgecolor="#F39C12",
                              linewidth=1, linestyle="--",
                              zorder=2, alpha=0.5)
    ax.add_patch(excl_circle)

    # Draw edge dies (gray)
    for col, row, cx, cy in edge_xy:
        rect = patches.Rectangle(
            (cx - die_side/2, cy - die_side/2),
            die_side, die_side,
            linewidth=0.3, edgecolor="#4A5568",
            facecolor=C_EDGE, alpha=0.5, zorder=3
        )
        ax.add_patch(rect)

    # Draw active dies (pass/fail)
    pass_count = 0
    fail_count = 0
    for col, row, cx, cy in die_xy:
        status = die_results.get((col, row), "pass")
        color  = C_PASS if status == "pass" else C_FAIL
        if status == "pass":
            pass_count += 1
        else:
            fail_count += 1

        rect = patches.Rectangle(
            (cx - die_side/2, cy - die_side/2),
            die_side, die_side,
            linewidth=0.3, edgecolor="#1A1A2E",
            facecolor=color, alpha=0.85, zorder=4
        )
        ax.add_patch(rect)

    # Draw defect locations
    if len(defects) > 0:
        ax.scatter(defects[:, 0], defects[:, 1],
                   c=C_GOLD, s=8, alpha=0.6,
                   zorder=5, label=f"Defects ({len(defects)})")

    # Draw wafer outline
    wafer_outline = plt.Circle((0, 0), radius_mm, fill=False,
                                edgecolor=C_WHITE, linewidth=2, zorder=6)
    ax.add_patch(wafer_outline)

    # Notch indicator at bottom
    notch = plt.Circle((0, -radius_mm), 2.5, color=C_BG, zorder=7)
    ax.add_patch(notch)

    # Yield calculation
    actual_yield = pass_count / (pass_count + fail_count) if (pass_count + fail_count) > 0 else 0
    murphy_y     = murphy_yield(params["defect_density"], params["die_size_mm2"])
    poisson_y    = poisson_yield(params["defect_density"], params["die_size_mm2"])

    # Title and labels
    wafer_spec = WAFER_SPECS[wafer_mm]
    ax.set_title(
        f"Wafer Yield Map — {wafer_spec['label']}\n"
        f"Die: {params['die_size_mm2']}mm²  |  "
        f"D₀: {params['defect_density']} def/cm²  |  "
        f"Simulated Yield: {actual_yield:.1%}",
        color=C_WHITE, fontsize=12, fontweight="bold", pad=15
    )

    # Stats box
    stats_text = (
        f"Active dies:      {pass_count + fail_count}\n"
        f"Passing dies:     {pass_count}  ({actual_yield:.1%})\n"
        f"Failing dies:     {fail_count}  ({1-actual_yield:.1%})\n"
        f"Total defects:    {len(defects)}\n"
        f"Murphy yield:     {murphy_y:.1%}\n"
        f"Poisson yield:    {poisson_y:.1%}\n"
        f"Edge excl. dies:  {len(edge_xy)}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, color=C_WHITE, verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0D1117",
                      alpha=0.85, edgecolor="#4A5568"))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_PASS,  label=f"Pass ({pass_count})"),
        Patch(facecolor=C_FAIL,  label=f"Fail ({fail_count})"),
        Patch(facecolor=C_EDGE,  label=f"Edge excl. ({len(edge_xy)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor="#0D1117", edgecolor="#4A5568",
              labelcolor=C_WHITE, fontsize=9)

    ax.set_xlim(-radius_mm * 1.15, radius_mm * 1.15)
    ax.set_ylim(-radius_mm * 1.15, radius_mm * 1.15)
    ax.set_xlabel("X position (mm)", color=C_WHITE, fontsize=10)
    ax.set_ylabel("Y position (mm)", color=C_WHITE, fontsize=10)
    ax.tick_params(colors=C_WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A5568")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Chart saved → {save_path}")

    return fig, actual_yield, pass_count, fail_count


def plot_comparison(results_200, results_300, save_path=None):
    """Side-by-side comparison of 200mm vs 300mm wafer maps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=C_BG)
    fig.suptitle("Wafer Yield Comparison — 200mm vs 300mm",
                 color=C_WHITE, fontsize=14, fontweight="bold")

    for ax, res, wafer_mm in zip(axes, [results_200, results_300], [200, 300]):
        ax.set_facecolor(C_BG)
        ax.set_aspect("equal")

        radius_mm = wafer_mm / 2
        die_side  = res["die_info"]["die_side_mm"]

        wafer_circle = plt.Circle((0, 0), radius_mm, color=C_WAFER,
                                   zorder=1, edgecolor="#4A5568", linewidth=2)
        ax.add_patch(wafer_circle)

        for col, row, cx, cy in res["edge_xy"]:
            rect = patches.Rectangle(
                (cx - die_side/2, cy - die_side/2), die_side, die_side,
                linewidth=0.3, edgecolor="#4A5568",
                facecolor=C_EDGE, alpha=0.5, zorder=3)
            ax.add_patch(rect)

        pass_c, fail_c = 0, 0
        for col, row, cx, cy in res["die_xy"]:
            status = res["die_results"].get((col, row), "pass")
            color  = C_PASS if status == "pass" else C_FAIL
            if status == "pass": pass_c += 1
            else: fail_c += 1
            rect = patches.Rectangle(
                (cx - die_side/2, cy - die_side/2), die_side, die_side,
                linewidth=0.3, edgecolor="#1A1A2E",
                facecolor=color, alpha=0.85, zorder=4)
            ax.add_patch(rect)

        if len(res["defects"]) > 0:
            ax.scatter(res["defects"][:, 0], res["defects"][:, 1],
                       c=C_GOLD, s=6, alpha=0.5, zorder=5)

        wafer_outline = plt.Circle((0, 0), radius_mm, fill=False,
                                    edgecolor=C_WHITE, linewidth=2, zorder=6)
        ax.add_patch(wafer_outline)

        actual_yield = pass_c / (pass_c + fail_c) if (pass_c + fail_c) > 0 else 0
        murphy_y     = murphy_yield(res["params"]["defect_density"],
                                    res["params"]["die_size_mm2"])

        spec = WAFER_SPECS[wafer_mm]
        ax.set_title(
            f"{spec['label']}\n"
            f"Dies: {pass_c + fail_c}  |  Yield: {actual_yield:.1%}  |  Murphy: {murphy_y:.1%}",
            color=C_WHITE, fontsize=11, fontweight="bold"
        )
        ax.set_xlim(-radius_mm*1.1, radius_mm*1.1)
        ax.set_ylim(-radius_mm*1.1, radius_mm*1.1)
        ax.set_xlabel("X (mm)", color=C_WHITE, fontsize=9)
        ax.set_ylabel("Y (mm)", color=C_WHITE, fontsize=9)
        ax.tick_params(colors=C_WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#4A5568")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
        print(f"   Comparison chart saved → {save_path}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_header(text):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(SEPARATOR)

def print_section(text):
    print(f"\n  ── {text}")

def run_simulation(wafer_mm, defect_density, die_size_mm2,
                   add_cluster=False, seed=None):
    """Run a complete wafer simulation and return all results."""
    if seed is not None:
        np.random.seed(seed)

    spec     = WAFER_SPECS[wafer_mm]
    edge_excl = spec["edge_excl_mm"]

    params = {
        "wafer_mm":       wafer_mm,
        "defect_density": defect_density,
        "die_size_mm2":   die_size_mm2,
        "edge_excl_mm":   edge_excl,
        "add_cluster":    add_cluster,
    }

    # Build grid
    grid, die_xy, edge_xy, die_info = build_wafer_grid(
        wafer_mm, die_size_mm2, edge_excl)

    # Generate defects
    defects = generate_defects(wafer_mm, defect_density,
                                add_cluster=add_cluster)

    # Assign pass/fail
    die_results = assign_die_results(die_xy, defects, die_info["die_side_mm"])

    # Yield calculations
    pass_count  = sum(1 for v in die_results.values() if v == "pass")
    fail_count  = sum(1 for v in die_results.values() if v == "fail")
    actual_yield = pass_count / (pass_count + fail_count) if (pass_count + fail_count) > 0 else 0
    murphy_y    = murphy_yield(defect_density, die_size_mm2)
    poisson_y   = poisson_yield(defect_density, die_size_mm2)

    wafer_area_cm2 = np.pi * (wafer_mm / 20) ** 2
    dies_per_wafer  = pass_count + fail_count

    return {
        "grid":        grid,
        "die_xy":      die_xy,
        "edge_xy":     edge_xy,
        "die_info":    die_info,
        "defects":     defects,
        "die_results": die_results,
        "pass_count":  pass_count,
        "fail_count":  fail_count,
        "actual_yield":  actual_yield,
        "murphy_yield":  murphy_y,
        "poisson_yield": poisson_y,
        "wafer_area_cm2": wafer_area_cm2,
        "dies_per_wafer": dies_per_wafer,
        "params":      params,
        "spec":        spec,
    }


def print_report(res):
    spec     = res["spec"]
    params   = res["params"]
    die_info = res["die_info"]

    print_header(f"WAFER YIELD REPORT — {spec['label']}")
    print(f"  Target fabs: {spec['fabs']}")

    print_section("Wafer Parameters")
    print(f"     Wafer diameter   : {params['wafer_mm']} mm")
    print(f"     Wafer area       : {res['wafer_area_cm2']:.1f} cm²")
    print(f"     Edge exclusion   : {params['edge_excl_mm']} mm")
    print(f"     Die size         : {params['die_size_mm2']} mm²  "
          f"({die_info['die_side_mm']:.1f} x {die_info['die_side_mm']:.1f} mm)")
    print(f"     Defect density   : {params['defect_density']} defects/cm²")
    print(f"     Defect cluster   : {'Yes' if params['add_cluster'] else 'No'}")

    print_section("Die Count")
    print(f"     Active dies      : {res['dies_per_wafer']}")
    print(f"     Edge excl. dies  : {len(res['edge_xy'])}")
    print(f"     Total defects    : {len(res['defects'])}")

    print_section("Yield Analysis")
    print(f"     Simulated yield  : {res['actual_yield']:.2%}  "
          f"({res['pass_count']} pass / {res['fail_count']} fail)")
    print(f"     Murphy's model   : {res['murphy_yield']:.2%}  ← industry standard")
    print(f"     Poisson model    : {res['poisson_yield']:.2%}  ← simple estimate")
    print(f"     Good dies/wafer  : {res['pass_count']}")

    # Yield interpretation
    print_section("Yield Interpretation")
    y = res['actual_yield']
    if y >= 0.90:
        print("     ✓ EXCELLENT — >90% yield. Mature process, ready for HVM.")
    elif y >= 0.75:
        print("     ✓ GOOD — 75–90% yield. Typical for ramping process nodes.")
    elif y >= 0.50:
        print("     ⚠ MODERATE — 50–75% yield. Process needs optimization.")
    elif y >= 0.25:
        print("     ✗ LOW — 25–50% yield. Significant defect issues present.")
    else:
        print("     ✗ CRITICAL — <25% yield. Major process excursion.")

    print_section("Murphy's Model Explained")
    A = params["die_size_mm2"] / 100
    D = params["defect_density"]
    print(f"     Y = [(1 - e^(-D₀×A)) / (D₀×A)]²")
    print(f"       = [(1 - e^(-{D}×{A:.3f})) / ({D}×{A:.3f})]²")
    print(f"       = {res['murphy_yield']:.4f}  ({res['murphy_yield']:.2%})")
    print(f"")
    print(f"     Key insight: larger dies = lower yield at same defect density.")
    print(f"     This is why chip designers minimize die size — it directly")
    print(f"     impacts cost per good die and fab profitability.")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Semiconductor wafer yield map simulator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--wafer",          default="300",
                        help="Wafer size: 200 | 300 | both (default: 300)")
    parser.add_argument("--defect-density", type=float, default=0.3,
                        help="Defect density in defects/cm² (default: 0.3)")
    parser.add_argument("--die-size",       type=float, default=100.0,
                        help="Die size in mm² (default: 100)")
    parser.add_argument("--cluster",        action="store_true",
                        help="Add a defect cluster (simulates contamination event)")
    parser.add_argument("--save",           action="store_true",
                        help="Save wafer map PNG files")
    parser.add_argument("--outdir",         default=".",
                        help="Output directory for PNG files (default: .)")
    parser.add_argument("--seed",           type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if args.save:
        os.makedirs(args.outdir, exist_ok=True)

    wafer_arg = args.wafer.lower()
    wafer_sizes = []
    if wafer_arg == "both":
        wafer_sizes = [200, 300]
    elif wafer_arg in ("200", "300"):
        wafer_sizes = [int(wafer_arg)]
    else:
        print(f"Error: --wafer must be 200, 300, or both. Got '{args.wafer}'")
        sys.exit(1)

    results = {}
    for wafer_mm in wafer_sizes:
        res = run_simulation(
            wafer_mm       = wafer_mm,
            defect_density = args.defect_density,
            die_size_mm2   = args.die_size,
            add_cluster    = args.cluster,
            seed           = args.seed
        )
        results[wafer_mm] = res
        print_report(res)

        if args.save:
            save_path = os.path.join(args.outdir, f"wafer_map_{wafer_mm}mm.png")
            plot_wafer_map(
                res["grid"], res["die_xy"], res["edge_xy"],
                res["die_results"], res["defects"],
                res["die_info"], wafer_mm, res["params"],
                save_path=save_path
            )

    # Comparison chart if both sizes
    if wafer_arg == "both" and args.save:
        comp_path = os.path.join(args.outdir, "wafer_comparison.png")
        plot_comparison(results[200], results[300], save_path=comp_path)

    if not args.save:
        print(f"\n  Tip: run with --save to generate wafer map PNG files.")

    print(SEPARATOR)
    print("  Simulation complete.")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
