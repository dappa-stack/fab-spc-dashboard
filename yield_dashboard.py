"""
yield_dashboard.py — Interactive Wafer Yield Map Dashboard
===========================================================
Run with: streamlit run yield_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from yield_simulator import (
    run_simulation, murphy_yield, poisson_yield,
    build_wafer_grid, generate_defects, assign_die_results,
    WAFER_SPECS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Yield Dashboard",
    page_icon="🔬",
    layout="wide"
)

# ── Colors ────────────────────────────────────────────────────────────────────
C_PASS  = "#27AE60"
C_FAIL  = "#E74C3C"
C_EDGE  = "#7F8C8D"
C_GOLD  = "#F39C12"
C_BLUE  = "#1F3A6B"
C_BG    = "#0E1117"


# ── Plotly wafer map builder ──────────────────────────────────────────────────
def build_plotly_wafer(res, show_defects=True, show_edge=True):
    """Build an interactive Plotly wafer map from simulation results."""
    die_side   = res["die_info"]["die_side_mm"]
    half       = die_side / 2
    wafer_mm   = res["params"]["wafer_mm"]
    radius_mm  = wafer_mm / 2

    shapes = []   # wafer circle + die rectangles
    traces = []   # scatter traces for hover

    pass_x, pass_y, pass_text = [], [], []
    fail_x, fail_y, fail_text = [], [], []
    edge_x, edge_y, edge_text = [], [], []

    # Active dies
    for col, row, cx, cy in res["die_xy"]:
        status = res["die_results"].get((col, row), "pass")
        label  = (f"Die ({col},{row})<br>"
                  f"Center: ({cx:.1f}, {cy:.1f}) mm<br>"
                  f"Status: {'✓ PASS' if status == 'pass' else '✗ FAIL'}")
        if status == "pass":
            pass_x.append(cx); pass_y.append(cy); pass_text.append(label)
        else:
            fail_x.append(cx); fail_y.append(cy); fail_text.append(label)

        shapes.append(dict(
            type="rect",
            x0=cx-half, y0=cy-half, x1=cx+half, y1=cy+half,
            fillcolor=C_PASS if status == "pass" else C_FAIL,
            opacity=0.85,
            line=dict(color="#0E1117", width=0.5),
            layer="below"
        ))

    # Edge exclusion dies
    if show_edge:
        for col, row, cx, cy in res["edge_xy"]:
            label = (f"Die ({col},{row})<br>"
                     f"Center: ({cx:.1f}, {cy:.1f}) mm<br>"
                     f"Status: Edge Exclusion")
            edge_x.append(cx); edge_y.append(cy); edge_text.append(label)
            shapes.append(dict(
                type="rect",
                x0=cx-half, y0=cy-half, x1=cx+half, y1=cy+half,
                fillcolor=C_EDGE, opacity=0.4,
                line=dict(color="#0E1117", width=0.5),
                layer="below"
            ))

    # Wafer circle outline
    shapes.append(dict(
        type="circle",
        x0=-radius_mm, y0=-radius_mm,
        x1=radius_mm,  y1=radius_mm,
        fillcolor="rgba(44,62,80,0.3)",
        line=dict(color="white", width=2),
        layer="below"
    ))

    # Edge exclusion ring
    excl_r = radius_mm - res["params"]["edge_excl_mm"]
    shapes.append(dict(
        type="circle",
        x0=-excl_r, y0=-excl_r, x1=excl_r, y1=excl_r,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color=C_GOLD, width=1, dash="dash"),
        layer="above"
    ))

    fig = go.Figure()

    # Pass dies (invisible scatter for hover)
    fig.add_trace(go.Scatter(
        x=pass_x, y=pass_y, mode="markers",
        name=f"Pass ({len(pass_x)})",
        marker=dict(size=die_side*0.6, color=C_PASS,
                    symbol="square", opacity=0),
        hovertemplate="%{text}<extra></extra>",
        text=pass_text
    ))

    # Fail dies
    fig.add_trace(go.Scatter(
        x=fail_x, y=fail_y, mode="markers",
        name=f"Fail ({len(fail_x)})",
        marker=dict(size=die_side*0.6, color=C_FAIL,
                    symbol="square", opacity=0),
        hovertemplate="%{text}<extra></extra>",
        text=fail_text
    ))

    # Edge dies
    if show_edge and edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="markers",
            name=f"Edge excl. ({len(edge_x)})",
            marker=dict(size=die_side*0.6, color=C_EDGE,
                        symbol="square", opacity=0),
            hovertemplate="%{text}<extra></extra>",
            text=edge_text
        ))

    # Defect scatter
    if show_defects and len(res["defects"]) > 0:
        fig.add_trace(go.Scatter(
            x=res["defects"][:, 0],
            y=res["defects"][:, 1],
            mode="markers",
            name=f"Defects ({len(res['defects'])})",
            marker=dict(size=4, color=C_GOLD, opacity=0.7,
                        symbol="circle"),
            hovertemplate="Defect at (%{x:.1f}, %{y:.1f}) mm<extra></extra>"
        ))

    spec = WAFER_SPECS[res["params"]["wafer_mm"]]
    actual_yield = res["actual_yield"]

    fig.update_layout(
        shapes=shapes,
        height=550,
        template="plotly_dark",
        title=dict(
            text=(f"{spec['label']} Wafer Map  |  "
                  f"Yield: {actual_yield:.1%}  |  "
                  f"Murphy: {res['murphy_yield']:.1%}  |  "
                  f"Good dies: {res['pass_count']}"),
            font=dict(size=13, color="white"), x=0.5
        ),
        xaxis=dict(title="X position (mm)", scaleanchor="y",
                   range=[-radius_mm*1.15, radius_mm*1.15],
                   gridcolor="#2C3E50"),
        yaxis=dict(title="Y position (mm)",
                   range=[-radius_mm*1.15, radius_mm*1.15],
                   gridcolor="#2C3E50"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11)),
        paper_bgcolor=C_BG,
        plot_bgcolor="#0D1117",
        hovermode="closest",
        margin=dict(t=80, b=40, l=40, r=40)
    )

    return fig


def build_yield_vs_density_chart(die_size_mm2):
    """Line chart showing yield vs defect density for Murphy and Poisson."""
    densities = np.linspace(0.01, 2.0, 200)
    murphy_y  = [murphy_yield(d, die_size_mm2) * 100 for d in densities]
    poisson_y = [poisson_yield(d, die_size_mm2) * 100 for d in densities]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=densities, y=murphy_y,
        name="Murphy's Model", line=dict(color=C_PASS, width=2.5),
        hovertemplate="D₀: %{x:.3f} def/cm²<br>Murphy yield: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=densities, y=poisson_y,
        name="Poisson Model", line=dict(color=C_GOLD, width=2, dash="dash"),
        hovertemplate="D₀: %{x:.3f} def/cm²<br>Poisson yield: %{y:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        height=300, template="plotly_dark",
        title=dict(text=f"Yield vs Defect Density  (Die size: {die_size_mm2} mm²)",
                   font=dict(size=12), x=0.5),
        xaxis=dict(title="Defect Density (defects/cm²)", gridcolor="#2C3E50"),
        yaxis=dict(title="Yield (%)", range=[0, 105], gridcolor="#2C3E50"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor=C_BG, plot_bgcolor="#0D1117",
        margin=dict(t=60, b=40, l=50, r=20)
    )
    return fig


def build_yield_vs_diesize_chart(defect_density):
    """Line chart showing how yield drops as die size increases."""
    die_sizes = np.linspace(10, 500, 200)
    murphy_y  = [murphy_yield(defect_density, d) * 100 for d in die_sizes]
    poisson_y = [poisson_yield(defect_density, d) * 100 for d in die_sizes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=die_sizes, y=murphy_y,
        name="Murphy's Model", line=dict(color=C_PASS, width=2.5),
        hovertemplate="Die: %{x:.0f} mm²<br>Murphy yield: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=die_sizes, y=poisson_y,
        name="Poisson Model", line=dict(color=C_GOLD, width=2, dash="dash"),
        hovertemplate="Die: %{x:.0f} mm²<br>Poisson yield: %{y:.1f}%<extra></extra>"
    ))
    # Mark common die sizes
    common_dies = {"DRAM (50mm²)": 50, "Mid (100mm²)": 100,
                   "GPU (400mm²)": 400, "CPU (200mm²)": 200}
    for label, size in common_dies.items():
        y_val = murphy_yield(defect_density, size) * 100
        fig.add_vline(x=size, line_dash="dot", line_color="#4A5568",
                      annotation_text=label, annotation_font_size=9,
                      annotation_position="top right")

    fig.update_layout(
        height=300, template="plotly_dark",
        title=dict(text=f"Yield vs Die Size  (D₀: {defect_density} def/cm²)",
                   font=dict(size=12), x=0.5),
        xaxis=dict(title="Die Size (mm²)", gridcolor="#2C3E50"),
        yaxis=dict(title="Yield (%)", range=[0, 105], gridcolor="#2C3E50"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor=C_BG, plot_bgcolor="#0D1117",
        margin=dict(t=60, b=40, l=50, r=20)
    )
    return fig


def build_cpk_gauge(actual_yield, murphy_y):
    """Yield gauge showing actual vs Murphy target."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=actual_yield * 100,
        title={"text": "Simulated Yield (%)", "font": {"size": 16}},
        delta={"reference": murphy_y * 100,
               "increasing": {"color": C_PASS},
               "decreasing": {"color": C_FAIL},
               "suffix": "% vs Murphy"},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "white"},
            "bar":  {"color": C_BLUE},
            "steps": [
                {"range": [0,  25],  "color": "rgba(231,76,60,0.3)"},
                {"range": [25, 50],  "color": "rgba(231,76,60,0.15)"},
                {"range": [50, 75],  "color": "rgba(243,156,18,0.2)"},
                {"range": [75, 90],  "color": "rgba(39,174,96,0.15)"},
                {"range": [90, 100], "color": "rgba(39,174,96,0.25)"},
            ],
            "threshold": {
                "line": {"color": C_GOLD, "width": 3},
                "thickness": 0.75,
                "value": murphy_y * 100
            }
        }
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor=C_BG,
        font={"color": "white"},
        margin=dict(t=40, b=20, l=30, r=30)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔬 Semiconductor Wafer Yield Dashboard")
st.markdown("Interactive wafer yield simulation using Murphy's model and Poisson defect statistics. "
            "Adjust parameters in the sidebar to see real-time wafer map updates.")
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Parameters")

    wafer_choice = st.selectbox(
        "Wafer size",
        ["300mm — Intel, TSMC, Samsung", "200mm — ON Semi, Microchip",
         "Compare both"],
        index=0
    )

    st.divider()
    defect_density = st.slider(
        "Defect density (def/cm²)",
        min_value=0.01, max_value=2.0, value=0.3, step=0.01,
        help="Number of random defects per cm² of wafer area. "
             "Industry target: <0.1 for mature nodes."
    )

    die_size = st.slider(
        "Die size (mm²)",
        min_value=10, max_value=500, value=100, step=5,
        help="Area of each individual die. Larger dies = lower yield "
             "at same defect density."
    )

    add_cluster = st.toggle(
        "Add defect cluster",
        value=False,
        help="Simulates a localized contamination event "
             "(e.g. particle shower, chemical spill)."
    )

    show_defects = st.toggle("Show defect locations", value=True)
    show_edge    = st.toggle("Show edge exclusion dies", value=True)

    st.divider()
    seed = st.number_input("Random seed", value=42, step=1,
                           help="Change seed to generate a different wafer.")

    st.divider()
    st.markdown("**Murphy's Model**")
    st.latex(r"Y = \left[\frac{1 - e^{-D_0 A}}{D_0 A}\right]^2")
    st.markdown("D₀ = defect density (def/cm²)  \nA = die area (cm²)")

# ── Determine wafer sizes to simulate ────────────────────────────────────────
if "300mm" in wafer_choice:
    wafer_sizes = [300]
elif "200mm" in wafer_choice:
    wafer_sizes = [200]
else:
    wafer_sizes = [200, 300]

# ── Run simulations ───────────────────────────────────────────────────────────
results = {}
for wm in wafer_sizes:
    results[wm] = run_simulation(
        wafer_mm       = wm,
        defect_density = defect_density,
        die_size_mm2   = die_size,
        add_cluster    = add_cluster,
        seed           = int(seed)
    )

# ── Summary metrics ───────────────────────────────────────────────────────────
res_primary = results[wafer_sizes[0]]

murphy_y  = murphy_yield(defect_density, die_size)
poisson_y = poisson_yield(defect_density, die_size)

if len(wafer_sizes) == 1:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Wafer Size",     f"{wafer_sizes[0]}mm")
    c2.metric("Active Dies",    res_primary["dies_per_wafer"])
    c3.metric("Good Dies",      res_primary["pass_count"])
    c4.metric("Simulated Yield", f"{res_primary['actual_yield']:.1%}")
    c5.metric("Murphy Yield",   f"{murphy_y:.1%}")
    c6.metric("Total Defects",  len(res_primary["defects"]))
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("200mm Good Dies",  results[200]["pass_count"],
              delta=f"{results[200]['actual_yield']:.1%} yield")
    c2.metric("300mm Good Dies",  results[300]["pass_count"],
              delta=f"{results[300]['actual_yield']:.1%} yield")
    c3.metric("Murphy Yield",     f"{murphy_y:.1%}")
    c4.metric("Die Advantage (300 vs 200)",
              f"{results[300]['pass_count'] - results[200]['pass_count']}x more good dies")

st.divider()

# ── Wafer maps ────────────────────────────────────────────────────────────────
if len(wafer_sizes) == 1:
    col_map, col_gauge = st.columns([3, 1])
    with col_map:
        fig_map = build_plotly_wafer(res_primary, show_defects, show_edge)
        st.plotly_chart(fig_map, use_container_width=True)
    with col_gauge:
        st.subheader("Yield Gauge")
        fig_gauge = build_cpk_gauge(res_primary["actual_yield"], murphy_y)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("**Yield Benchmarks**")
        y = res_primary["actual_yield"]
        if y >= 0.90:
            st.success("✓ Excellent — >90%\nMature HVM process")
        elif y >= 0.75:
            st.success("✓ Good — 75–90%\nRamping process")
        elif y >= 0.50:
            st.warning("⚠ Moderate — 50–75%\nNeeds optimization")
        else:
            st.error("✗ Critical — <50%\nMajor excursion")

        st.markdown("**Murphy vs Simulated**")
        gap = abs(res_primary["actual_yield"] - murphy_y)
        if gap > 0.05:
            st.error(f"Gap: {gap:.1%} — investigate\nabnormal defect source")
        else:
            st.success(f"Gap: {gap:.1%} — within\nnormal variation")

else:
    col1, col2 = st.columns(2)
    with col1:
        fig_200 = build_plotly_wafer(results[200], show_defects, show_edge)
        st.plotly_chart(fig_200, use_container_width=True)
    with col2:
        fig_300 = build_plotly_wafer(results[300], show_defects, show_edge)
        st.plotly_chart(fig_300, use_container_width=True)

    st.info(f"**Key insight:** Same defect density ({defect_density} def/cm²) and die size ({die_size} mm²) "
            f"gives {results[200]['actual_yield']:.1%} yield on 200mm vs "
            f"{results[300]['actual_yield']:.1%} on 300mm — but the 300mm wafer produces "
            f"{results[300]['pass_count'] - results[200]['pass_count']} more good dies per run. "
            f"This is why the industry moved to 300mm.")

# ── Analysis charts ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Yield Analysis")

tab1, tab2 = st.tabs(["Yield vs Defect Density", "Yield vs Die Size"])

with tab1:
    fig_density = build_yield_vs_density_chart(die_size)
    # Mark current operating point
    fig_density.add_vline(x=defect_density, line_color=C_FAIL,
                          line_dash="dash", line_width=2,
                          annotation_text=f"Current: {defect_density}",
                          annotation_font_color=C_FAIL)
    st.plotly_chart(fig_density, use_container_width=True)
    st.caption("Murphy's model is the industry standard because it accounts for defect clustering. "
               "Poisson assumes perfectly random defects — Murphy is more realistic for real fabs.")

with tab2:
    fig_diesize = build_yield_vs_diesize_chart(defect_density)
    fig_diesize.add_vline(x=die_size, line_color=C_FAIL,
                          line_dash="dash", line_width=2,
                          annotation_text=f"Current: {die_size}mm²",
                          annotation_font_color=C_FAIL)
    st.plotly_chart(fig_diesize, use_container_width=True)
    st.caption("Larger dies are exponentially harder to yield. This is why chip architects "
               "split large SoCs into chiplets — smaller dies yield better and can be "
               "combined using advanced packaging (Intel Foveros, AMD 3D V-Cache).")

# ── Raw data ──────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📄 Die-level data table"):
    rows = []
    for col, row, cx, cy in res_primary["die_xy"]:
        status = res_primary["die_results"].get((col, row), "pass")
        rows.append({
            "Col": col, "Row": row,
            "X (mm)": round(cx, 2), "Y (mm)": round(cy, 2),
            "Dist from center (mm)": round(np.sqrt(cx**2 + cy**2), 2),
            "Status": "PASS" if status == "pass" else "FAIL"
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.apply(
            lambda x: ["background-color: #1a3a2a" if v == "PASS"
                       else "background-color: #3a1a1a" for v in x],
            subset=["Status"]
        ),
        use_container_width=True, height=300
    )
    st.download_button("Download die data CSV",
                       df.to_csv(index=False),
                       file_name="wafer_die_data.csv",
                       mime="text/csv")
