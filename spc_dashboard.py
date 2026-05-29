"""
spc_dashboard.py — Interactive SPC Dashboard
=============================================
Run with: streamlit run spc_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fab SPC Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
A2 = {2:1.880, 3:1.023, 4:0.729, 5:0.577, 6:0.483, 7:0.419, 8:0.373, 9:0.337, 10:0.308}
D3 = {2:0,     3:0,     4:0,     5:0,     6:0,     7:0.076, 8:0.136, 9:0.184, 10:0.223}
D4 = {2:3.267, 3:2.574, 4:2.282, 5:2.114, 6:2.004, 7:1.924, 8:1.864, 9:1.816, 10:1.777}

MEASUREMENTS = {
    "Furnace Temperature":    {"col": "temp_C",            "unit": "°C",     "type": "xbar_r", "target": 1000.0, "lsl": 994.0,  "usl": 1006.0},
    "Etch Chamber Pressure":  {"col": "pressure_mTorr",    "unit": "mTorr",  "type": "xbar_r", "target": 50.0,  "lsl": 47.0,   "usl": 53.0},
    "Oxide Thickness":        {"col": "thickness_A",       "unit": "Å",      "type": "xbar_r", "target": 500.0, "lsl": 490.0,  "usl": 510.0},
    "Gate Leakage Current":   {"col": "leakage_nA",        "unit": "nA",     "type": "xbar_r", "target": 1.2,   "lsl": 0.0,    "usl": 2.5},
    "Defects per Wafer":      {"col": "defects_per_wafer", "unit": "count",  "type": "c_chart","target": None,  "lsl": None,   "usl": None},
}

COLORS = {
    "cl":      "#1F3A6B",
    "ucl":     "#E74C3C",
    "lcl":     "#E74C3C",
    "target":  "#27AE60",
    "data":    "#2C3E50",
    "ooc":     "#E74C3C",
    "warn":    "rgba(231,76,60,0.08)",
    "in":      "#27AE60",
}

# ── Western Electric Rules ────────────────────────────────────────────────────
def we_rules(values, cl, ucl, lcl):
    n = len(values)
    sigma = (ucl - cl) / 3.0 if ucl != cl else 1e-9
    z = (np.array(values) - cl) / sigma
    v = {"Rule 1 — Beyond 3σ": [], "Rule 2 — 9 pts same side": [],
         "Rule 3 — 6 pts trending": [], "Rule 4 — 14 pts alternating": []}
    for i in range(n):
        if abs(z[i]) > 3.0:
            v["Rule 1 — Beyond 3σ"].append(i)
    for i in range(8, n):
        w = z[i-8:i+1]
        if all(x > 0 for x in w) or all(x < 0 for x in w):
            v["Rule 2 — 9 pts same side"].append(i)
    for i in range(5, n):
        w = values[i-5:i+1]
        d = np.diff(w)
        if all(x > 0 for x in d) or all(x < 0 for x in d):
            v["Rule 3 — 6 pts trending"].append(i)
    for i in range(13, n):
        w = values[i-13:i+1]
        d = np.diff(w)
        if all(d[j]*d[j+1] < 0 for j in range(len(d)-1)):
            v["Rule 4 — 14 pts alternating"].append(i)
    return v

def ooc_set(v):
    s = set()
    for pts in v.values():
        s.update(pts)
    return sorted(s)

# ── Stats computation ─────────────────────────────────────────────────────────
def xbar_r_stats(data, col, n):
    g = data.groupby("subgroup")[col]
    xbar = g.mean()
    R    = g.max() - g.min()
    sg   = xbar.index.values
    xbar_cl  = xbar.mean();  R_cl = R.mean()
    xbar_ucl = xbar_cl + A2[n] * R_cl
    xbar_lcl = xbar_cl - A2[n] * R_cl
    R_ucl    = D4[n] * R_cl;  R_lcl = D3[n] * R_cl
    sigma    = R_cl / 2.326
    return dict(sg=sg, xbar=xbar.values, R=R.values,
                xbar_cl=xbar_cl, xbar_ucl=xbar_ucl, xbar_lcl=xbar_lcl,
                R_cl=R_cl, R_ucl=R_ucl, R_lcl=R_lcl, sigma=sigma)

def c_chart_stats(data, col):
    g   = data.groupby("subgroup")[col]
    c   = g.mean()
    sg  = c.index.values
    cl  = c.mean()
    ucl = cl + 3*np.sqrt(cl)
    lcl = max(0, cl - 3*np.sqrt(cl))
    return dict(sg=sg, c=c.values, cl=cl, ucl=ucl, lcl=lcl)

def cp_cpk(xbar_cl, sigma, lsl, usl):
    cp  = (usl - lsl) / (6*sigma)
    cpu = (usl - xbar_cl) / (3*sigma)
    cpl = (xbar_cl - lsl) / (3*sigma)
    cpk = min(cpu, cpl)
    return cp, cpk

# ── Plotly chart builders ─────────────────────────────────────────────────────
def add_limit_lines(fig, sg, cl, ucl, lcl, target, unit, row, sigma, show_warn=True):
    """Add CL, UCL, LCL, target, and warning zone to a subplot row."""
    x_fill = list(sg) + list(sg[::-1])

    if show_warn:
        # upper warning zone
        fig.add_trace(go.Scatter(
            x=list(sg) + list(sg[::-1]),
            y=[cl + 2*sigma]*len(sg) + [ucl]*len(sg),
            fill="toself", fillcolor=COLORS["warn"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1)
        # lower warning zone
        fig.add_trace(go.Scatter(
            x=list(sg) + list(sg[::-1]),
            y=[max(lcl, cl - 3*sigma)]*len(sg) + [max(lcl, cl - 2*sigma)]*len(sg),
            fill="toself", fillcolor=COLORS["warn"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=row, col=1)

    for val, color, dash, name in [
        (ucl, COLORS["ucl"], "dash",  f"UCL = {ucl:.3f} {unit}"),
        (cl,  COLORS["cl"],  "solid", f"CL  = {cl:.3f} {unit}"),
        (lcl, COLORS["lcl"], "dash",  f"LCL = {lcl:.3f} {unit}"),
    ]:
        fig.add_hline(y=val, line_color=color, line_dash=dash,
                      line_width=1.8, row=row, col=1,
                      annotation_text=name,
                      annotation_position="top right",
                      annotation_font_size=10)

    if target is not None:
        fig.add_hline(y=target, line_color=COLORS["target"], line_dash="dot",
                      line_width=1.5, row=row, col=1,
                      annotation_text=f"Target = {target}",
                      annotation_position="bottom right",
                      annotation_font_size=10)

def build_xbar_r_chart(stats, meta, name, show_rules):
    sg     = stats["sg"]
    unit   = meta["unit"]
    target = meta["target"]
    sigma  = (stats["xbar_ucl"] - stats["xbar_cl"]) / 3

    xbar_v = we_rules(stats["xbar"], stats["xbar_cl"], stats["xbar_ucl"], stats["xbar_lcl"])
    R_v    = we_rules(stats["R"],    stats["R_cl"],    stats["R_ucl"],    max(stats["R_lcl"], 0))
    xbar_ooc = ooc_set(xbar_v) if show_rules else []
    R_ooc    = ooc_set(R_v)    if show_rules else []

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Xbar Chart — Subgroup Means", "R Chart — Subgroup Ranges"),
                        vertical_spacing=0.18)

    # ── Xbar ──
    in_ctrl_x = [i for i in range(len(sg)) if i not in xbar_ooc]
    fig.add_trace(go.Scatter(
        x=sg, y=stats["xbar"], mode="lines+markers",
        name="Subgroup mean",
        line=dict(color=COLORS["data"], width=1.8),
        marker=dict(size=7, color=[COLORS["ooc"] if i in xbar_ooc else COLORS["data"] for i in range(len(sg))],
                    line=dict(width=1, color="white")),
        customdata=[[f"Rule violations: {', '.join(r for r,pts in xbar_v.items() if i in pts) or 'None'}"] for i in range(len(sg))],
        hovertemplate=f"Subgroup %{{x}}<br>Mean: %{{y:.4f}} {unit}<br>%{{customdata[0]}}<extra></extra>"
    ), row=1, col=1)

    if xbar_ooc:
        fig.add_trace(go.Scatter(
            x=sg[xbar_ooc], y=stats["xbar"][xbar_ooc],
            mode="markers", name="Out of control",
            marker=dict(size=13, color=COLORS["ooc"], symbol="circle",
                        line=dict(width=2, color="white")),
            hovertemplate=f"Subgroup %{{x}}<br>Mean: %{{y:.4f}} {unit}<br><b>OUT OF CONTROL</b><extra></extra>"
        ), row=1, col=1)

    add_limit_lines(fig, sg, stats["xbar_cl"], stats["xbar_ucl"], stats["xbar_lcl"],
                    target, unit, row=1, sigma=sigma)

    # ── R chart ──
    R_sigma = (stats["R_ucl"] - stats["R_cl"]) / 3
    fig.add_trace(go.Scatter(
        x=sg, y=stats["R"], mode="lines+markers",
        name="Subgroup range",
        line=dict(color=COLORS["data"], width=1.8, dash="dot"),
        marker=dict(size=7, color=[COLORS["ooc"] if i in R_ooc else COLORS["data"] for i in range(len(sg))],
                    symbol="square", line=dict(width=1, color="white")),
        hovertemplate=f"Subgroup %{{x}}<br>Range: %{{y:.4f}} {unit}<extra></extra>"
    ), row=2, col=1)

    if R_ooc:
        fig.add_trace(go.Scatter(
            x=sg[R_ooc], y=stats["R"][R_ooc],
            mode="markers", name="R — Out of control",
            marker=dict(size=13, color=COLORS["ooc"], symbol="square",
                        line=dict(width=2, color="white")),
            hovertemplate=f"Subgroup %{{x}}<br>Range: %{{y:.4f}} {unit}<br><b>OUT OF CONTROL</b><extra></extra>"
        ), row=2, col=1)

    add_limit_lines(fig, sg, stats["R_cl"], stats["R_ucl"], max(stats["R_lcl"], 0),
                    None, unit, row=2, sigma=R_sigma, show_warn=False)

    fig.update_layout(height=600, template="plotly_white",
                      title=dict(text=f"Xbar-R Control Chart — {name} ({unit})",
                                 font=dict(size=16, color=COLORS["cl"]), x=0.5),
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(tickmode="linear", dtick=1, title_text="Subgroup Number", row=2, col=1)
    fig.update_xaxes(tickmode="linear", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text=f"Mean ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Range ({unit})", row=2, col=1)

    return fig, xbar_v, R_v


def build_c_chart(stats, meta, name, show_rules):
    sg   = stats["sg"]
    unit = meta["unit"]
    sigma = np.sqrt(stats["cl"])

    v   = we_rules(stats["c"], stats["cl"], stats["ucl"], stats["lcl"])
    ooc = ooc_set(v) if show_rules else []

    fig = go.Figure()

    # Warning zones
    fig.add_trace(go.Scatter(
        x=list(sg) + list(sg[::-1]),
        y=[stats["cl"] + 2*sigma]*len(sg) + [stats["ucl"]]*len(sg),
        fill="toself", fillcolor=COLORS["warn"],
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=sg, y=stats["c"], mode="lines+markers",
        name="Mean defects/subgroup",
        line=dict(color=COLORS["data"], width=1.8),
        marker=dict(size=7, color=[COLORS["ooc"] if i in ooc else COLORS["data"] for i in range(len(sg))],
                    line=dict(width=1, color="white")),
        hovertemplate="Subgroup %{x}<br>Defects: %{y:.2f}<extra></extra>"
    ))

    if ooc:
        fig.add_trace(go.Scatter(
            x=sg[ooc], y=stats["c"][ooc],
            mode="markers", name="Out of control",
            marker=dict(size=13, color=COLORS["ooc"], line=dict(width=2, color="white")),
            hovertemplate="Subgroup %{x}<br>Defects: %{y:.2f}<br><b>OUT OF CONTROL</b><extra></extra>"
        ))

    for val, color, dash, label in [
        (stats["ucl"], COLORS["ucl"], "dash",  f"UCL = {stats['ucl']:.3f}"),
        (stats["cl"],  COLORS["cl"],  "solid", f"CL  = {stats['cl']:.3f}"),
        (stats["lcl"], COLORS["lcl"], "dash",  f"LCL = {stats['lcl']:.3f}"),
    ]:
        fig.add_hline(y=val, line_color=color, line_dash=dash, line_width=1.8,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_size=10)

    fig.update_layout(height=420, template="plotly_white",
                      title=dict(text=f"C Chart — {name}", font=dict(size=16, color=COLORS["cl"]), x=0.5),
                      hovermode="x unified",
                      xaxis=dict(tickmode="linear", dtick=1, title="Subgroup Number"),
                      yaxis=dict(title="Mean Defects per Wafer"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, v


# ── Violation summary helper ──────────────────────────────────────────────────
def violation_summary(v, chart_name):
    total = sum(len(pts) for pts in v.values())
    if total == 0:
        st.success(f"✓ {chart_name}: **IN CONTROL** — no violations detected")
        return
    st.error(f"✗ {chart_name}: **{total} violation(s)** detected")
    for rule, pts in v.items():
        if pts:
            sg_nums = [p + 1 for p in pts]
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**{rule}** → subgroups {sg_nums}")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚙️ Semiconductor Fab — SPC Dashboard")
st.markdown("Interactive Statistical Process Control for fab process monitoring. Hover over any point for details.")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    selected = st.selectbox("Select measurement", list(MEASUREMENTS.keys()))
    show_rules = st.toggle("Show run rule violations", value=True)
    st.divider()
    st.markdown("**About this dashboard**")
    st.markdown("Built with Python · Streamlit · Plotly")
    st.markdown("Applies all 4 Western Electric run rules to detect out-of-control conditions.")
    st.divider()
    st.markdown("**Western Electric Rules**")
    st.markdown("1. 1 point beyond ±3σ")
    st.markdown("2. 9 pts same side of CL")
    st.markdown("3. 6 pts trending up/down")
    st.markdown("4. 14 pts alternating ±")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    data = pd.read_csv("fab_process_data.csv")
except FileNotFoundError:
    st.error("fab_process_data.csv not found. Run generate_data.py first.")
    st.stop()

meta = MEASUREMENTS[selected]
col  = meta["col"]
n    = int(data.groupby("subgroup")[col].count().iloc[0])

# ── Summary metrics row ───────────────────────────────────────────────────────
st.subheader(f"📊 {selected} ({meta['unit']})")

if meta["type"] == "xbar_r":
    stats = xbar_r_stats(data, col, n)

    total_ooc_xbar = len(ooc_set(we_rules(stats["xbar"], stats["xbar_cl"], stats["xbar_ucl"], stats["xbar_lcl"])))
    total_ooc_R    = len(ooc_set(we_rules(stats["R"],    stats["R_cl"],    stats["R_ucl"],    max(stats["R_lcl"], 0))))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Centerline (CL)", f"{stats['xbar_cl']:.3f} {meta['unit']}")
    c2.metric("UCL", f"{stats['xbar_ucl']:.3f}")
    c3.metric("LCL", f"{stats['xbar_lcl']:.3f}")
    c4.metric("Xbar violations", total_ooc_xbar, delta=None if total_ooc_xbar == 0 else f"{total_ooc_xbar} OOC",
              delta_color="inverse")
    c5.metric("R violations", total_ooc_R, delta=None if total_ooc_R == 0 else f"{total_ooc_R} OOC",
              delta_color="inverse")

    st.divider()

    # Chart
    fig, xbar_v, R_v = build_xbar_r_chart(stats, meta, selected, show_rules)
    st.plotly_chart(fig, use_container_width=True)

    # Violation details
    st.subheader("🔍 Run Rule Results")
    col_a, col_b = st.columns(2)
    with col_a:
        violation_summary(xbar_v, "Xbar Chart")
    with col_b:
        violation_summary(R_v, "R Chart")

    # Capability
    if meta["lsl"] is not None:
        st.divider()
        st.subheader("📐 Process Capability")
        cp, cpk = cp_cpk(stats["xbar_cl"], stats["sigma"], meta["lsl"], meta["usl"])

        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        cc1.metric("σ̂ (estimated)", f"{stats['sigma']:.4f}")
        cc2.metric("Cp", f"{cp:.3f}", delta="Capable" if cp >= 1.33 else "Not capable",
                   delta_color="normal" if cp >= 1.33 else "inverse")
        cc3.metric("Cpk", f"{cpk:.3f}", delta="Centered" if cpk >= 1.33 else "Needs attention",
                   delta_color="normal" if cpk >= 1.33 else "inverse")
        cc4.metric("LSL", f"{meta['lsl']} {meta['unit']}")
        cc5.metric("USL", f"{meta['usl']} {meta['unit']}")

        # Capability gauge
        st.divider()
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cpk,
            title={"text": "Cpk", "font": {"size": 20}},
            delta={"reference": 1.33, "increasing": {"color": COLORS["in"]},
                   "decreasing": {"color": COLORS["ooc"]}},
            gauge={
                "axis": {"range": [0, 2.5], "tickwidth": 1},
                "bar":  {"color": COLORS["cl"]},
                "steps": [
                    {"range": [0, 1.0],  "color": "rgba(231,76,60,0.2)"},
                    {"range": [1.0, 1.33], "color": "rgba(243,156,18,0.2)"},
                    {"range": [1.33, 1.67], "color": "rgba(39,174,96,0.15)"},
                    {"range": [1.67, 2.5], "color": "rgba(39,174,96,0.25)"},
                ],
                "threshold": {"line": {"color": COLORS["ooc"], "width": 3},
                              "thickness": 0.75, "value": 1.33}
            }
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=20, l=40, r=40))

        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(gauge, use_container_width=True)
        with g2:
            st.markdown("**Cpk Interpretation**")
            st.markdown("""
| Cpk | Status |
|-----|--------|
| < 1.0 | 🔴 Not capable — making defects |
| 1.0 – 1.33 | 🟡 Marginal — no safety margin |
| 1.33 – 1.67 | 🟢 Capable — industry minimum |
| ≥ 1.67 | ✅ Semiconductor standard |
| ≥ 2.0 | ⭐ Six Sigma level |
""")
            if cpk < 1.0:
                st.error(f"**Action required:** Cpk = {cpk:.3f}. Process is not capable of meeting spec limits. Investigate immediately.")
            elif cpk < 1.33:
                st.warning(f"**Monitor closely:** Cpk = {cpk:.3f}. Process has no margin. Any drift will produce defects.")
            else:
                st.success(f"**Process capable:** Cpk = {cpk:.3f}. Within acceptable range.")

elif meta["type"] == "c_chart":
    stats = c_chart_stats(data, col)
    v     = we_rules(stats["c"], stats["cl"], stats["ucl"], stats["lcl"])
    total_ooc = len(ooc_set(v))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Centerline (CL)", f"{stats['cl']:.3f}")
    c2.metric("UCL", f"{stats['ucl']:.3f}")
    c3.metric("LCL", f"{stats['lcl']:.3f}")
    c4.metric("Violations", total_ooc, delta=None if total_ooc == 0 else f"{total_ooc} OOC",
              delta_color="inverse")

    st.divider()
    fig, v = build_c_chart(stats, meta, selected, show_rules)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Run Rule Results")
    violation_summary(v, "C Chart")

# ── Raw data expander ─────────────────────────────────────────────────────────
st.divider()
with st.expander("📄 View raw process data"):
    st.dataframe(data, use_container_width=True, height=300)
    st.download_button("Download CSV", data.to_csv(index=False),
                       file_name="fab_process_data.csv", mime="text/csv")
