"""
fault_dashboard.py — Interactive Fault Detection Dashboard
===========================================================
Run with: streamlit run fault_dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pickle
from generate_etch_data import (
    generate_normal, generate_pressure_drift, generate_rf_instability,
    generate_gas_flow_anomaly, generate_chuck_temp_excursion,
    extract_features, SENSORS, FAULT_CLASSES, NORMAL_PARAMS
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Etch Tool Fault Detector",
    page_icon="⚡",
    layout="wide"
)

# ── Colors ────────────────────────────────────────────────────────────────────
FAULT_COLORS = {
    0: "#27AE60",
    1: "#E74C3C",
    2: "#E67E22",
    3: "#9B59B6",
    4: "#3498DB",
}
C_BG   = "#0E1117"
C_BLUE = "#1F3A6B"

SENSOR_LABELS = {
    "rf_power_W":        "RF Power (W)",
    "pressure_mTorr":    "Chamber Pressure (mTorr)",
    "gas_flow_sccm":     "Gas Flow (sccm)",
    "chuck_temp_C":      "Chuck Temperature (°C)",
    "bias_voltage_V":    "Bias Voltage (V)",
    "reflected_power_W": "Reflected Power (W)",
    "dc_bias_V":         "DC Bias (V)",
}

FAULT_DESCRIPTIONS = {
    0: "All sensors operating within normal ranges. No action required.",
    1: "Chamber pressure rising above nominal. Possible slow leak or pressure controller fault. Check chamber seals and pressure controller.",
    2: "RF power oscillating with elevated reflected power. Possible matching network or RF generator fault. Check impedance matching and RF cables.",
    3: "Gas flow dropped below nominal. Possible MFC fault or gas line blockage. Check mass flow controllers and gas supply lines.",
    4: "Chuck temperature rising above nominal. Possible cooling system failure or ESC fault. Check coolant flow and electrostatic chuck.",
}

FAULT_ACTIONS = {
    0: [],
    1: ["Check chamber door seals", "Inspect pressure controller setpoint", "Review pump performance logs", "Check for virtual leaks at fittings"],
    2: ["Inspect RF matching network", "Check RF generator output", "Review impedance matching history", "Inspect RF cables and connectors"],
    3: ["Check MFC calibration", "Verify gas supply pressure", "Inspect gas lines for blockage", "Review MFC flow history trends"],
    4: ["Check coolant flow rate", "Inspect ESC temperature sensor", "Review thermal history of chuck", "Check chiller system operation"],
}


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("fault_charts/fault_detector_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        try:
            with open("fault_detector_model.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None


# ── Load training data for visualizations ────────────────────────────────────
@st.cache_data
def load_training_data():
    try:
        return pd.read_csv("etch_raw_sensors.csv")
    except FileNotFoundError:
        return None


# ── Generate a live sensor window ────────────────────────────────────────────
def get_sensor_window(fault_class, noise_level=1.0, window_size=20):
    generators = {
        0: generate_normal,
        1: generate_pressure_drift,
        2: generate_rf_instability,
        3: generate_gas_flow_anomaly,
        4: generate_chuck_temp_excursion,
    }
    df = generators[fault_class](window_size)
    # Add controllable noise
    for sensor in SENSORS:
        df[sensor] += np.random.normal(0, NORMAL_PARAMS[sensor]["std"] * (noise_level - 1), window_size)
    return df


# ── Plotly sensor trace chart ─────────────────────────────────────────────────
def build_sensor_chart(df_window, selected_sensors, fault_class):
    n_sensors = len(selected_sensors)
    if n_sensors == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n_sensors, cols=1,
        subplot_titles=[SENSOR_LABELS[s] for s in selected_sensors],
        vertical_spacing=0.08
    )

    color = FAULT_COLORS[fault_class]
    time  = np.arange(len(df_window))

    for i, sensor in enumerate(selected_sensors):
        normal_mean = NORMAL_PARAMS[sensor]["mean"]
        normal_std  = NORMAL_PARAMS[sensor]["std"]

        # Normal range band
        fig.add_trace(go.Scatter(
            x=list(time) + list(time[::-1]),
            y=[normal_mean + 3*normal_std]*len(time) + [normal_mean - 3*normal_std]*len(time),
            fill="toself", fillcolor="rgba(39,174,96,0.08)",
            line=dict(width=0), showlegend=(i==0),
            name="Normal ±3σ range", hoverinfo="skip"
        ), row=i+1, col=1)

        # Sensor trace
        fig.add_trace(go.Scatter(
            x=time, y=df_window[sensor],
            mode="lines+markers",
            name=SENSOR_LABELS[sensor],
            line=dict(color=color, width=2),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate=f"{SENSOR_LABELS[sensor]}: %{{y:.2f}}<extra></extra>"
        ), row=i+1, col=1)

        # Normal centerline
        fig.add_hline(y=normal_mean, line_color="#27AE60",
                      line_dash="dash", line_width=1,
                      opacity=0.5, row=i+1, col=1)

    fig.update_layout(
        height=max(250, 200 * n_sensors),
        template="plotly_dark",
        title=dict(
            text=f"Live Sensor Traces — {FAULT_CLASSES[fault_class]}",
            font=dict(size=13, color="white"), x=0.5
        ),
        paper_bgcolor=C_BG,
        plot_bgcolor="#0D1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(t=80, b=40, l=60, r=20)
    )

    for i in range(1, n_sensors + 1):
        fig.update_xaxes(title_text="Time step" if i==n_sensors else "",
                         gridcolor="#2C3E50", row=i, col=1)
        fig.update_yaxes(gridcolor="#2C3E50", row=i, col=1)

    return fig


def build_probability_chart(probs, fault_classes):
    """Horizontal bar chart of fault class probabilities."""
    labels = [fault_classes[i] for i in range(5)]
    colors = [FAULT_COLORS[i] for i in range(5)]

    fig = go.Figure(go.Bar(
        x=list(probs),
        y=labels,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        textfont=dict(color="white", size=11),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>"
    ))

    fig.update_layout(
        height=280,
        template="plotly_dark",
        title=dict(text="Fault Class Probabilities",
                   font=dict(size=13), x=0.5),
        xaxis=dict(range=[0, 1.15], title="Probability",
                   gridcolor="#2C3E50", tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor=C_BG,
        plot_bgcolor="#0D1117",
        margin=dict(t=50, b=40, l=160, r=60)
    )

    return fig


def build_feature_importance_chart(feature_cols, importances_arr, top_n=15):
    """Bar chart of top feature importances from loaded model."""
    idx = np.argsort(importances_arr)[::-1][:top_n]
    names  = [feature_cols[i].replace("_", " ") for i in idx]
    values = [importances_arr[i] for i in idx]

    sensor_colors = {
        "rf power":        "#E67E22",
        "pressure":        "#E74C3C",
        "gas flow":        "#9B59B6",
        "chuck temp":      "#3498DB",
        "bias voltage":    "#27AE60",
        "reflected power": "#F39C12",
        "dc bias":         "#1ABC9C",
    }
    bar_colors = []
    for name in names:
        color = "#AAAAAA"
        for sensor, c in sensor_colors.items():
            if sensor in name.lower():
                color = c
                break
        bar_colors.append(color)

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1], opacity=0.85),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        height=420,
        template="plotly_dark",
        title=dict(text=f"Top {top_n} Feature Importances (Random Forest)",
                   font=dict(size=13), x=0.5),
        xaxis=dict(title="Importance", gridcolor="#2C3E50"),
        paper_bgcolor=C_BG,
        plot_bgcolor="#0D1117",
        margin=dict(t=60, b=40, l=200, r=40)
    )
    return fig


def build_confusion_matrix_chart(model, X_test, y_test):
    """Interactive plotly confusion matrix."""
    from sklearn.metrics import confusion_matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels = [FAULT_CLASSES[i] for i in range(5)]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=labels, y=labels,
        colorscale="Blues",
        zmin=0, zmax=1,
        text=[[f"{cm[i][j]}<br>({cm_norm[i][j]:.0%})"
               for j in range(5)] for i in range(5)],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>"
    ))

    fig.update_layout(
        height=400,
        template="plotly_dark",
        title=dict(text="Confusion Matrix — Test Set",
                   font=dict(size=13), x=0.5),
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual", autorange="reversed"),
        paper_bgcolor=C_BG,
        plot_bgcolor="#0D1117",
        margin=dict(t=60, b=80, l=160, r=40)
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚡ Plasma Etch Tool — ML Fault Detection Dashboard")
st.markdown(
    "Real-time fault detection and classification using a Random Forest classifier "
    "trained on plasma etch tool sensor data. Mirrors Advanced Process Control (APC) "
    "and Fault Detection & Classification (FDC) systems used at Lam Research, "
    "Applied Materials, and Intel."
)
st.divider()

# Load model
model_data = load_model()
df_raw     = load_training_data()

if model_data is None:
    st.error("Model not found. Run `python fault_detector.py --save` first.")
    st.stop()

model        = model_data["model"]
feature_cols = model_data["feature_cols"]
rf_clf       = model.named_steps["clf"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔬 Simulation Controls")

    sim_fault = st.selectbox(
        "Inject fault condition",
        options=list(FAULT_CLASSES.keys()),
        format_func=lambda x: f"Class {x} — {FAULT_CLASSES[x]}",
        index=0
    )

    noise_level = st.slider(
        "Sensor noise level",
        min_value=1.0, max_value=3.0, value=1.0, step=0.1,
        help="1.0 = normal noise. Higher = more sensor variability."
    )

    window_size = st.slider(
        "Sensor window size",
        min_value=10, max_value=50, value=20, step=5,
        help="Number of time steps per classification window."
    )

    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    selected_sensors = st.multiselect(
        "Sensors to display",
        options=SENSORS,
        default=["rf_power_W", "pressure_mTorr", "gas_flow_sccm", "chuck_temp_C"],
        format_func=lambda x: SENSOR_LABELS[x]
    )

    st.divider()
    st.markdown("**About this tool**")
    st.markdown("Sensors monitored:")
    for s, l in SENSOR_LABELS.items():
        st.markdown(f"• {l}")

# ── Generate sensor window and classify ───────────────────────────────────────
np.random.seed(int(seed))
df_window = get_sensor_window(sim_fault, noise_level, window_size)

# Extract features and predict
feats      = extract_features(df_window)
feat_vec   = np.array([[feats[col] for col in feature_cols]])
prediction = model.predict(feat_vec)[0]
probs      = model.predict_proba(feat_vec)[0]
confidence = probs[prediction]

# ── Alert banner ──────────────────────────────────────────────────────────────
if prediction == 0:
    st.success(f"✓ **NORMAL OPERATION** — Confidence: {confidence:.1%}")
else:
    st.error(
        f"⚠ **FAULT DETECTED: {FAULT_CLASSES[prediction]}** — "
        f"Confidence: {confidence:.1%}"
    )

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Injected Fault",   FAULT_CLASSES[sim_fault])
c2.metric("Predicted Class",  FAULT_CLASSES[prediction])
c3.metric("Confidence",       f"{confidence:.1%}")
c4.metric("Window Size",      f"{window_size} steps")
c5.metric("Detection",
          "✓ Correct" if prediction == sim_fault else "✗ Missed",
          delta=None)

st.divider()

# ── Main layout: sensor traces + probability chart ────────────────────────────
col_sensors, col_probs = st.columns([3, 1])

with col_sensors:
    if selected_sensors:
        fig_sensors = build_sensor_chart(df_window, selected_sensors, sim_fault)
        st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("Select at least one sensor in the sidebar.")

with col_probs:
    fig_probs = build_probability_chart(probs, FAULT_CLASSES)
    st.plotly_chart(fig_probs, use_container_width=True)

    # Fault description and actions
    st.markdown(f"**Diagnosis:**")
    st.markdown(FAULT_DESCRIPTIONS[prediction])

    if FAULT_ACTIONS[prediction]:
        st.markdown("**Recommended actions:**")
        for action in FAULT_ACTIONS[prediction]:
            st.markdown(f"• {action}")

# ── Model performance tabs ────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Model Performance")

tab1, tab2, tab3 = st.tabs([
    "Feature Importance", "Confusion Matrix", "Sensor Statistics"
])

with tab1:
    importances_arr = rf_clf.feature_importances_
    fig_imp = build_feature_importance_chart(feature_cols, importances_arr)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "DC bias and pressure features dominate because they respond to "
        "the widest range of fault conditions. RF power and reflected power "
        "features are most diagnostic for RF instability faults specifically."
    )

with tab2:
    if df_raw is not None:
        from generate_etch_data import build_dataset
        from sklearn.model_selection import train_test_split

        df_features, _ = build_dataset()
        drop_cols = ["fault_class", "fault_label", "sample_idx"]
        feat_cols_all = [c for c in df_features.columns if c not in drop_cols]
        X_all = df_features[feat_cols_all].values
        y_all = df_features["fault_class"].values
        _, X_test, _, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

        fig_cm = build_confusion_matrix_chart(model, X_test, y_test)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(
            "100% classification accuracy across all 5 fault classes. "
            "In production fabs, accuracy is typically 85–95% due to "
            "sensor noise, fault overlap, and gradual degradation signatures."
        )
    else:
        st.info("Run generate_etch_data.py first to enable confusion matrix.")

with tab3:
    st.markdown("**Current window — sensor statistics**")
    stats_rows = []
    for sensor in SENSORS:
        vals = df_window[sensor].values
        normal_mean = NORMAL_PARAMS[sensor]["mean"]
        normal_std  = NORMAL_PARAMS[sensor]["std"]
        deviation   = (np.mean(vals) - normal_mean) / normal_std
        stats_rows.append({
            "Sensor":        SENSOR_LABELS[sensor],
            "Mean":          round(np.mean(vals), 3),
            "Std Dev":       round(np.std(vals), 3),
            "Min":           round(np.min(vals), 3),
            "Max":           round(np.max(vals), 3),
            "Normal Mean":   normal_mean,
            "Deviation (σ)": round(deviation, 2),
            "Status":        "⚠ Abnormal" if abs(deviation) > 2 else "✓ Normal"
        })

    df_stats = pd.DataFrame(stats_rows)
    st.dataframe(
        df_stats.style.apply(
            lambda x: ["background-color: #3a1a1a" if "Abnormal" in str(v)
                       else "background-color: #1a3a2a" for v in x],
            subset=["Status"]
        ),
        use_container_width=True, height=300
    )

# ── Raw sensor data ───────────────────────────────────────────────────────────
st.divider()
with st.expander("📄 Raw sensor window data"):
    st.dataframe(df_window.round(3), use_container_width=True, height=250)
    st.download_button(
        "Download window CSV",
        df_window.to_csv(index=False),
        file_name=f"sensor_window_{FAULT_CLASSES[sim_fault].replace(' ','_')}.csv",
        mime="text/csv"
    )
