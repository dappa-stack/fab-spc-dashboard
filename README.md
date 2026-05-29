# Semiconductor Fab SPC Dashboard

Interactive Statistical Process Control (SPC) monitoring tool for semiconductor
fabrication process data. Built with Python, Streamlit, and Plotly.

## What it does

- Computes **Xbar-R control charts** for continuous measurements (temperature,
  pressure, oxide thickness, leakage current)
- Computes **C charts** for defect count data
- Applies all 4 **Western Electric run rules** to detect out-of-control patterns
- Calculates **Cp and Cpk** process capability indices
- Interactive **Streamlit dashboard** with hover tooltips, real-time violation
  detection, and capability gauges

## Process parameters monitored

| Parameter             | Target   | Chart Type |
| --------------------- | -------- | ---------- |
| Furnace Temperature   | 1000°C   | Xbar-R     |
| Etch Chamber Pressure | 50 mTorr | Xbar-R     |
| Oxide Thickness       | 500 Å    | Xbar-R     |
| Gate Leakage Current  | 1.2 nA   | Xbar-R     |
| Defects per Wafer     | —        | C Chart    |

## How to run

**Install dependencies**

```bash
pip install numpy pandas matplotlib streamlit plotly
```

**Generate sample data**

```bash
python generate_data.py
```

**Command-line analysis**

```bash
python spc_tool.py --save
```

**Interactive dashboard**

```bash
streamlit run spc_dashboard.py
```

## Skills demonstrated

Python · Statistical Process Control · Semiconductor Process Monitoring ·
Streamlit · Plotly · Numpy · Pandas · Western Electric Run Rules ·
Process Capability Analysis (Cp/Cpk)
