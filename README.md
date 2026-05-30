# Semiconductor Fab Engineering Portfolio

Python-based semiconductor process engineering tools covering SPC monitoring,
wafer yield analysis, and equipment fault detection. Built to demonstrate
process, yield, and equipment engineering skills for roles at companies like
Lam Research, Intel, TSMC, Applied Materials, and KLA.

---

## Project 1 — SPC Process Monitoring Dashboard

Statistical Process Control tool for semiconductor fab process data.

**What it does:**

- Computes Xbar-R control charts for continuous measurements (temperature,
  pressure, oxide thickness, leakage current)
- Computes C charts for defect count data
- Applies all 4 Western Electric run rules to detect out-of-control patterns
- Calculates Cp and Cpk process capability indices
- Interactive Streamlit dashboard with hover tooltips and capability gauges

**Run it:**

```bash
python generate_data.py
python spc_tool.py --save
streamlit run spc_dashboard.py
```

**Process parameters monitored:**
| Parameter | Target | Chart Type |
|-----------|--------|------------|
| Furnace Temperature | 1000°C | Xbar-R |
| Etch Chamber Pressure | 50 mTorr | Xbar-R |
| Oxide Thickness | 500 Å | Xbar-R |
| Gate Leakage Current | 1.2 nA | Xbar-R |
| Defects per Wafer | — | C Chart |

---

## Project 2 — Wafer Yield Map Simulator

Wafer-level defect distribution and die yield simulation using
industry-standard models.

**What it does:**

- Simulates random defect distribution using Poisson statistics
- Calculates die yield using Murphy's model (industry standard)
- Generates color-coded wafer maps showing pass/fail die locations
- Supports 200mm and 300mm wafer sizes with edge exclusion zones
- Simulates localized defect clusters (contamination events)
- Interactive Streamlit dashboard with real-time parameter sliders
- Yield vs defect density and yield vs die size analysis charts

**Run it:**

```bash
python yield_simulator.py --wafer both --save
python yield_simulator.py --cluster --save
streamlit run yield_dashboard.py
```

**Key models:**
| Model | Formula | Use case |
|-------|---------|---------|
| Murphy's | Y = [(1-e^(-D₀A)) / D₀A]² | Industry standard |
| Poisson | Y = e^(-D₀A) | Simple estimate |

---

## Installation

```bash
pip install numpy pandas matplotlib streamlit plotly
```

---

## Skills demonstrated

Python · Statistical Process Control · Semiconductor Process Monitoring ·
Wafer Yield Analysis · Murphy's Yield Model · Streamlit · Plotly · Numpy ·
Pandas · Western Electric Run Rules · Process Capability (Cp/Cpk) ·
Defect Density Modeling · Semiconductor Fab Process Knowledge
