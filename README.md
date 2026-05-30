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

## Project 3 — ML Fault Detection (Plasma Etch Tool)

Machine learning fault detection and classification system for plasma
etch tool sensor data. Mirrors Fault Detection & Classification (FDC)
systems used at Lam Research, Applied Materials, and Intel.

**What it does:**

- Generates realistic plasma etch tool sensor data across 7 sensors
  (RF power, chamber pressure, gas flow, chuck temperature, bias voltage,
  reflected power, DC bias)
- Simulates 4 fault conditions: pressure drift, RF instability, gas flow
  anomaly, chuck temperature excursion
- Extracts 49 statistical features per sensor window (mean, std, slope,
  RMS, range)
- Trains Random Forest, Gradient Boosting, and Logistic Regression
  classifiers — 100% test accuracy across all 5 classes
- Interactive Streamlit dashboard with real-time fault injection,
  sensor trace visualization, probability charts, and recommended
  corrective actions

**Run it:**

```bash
python generate_etch_data.py
python fault_detector.py --save
streamlit run fault_dashboard.py
```

**Fault classes:**
| Class | Fault | Primary Sensors Affected |
|-------|-------|--------------------------|
| 0 | Normal | All within spec |
| 1 | Pressure Drift | Pressure ↑, DC Bias ↓ |
| 2 | RF Instability | RF Power oscillates, Reflected Power ↑ |
| 3 | Gas Flow Anomaly | Gas Flow ↓, Pressure ↓, DC Bias ↓ |
| 4 | Chuck Temp Excursion | Chuck Temp ↑, Bias Voltage ↑ |

---

## Installation

```bash
pip install numpy pandas matplotlib streamlit plotly scikit-learn
```

## Skills demonstrated

Python · Machine Learning · Random Forest · Feature Engineering ·
Statistical Process Control · Wafer Yield Analysis · Murphy's Yield Model ·
Semiconductor Fab Process Knowledge · Plasma Etch · Fault Detection &
Classification (FDC) · Streamlit · Plotly · Scikit-learn · Numpy · Pandas ·
Western Electric Run Rules · Process Capability (Cp/Cpk)
