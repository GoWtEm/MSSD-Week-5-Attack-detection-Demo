# SUTD 51.508: Secure Cyber-Physical Systems
## Project: Anomaly Detection in Sensor Data (LIT101)

### Overview
This project demonstrates two distinct methodologies for detecting anomalies in Cyber-Physical Systems (CPS). We specifically focus on the **LIT101** sensor (Water Level Indicator) measurements obtained from the **Secure Water Treatment (SWaT) testbed**.

For more datasets from other testbeds at iTrust, please visit: [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/)

**The demo explores:**
1.  **CUSUM (Cumulative Sum)**: A statistical control chart method used to detect small, persistent shifts in the mean of a process.
2.  **MLP (Multi-Layer Perceptron)**: A Deep Learning approach that learns normal behavioral patterns of the system to flag deviations as potential security breaches.

---

### Project Structure
```text
.
├── CUSUM_demo.py          # Statistical Anomaly Detection Script
├── MLP_demo.py            # Deep Learning Anomaly Detection Script
├── pyproject.toml         # Project metadata and dependencies (Modern PEP 621)
├── Dataset/
│   └── dataset.csv        # Raw sensor data (LIT101, Valve, Pump)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

### Prerequisites
- **Python**: 3.12.10
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

---

### Setup Instructions

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   # Using uv
   uv pip install -r requirements.txt
   
   # Using pip
   pip install -r requirements.txt
   ```

---

### Driving the Demos

#### 1. Statistical Analysis (CUSUM)
This script calculates baseline statistics and applies the CUSUM algorithm to detect operational shifts.
```bash
python CUSUM_demo.py
```
*   **Expectation**: An interactive plot showing sensor readings with red markers highlighting detected anomalies.
*   **Outputs**: Visual results are saved to the `CUSUM_results/` directory.

#### 2. Machine Learning Analysis (MLP)
This script trains a neural network to predict water level readings, identifying potential attacks when predictions deviate significantly from actual values.
```bash
python MLP_demo.py
```
*   **Expectation**: Multiple visualizations illustrating system behavior, training convergence, and an "Attack Scenario" where the model identifies a synthetic data mutation attack.
*   **Outputs**: Plots and the trained model (`LIT101.h5`) are saved to the `MLP_results/` directory.

---

### Key Educational Concepts
- **Baseline Establishment**: Utilizing historical data to define "secure" and "normal" operational states.
- **Threshold Configuration**: Defining safety limits and sensitivity parameters for anomaly detection.
- **Attack Simulation**: Demonstrating how Man-in-the-Middle (MitM) data mutation attacks can be identified through behavioral analysis.
- **Reproducibility**: Implementing fixed stochastic seeds (Seed 14) to ensure verifiable and consistent research results.

---

### Copyright & Usage Notice
**© 2026 iTrust SUTD. All Rights Reserved.**

These scripts and materials are provided strictly for **educational and research purposes**. This code should not be utilized in production environments or for commercial applications without explicit authorization.

*Created for MSSD Class 51.508: Secure Cyber-Physical Systems*
