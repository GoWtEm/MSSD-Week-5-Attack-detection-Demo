# SUTD 51.508: Secure Cyber Physical Systems
## Project: Anomaly Detection in Sensor Data (LIT101)

### Overview
This project demonstrates two different methodologies for detecting anomalies in Cyber-Physical Systems (CPS). We specifically focus on the **LIT101** sensor (Water Level Indicator) measurements obtained from the **Secure Water Treatment (SWaT) testbed**. For more such datasets of other testbeds at iTrust, visit [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

The demo explores:
1.  **CUSUM (Cumulative Sum)**: A statistical control chart method used to detect small, persistent shifts in the mean of a process.
2.  **MLP (Multi-Layer Perceptron)**: A Deep Learning approach that learns the normal behavioral patterns of the system and flags deviations as potential security breaches.

---

### Project Structure
```text
.
├── CUSUM_demo.py          # Statistical Anomaly Detection Script
├── MLP_demo.py            # Deep Learning Anomaly Detection Script
├── Dataset/
│   └── dataset.csv        # Raw sensor data (LIT101, Valve, Pump)
├── CUSUM_results/         # Generated CUSUM plots and CSV results
├── MLP_results/           # Generated MLP plots and trained model (.h5)
├── requirements.txt       # Project dependencies
└── .venv/                 # Python 3.12.10 Virtual Environment
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

2. **Install Dependencies** (if not already installed):
   ```bash
   uv pip install -r requirements.txt
   # OR
   pip install -r requirements.txt
   ```

---

### How to Run the Demos

#### 1. Statistical Demo (CUSUM)
This script calculates baseline statistics and applies the CUSUM algorithm to detect shifts.
```bash
python CUSUM_demo.py
```
*   **What to expect**: A plot showing sensor readings with red dots marking anomalies.
*   **Outputs**: Results are saved in `CUSUM_results/`.

#### 2. Machine Learning Demo (MLP)
This script trains a Neural Network to "guess" the next water level reading and identifies attacks when those guesses are wildly inaccurate.
```bash
python MLP_demo.py
```
*   **What to expect**: Multiple plots showing system behavior, training progress, and an "Attack Scenario" where the AI identifies a malicious data mutation.
*   **Outputs**: All plots and the trained brain (`LIT101.h5`) are saved in `MLP_results/`.

---

### Key Educational Concepts for MSSD
- **Baseline Establishment**: Using normal data to define what "secure" operation looks like.
- **Threshold Setting**: Defining the boundaries of acceptable operation (Safety Limits).
- **Attack Simulation**: Demonstrating how data mutation attacks (Man-in-the-Middle) can be detected through behavioral analysis.
- **Reproducibility**: Using fixed seeds (Seed 14) to ensure security research results are verifiable and consistent.

---
*Created for MSSD Class 51.508: Secure Cyber Physical Systems*

### Copyright & Usage Notice
**© 2026. All Rights Reserved.**
These scripts and materials are provided strictly for **educational and research purposes**. This code should not be used in production environments or for any commercial applications without explicit authorization.
