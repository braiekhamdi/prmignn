
# PR-MF-GNN: A Physics-Informed Multi-Fidelity Graph Neural Network

**Authors:** Hamdi Braiek and Anis Bel Hadj Hassin 
**Repository:** [braiekhamdi/prmignn](https://github.com/braiekhamdi/prmignn)

## 📖 Overview

Graph neural network (GNN) surrogates for partial differential equations (PDEs) on unstructured meshes typically require hundreds of high-fidelity (HF) snapshots to generalize. **PR-MF-GNN** addresses this bottleneck through a two-stage, multi-fidelity training strategy:

1. **Stage 1 (Pre-training):** A message-passing processor is trained on abundant low-fidelity (LF) mesh data using a combined data and PDE-residual loss.
2. **Stage 2 (Fine-tuning):** The processor is frozen, and only a lightweight decoder head is fine-tuned on a handful of HF snapshots to correct systematic fidelity bias.

Because message passing operates on local, relative node and edge features, the learned processor transfers across meshes of different resolutions and node coordinates. To ensure robust convergence under severe data scarcity, key hyperparameters are optimized via Bayesian Optimization (BO).

## 📂 Repository Structure

```text
prmignn/
├── README.md
├── requirements.txt
├── main.py                # Entry point to run experiments (proposed & ablation)
├── tune.py                # Bayesian hyperparameter optimization (Optuna)
├── plot_from_json.py      # Optional script to regenerate figures from saved JSON
└── prmignn/               # Core Python package
    ├── __init__.py
    ├── config.py          # Hyperparameters and experimental configuration
    ├── mesh.py            # Irregular point-cloud and k-NN graph generation
    ├── pde.py             # Heat diffusion simulators (LF & HF)
    ├── features.py        # Node feature extraction and normalization
    ├── model.py           # NumPy implementation of the GNN and Adam optimizer
    ├── train.py           # Training loops and evaluation metrics
    ├── experiments.py     # Orchestration of baselines, ablations, and efficiency curves
    └── plotting.py        # Figure generation
```

## ⚙️ Installation

To ensure full reproducibility, set up a clean Python virtual environment. The implementation is entirely in **NumPy** and runs efficiently on a standard CPU.

**1. Clone the repository:**
```bash
git clone https://github.com/braiekhamdi/prmignn.git
cd prmignn
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Reproducibility Guide

To reproduce the results from the paper, follow these steps sequentially. The workflow is split into hyperparameter tuning, running the experiments, and generating the figures.

### Step 1: Bayesian Hyperparameter Tuning (Optional)
We use [Optuna](https://optuna.org/) to automatically optimize the three most relevant hyperparameters: Pre-training learning rate, PDE loss weight ($\lambda_{pde}$), and Fine-tuning learning rate. 
*Note: This step takes ~15-20 minutes. If you skip it, the code will use the default optimal parameters already set in `prmignn/config.py`.*

```bash
python tune.py
```
The best parameters will be saved to `results/best_params.json`. You may copy these into `prmignn/config.py` before proceeding.

### Step 2: Run the Experiments
You can run the proposed PR-MF-GNN model and the ablation/baseline studies separately, or all at once. 

**Option A: Run everything sequentially (Recommended)**
This will run the proposed model, baselines, ablations, and data efficiency curves, and will automatically generate all the figures at the end.
```bash
python main.py --run all
```

**Option B: Run separately**
If you prefer to run the proposed model and the ablations in isolated executions:
```bash
python main.py --run proposed
python main.py --run ablation
```

### Step 3: Generate Figures (If run separately)
If you used Option B above, or if you just want to regenerate the PDF figures from the saved JSON files without rerunning the experiments:
```bash
python plot_from_json.py
```
The plotting script automatically detects `results_proposed.json` and `results_ablation.json`, merges them, and generates all PDF figures used in the paper. Figures are saved in the `results/` directory.

## 📊 Expected Results

If the code is run with the default parameters in `prmignn/config.py`, you should achieve the following numerical results (averaged over 4 seeds with 10 HF snapshots):

### Main Comparison
| Method | Rel. $L_2$ Error (%) | PDE Residual |
| :--- | :--- | :--- |
| HF-only Baseline | $0.81 \pm 0.03$ | $0.0062$ |
| **PR-MF-GNN (Proposed)** | $\mathbf{0.57 \pm 0.02}$ | $\mathbf{0.0051}$ |

* **Improvement:** $1.4\times$ reduction in relative $L_2$ error and $1.2\times$ improvement in physical consistency (PDE residual).
* **Fidelity Gap:** $1.38\%$ (measured discrepancy between LF and HF solvers).
* **LF-pretrained Only (no HF FT):** $1.53 \pm 0.07\%$ (proves HF fine-tuning successfully corrects the fidelity bias).

### Ablation Study (10 HF Snapshots)
| Variant | Rel. $L_2$ Error (%) |
| :--- | :--- |
| HF-only Baseline | $0.81 \pm 0.03$ |
| No PDE-residual loss | $4.24 \pm 0.66$ |
| Full-network fine-tuning | $0.73 \pm 0.07$ |
| **PR-MF-GNN (full)** | $\mathbf{0.57 \pm 0.02}$ |

### Data Efficiency Highlights
| HF Snapshots | Baseline Error (%) | Proposed Error (%) |
| :--- | :--- | :--- |
| 5 | $1.05 \pm 0.09$ | $0.72 \pm 0.06$ |
| 10 | $0.79 \pm 0.12$ | $0.57 \pm 0.02$ |
| 20 | $0.58 \pm 0.06$ | $0.50 \pm 0.01$ |
| 50 | $0.48 \pm 0.06$ | $0.42 \pm 0.01$ |

## 📜 Citation

If you use this code or build upon this work, please cite the paper:

```bibtex
@article{braiek2026prmfgnn,
  title={{PR-MF-GNN}: Physics-Regularized Multi-Fidelity Graph Neural Networks for Surrogate Modeling under High-Fidelity Data Scarcity},
  author={Braiek, Hamdi and Bel Hadj Hassin, Anis},
  year={2026}
}
