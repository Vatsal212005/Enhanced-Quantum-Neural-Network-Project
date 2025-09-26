# Enhanced Quantum Neural Network Project

This project explores **Quantum Machine Learning (QML)** models for binary classification on the Kaggle *Cardiovascular Disease* dataset. It benchmarks a classical NN baseline, several pure-QML templates, and two hybrid quantum–classical models: **EnhancedQNN** and **DeepHybridQNN**.

> Scripts are VS Code–friendly (non-interactive plotting) and save all artifacts to `./outputs`.

---

## 📌 Features
- **Robust data pipeline**
  - Auto-detect semicolon/comma delimiters for CSV.
  - Cleanups: convert age (days → years); clamp blood pressure to sane ranges.
  - MinMax scaling; stratified train/test split.
- **Models Implemented**
  - **Classical baseline**: small 2‑layer MLP.
  - **QML templates (4‑qubit)**: `BasicEntanglerLayers`, `StronglyEntanglingLayers`, `RandomLayers`, `AmplitudeEmbedding`.
  - **EnhancedQNN (1Q + 3C)**: Quantum front-end → Dense(8) → Dense(4) → Out (logits).
  - **DeepHybridQNN (3Q + 3C)**: Q1→C1→Q2→C2→Q3→C3→Out; supports `--dropout_p`, `--use_layernorm`, `--half_angles`, and MI/PCA-based quantum inputs.
- **Evaluation**
  - Accuracy, confusion matrix PNGs, and classification reports.
  - Run summaries exported as CSVs in `./outputs`.

---

## 📂 Repository Structure
```
├── QML_model_enhanced.py       # Baselines + EnhancedQNN runner
├── QML_model_deep_hybrid.py    # DeepHybridQNN vs EnhancedQNN vs Classical
├── cardio_train.csv            # Sample CSV (Kaggle cardio subset)
├── outputs/                    # Confusion matrices & summary CSVs
└── README.md                   # You are here
```

---

## ⚙️ Setup

### 1) Clone & enter
```bash
git clone https://github.com/Vatsal212005/Enhanced-Quantum-Neural-Network-Project.git
cd Enhanced-Quantum-Neural-Network-Project
```

### 2) Create & activate a virtual environment
```bash
python -m venv venv
# Activate:
# Linux/Mac
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### 3) Install dependencies
If you already have a `requirements.txt`, use it. Otherwise:
```bash
pip install numpy pandas torch scikit-learn matplotlib seaborn pennylane
```

> **Note (Windows/OneDrive):** If you see “git not recognized” or path issues, add `C:\Program Files\Git\cmd` to PATH and restart VS Code/terminal.

---

## 🗂 Dataset
- Uses the Kaggle *Cardiovascular Disease* dataset (`cardio_train.csv`).
- Expected columns:
  - Features (11): `age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active`
  - Target: `cardio` (0/1)
- The scripts auto-switch to `sep=";"` if the file is semicolon-delimited.

---

## ▶️ How to Run

### A) Baselines + EnhancedQNN
Runs classical + multiple QML templates + EnhancedQNN end-to-end.
```bash
python QML_model_enhanced.py \
  --csv ./cardio_train.csv \
  --epochs 20 \
  --enhanced_epochs 15 \
  --lr 5e-3 \
  --enhanced_lr 1e-2 \
  --outdir ./outputs
```
**Outputs**
- `outputs/cm_classical.png`, `cm_qml_basic.png`, `cm_qml_strong.png`, `cm_qml_random.png`, `cm_qml_amp.png`, `cm_enhanced_qnn.png`
- `outputs/summary.csv` (model vs accuracy)

---

### B) DeepHybridQNN vs EnhancedQNN vs Classical
Compares the deeper hybrid architecture with tunable knobs.
```bash
python QML_model_deep_hybrid.py \
  --csv ./cardio_train.csv \
  --model all \
  --epochs_classical 20 \
  --epochs_enhanced 25 \
  --epochs_deep 25 \
  --lr_classical 5e-3 \
  --lr_enhanced 5e-3 \
  --lr_deep 1e-3 \
  --n_qubits 4 \
  --q_reps_enh 8 \
  --q_reps_block 6 \
  --batch_size 256 \
  --outdir ./outputs
```
**Useful flags**
- `--model {all,classical,enhanced,deep}`: choose subset to run.
- `--half_angles`: squash angles to ±π/2 before Q2/Q3 (stability).
- `--dropout_p 0.15`: dropout after C2.
- `--use_layernorm`: LayerNorm after C1/C2.
- `--amp_embed`: Use PCA → `2^n_qubits` comps with AmplitudeEmbedding instead of MI+AngleEmbedding.

**Outputs**
- `outputs/cm_classical.png`, `cm_enhanced_qnn.png`, `cm_deep_hybrid_qnn.png` (depending on `--model`)
- `outputs/summary_deep_vs_enhanced.csv`

---

## 🔧 CLI Reference

### `QML_model_enhanced.py`
| Arg | Type | Default | Description |
|---|---|---:|---|
| `--csv` | str | `./cardio_train.csv` | Path to dataset (auto-detects `;`) |
| `--epochs` | int | `20` | CE epochs for baselines |
| `--enhanced_epochs` | int | `15` | BCE epochs for EnhancedQNN |
| `--lr` | float | `5e-3` | LR for baselines |
| `--enhanced_lr` | float | `1e-2` | LR for EnhancedQNN |
| `--outdir` | str | `./outputs` | Output directory |

### `QML_model_deep_hybrid.py`
| Arg | Type | Default | Description |
|---|---|---:|---|
| `--csv` | str | `./cardio_train.csv` | Path to dataset |
| `--epochs_classical` | int | `20` | Classical epochs |
| `--epochs_enhanced` | int | `25` | EnhancedQNN epochs |
| `--epochs_deep` | int | `25` | DeepHybridQNN epochs |
| `--lr_classical` | float | `5e-3` | LR for classical |
| `--lr_enhanced` | float | `5e-3` | LR for EnhancedQNN |
| `--lr_deep` | float | `1e-3` | LR for DeepHybridQNN |
| `--n_qubits` | int | `4` | Qubits / quantum features |
| `--q_reps_enh` | int | `8` | Depth for EnhancedQNN’s Q block |
| `--q_reps_block` | int | `6` | Depth for each Q block in deep model |
| `--batch_size` | int | `256` | Batch size for hybrid models |
| `--half_angles` | flag | `False` | Use ±π/2 caps before Q2/Q3 |
| `--dropout_p` | float | `0.15` | Dropout after C2 |
| `--use_layernorm` | flag | `False` | LayerNorm after C1/C2 |
| `--model` | choice | `all` | `{all,classical,enhanced,deep}` |
| `--amp_embed` | flag | `False` | Use PCA + AmplitudeEmbedding |
| `--outdir` | str | `./outputs` | Output directory |

---

## 🧪 Reproducibility
- Fixed seed via `set_seed(42)`.
- Deterministic cuDNN flags.
- Saved confusion matrices and CSV summaries for auditability.

---

## ⚠️ Troubleshooting

**“git not recognized” in PowerShell/VS Code**
1) Install Git for Windows. 2) Add `C:\Program Files\Git\cmd` to PATH. 3) Restart terminal.

**Line ending warnings (LF → CRLF)**
- Safe to ignore, but you can standardize:
  ```bash
  git config --global core.autocrlf true
  ```

**Do not commit `venv/`**
- Keep `venv/` in `.gitignore` to avoid massive diffs.

**PennyLane / Torch versions**
- If you hit ABI issues, pin versions (e.g., `pennylane>=0.34`, `torch>=2.1`).

---

## 🧭 Notes & Extensions
- Swap MI top‑k features with PCA + `AmplitudeEmbedding` using `--amp_embed`.
- Try different `n_qubits` (4, 6, 8) and compare capacity vs overfitting.
- Explore different optimizers (`Adam`, `Adagrad`) and schedulers.
- Export trained weights or ONNX for the classical parts if needed.

---

## 📜 License
MIT — feel free to use and modify with attribution.

---

## 🙏 Acknowledgements
- PennyLane for differentiable quantum circuits.
- scikit-learn for preprocessing & metrics.
- PyTorch for training utilities.
