# Enhanced Quantum Neural Network Project

A practical, VS Code–friendly playground for **Quantum Machine Learning (QML)** on the Kaggle **Cardiovascular Disease** dataset.  
You can compare a compact **classical neural net**, several **pure QML circuits**, a concise **EnhancedQNN (1Q+3C)**, and a deeper **DeepHybridQNN (3Q+3C)**—all on the **same preprocessing** for a fair comparison.

> All artifacts (plots, CSVs) are saved to `./outputs`. Scripts use non-interactive Matplotlib backends.

---

## 🎯 Objectives
- Provide **clean baselines** (classical + multiple QML templates).
- Demonstrate **hybrid quantum–classical** models where a quantum front‑end feeds a classical head.
- Keep experiments **reproducible** (fixed seeds) and **portable** (PyTorch + PennyLane).

---

## 📦 Repository Layout
```
├── QML_model_enhanced.py       # Baselines + EnhancedQNN runner
├── QML_model_deep_hybrid.py    # DeepHybridQNN vs EnhancedQNN vs Classical
├── cardio_train.csv            # Sample CSV (Kaggle cardio subset)
├── outputs/                    # Confusion matrices (.png) & summaries (.csv)
└── README.md                   # This file
```

---

## 🧠 Dataset & Preprocessing
**Dataset**: Kaggle *Cardiovascular Disease* training CSV (`cardio_train.csv`).

**Features (11)**: `age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active`  
**Target (1)**: `cardio` (0/1)

**Preprocessing (shared by all experiments):**
- **Delimiter auto-detect**: try comma; if the CSV collapses to one column, retry with `sep=";"`.
- **Sanity fixes**: `age` from *days → years*; clamp blood pressure to `ap_hi∈[60,240]`, `ap_lo∈[30,150]`.
- **Split & scale**: stratified 80/20 split; `MinMaxScaler` on features.

**Quantum input builders (choose one):**
- **MI top‑k (default)**: select the top `k = n_qubits` features using mutual information (supervised, simple).
- **PCA → AmplitudeEmbedding (`--amp_embed`)**: project to `2^n_qubits` principal components and feed as a normalized amplitude vector.

---

## 🧩 Models — What They Are & How They Work

### 1) Classical Baseline (small MLP)
**Why**: establish a reference using *all 11 features*.  
**Architecture**:
```
Input(11) → Linear(11→H) → ReLU → Linear(H→2) → Softmax
```
- `H = max(6, in_dim)` (scales with feature count).
- **Loss**: CrossEntropyLoss (multi-class over {0,1}).
- **Optimizer**: Adagrad (simple & stable for small nets).

---

### 2) Pure QML Templates (4‑qubit heads)
Each QML model encodes a length‑`n_qubits` vector `x` into a quantum state, applies a parametrized circuit, and **measures Pauli‑Z expectations** on each wire:
\
`f_θ(x) = [⟨Z₀⟩, ⟨Z₁⟩, …, ⟨Z_{n_q-1}⟩]`  
The result is then mapped to 2 logits via a linear layer + Softmax.

Common readout head:
```
QuantumExpectations(n_qubits) → Linear(n_qubits→2) → Softmax
```

**Templates included:**
- **BasicEntanglerLayers**  
  - *Encoding*: `AngleEmbedding(x)` (rotations per qubit).  
  - *Ansatz*: layers of single‑qubit rotations + ring entanglers.  
  - *Use case*: fast baseline with modest expressive power.
- **StronglyEntanglingLayers**  
  - Deeper circuit with stronger entanglement patterns.  
  - *Effect*: higher capacity at the cost of train stability; depth set via `weights` repetitions.
- **RandomLayers**  
  - Pseudorandom parametrized layers (seeded).  
  - *Use case*: stress‑test whether structured entanglement matters for your data.
- **AmplitudeEmbedding**  
  - *Encoding*: normalized amplitude vector (dimension `2^n_qubits`).  
  - *When `--amp_embed` is enabled*: inputs come from PCA to match amplitude dimension.  
  - *Effect*: global, compact encoding that can capture more variance at small qubit counts.

**Training**: like the classical baseline (CrossEntropyLoss), but the forward pass is differentiable through the QNode (PennyLane `TorchLayer`).

---

### 3) EnhancedQNN (1Q + 3C) — *compact hybrid*
**Idea**: use a **single quantum block** to extract a non-linear representation, then process with a small classical head for binary logits.

**Block diagram**:
```
x (k dims)
 └─→ [QBlock: Angle/Amplitude + StronglyEntanglingLayers] → z ∈ ℝ^{n_qubits}
      → ReLU → Linear(nq→8) → ReLU → Linear(8→4) → Linear(4→1) (logit)
```
- **Embedding**: default `AngleEmbedding` on MI top‑k features; optional `AmplitudeEmbedding` with PCA.  
- **Loss**: `BCEWithLogitsLoss` (numerically stable).  
- **Why it works**: the quantum layer can reshape feature interactions; the classical head aggregates them into a decision boundary with low parameter count.

---

### 4) DeepHybridQNN (3Q + 3C) — *stacked hybrid with stabilizers*
**Idea**: alternate **three** quantum blocks with classical transforms, allowing information to be iteratively re‑encoded and re‑entangled.

**Macro architecture**:
```
x → Q1 → C1 → (angle_squash) → Q2 → C2 → Dropout → (angle_squash) → Q3 → C3(→4) → Out(→1 logit)
```
- **Q blocks**: each is `Angle/AmplitudeEmbedding + StronglyEntanglingLayers` (depth via `--q_reps_block`).  
- **C blocks**: linear transforms matching `n_qubits` width (`C1`, `C2`) + a `C3` projection to 4 dims.  
- **Stabilizers**:
  - `angle_squash`: `x ← tanh(x) * cap`, with `cap ∈ {π, π/2}` (use `--half_angles` for ±π/2) to keep angles in a reasonable range and reduce saturation.
  - Optional **LayerNorm** after `C1`/`C2` (`--use_layernorm`) to stabilize distributional shifts.
  - **Dropout** after `C2` (`--dropout_p`) to regularize.  
- **Loss**: `BCEWithLogitsLoss`.

**Why stack?**  
Alternating Q/C blocks lets the model *re-embed* intermediate features into the quantum state space, potentially capturing higher‑order interactions that a single Q block might miss.

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
- `--model {all,classical,enhanced,deep}`: run a subset.
- `--half_angles`: use ±π/2 caps before Q2/Q3 (`angle_squash`) to reduce saturation.
- `--dropout_p 0.15`: dropout after C2.
- `--use_layernorm`: LayerNorm after C1/C2.
- `--amp_embed`: switch to PCA + `AmplitudeEmbedding` instead of MI + `AngleEmbedding`.

**Outputs**
- `outputs/cm_classical.png`, `cm_enhanced_qnn.png`, `cm_deep_hybrid_qnn.png` (depending on `--model`)
- `outputs/summary_deep_vs_enhanced.csv`

---

## 🔬 Training & Evaluation Details
- **Optimizers**: `Adagrad` for CE baselines; `Adam` for hybrid BCE models.
- **Losses**:  
  - CE models (classical & QML templates): `CrossEntropyLoss` (2‑class).  
  - Hybrid models (Enhanced/Deep): `BCEWithLogitsLoss` with sigmoid at eval time.
- **Batching**: DataLoader with `batch_size` (default 256 for hybrids).
- **Metrics**: Accuracy, confusion matrix (PNG), full `classification_report` (printed).
- **Reproducibility**: `set_seed(42)` (+ deterministic cuDNN flags).

---

## 📈 Artifacts
All saved to `./outputs`:
- `cm_*.png` — confusion matrices for each model variant.
- `summary.csv` — accuracy leaderboard for baselines + EnhancedQNN.
- `summary_deep_vs_enhanced.csv` — accuracy comparison for classical vs EnhancedQNN vs DeepHybridQNN.

---

## 🧭 Extensions / Ideas
- Try different `n_qubits` (e.g., 6 or 8) and compare capacity vs overfitting.
- Explore learning-rate schedules, weight decay, or gradient clipping in hybrids.
- Swap `StronglyEntanglingLayers` with custom ansätze or hardware-efficient patterns.
- Evaluate calibration (e.g., reliability diagrams) for the logits from hybrid heads.
- Add ROC‑AUC/PR‑AUC for class‑imbalance sensitivity.

---

## 📜 License
MIT — free to use and modify with attribution.
