# QML_model_deep_hybrid.py
# Compare EnhancedQNN (1Q+3C) vs DeepHybridQNN (3Q+3C) with the same preprocessing.
# - Robust CSV loader (auto-detect semicolon)
# - Cardio dataset cleaning (age->years, BP clamping)
# - MI-based top-k feature selection for quantum input (k = n_qubits)
# - BCEWithLogitsLoss for hybrid models
# - Clear checkpoints and confusion matrices saved to ./outputs
# - NEW: --model {all,classical,enhanced,deep}, --half_angles, --dropout_p, --use_layernorm

import argparse
import os
import random
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import mutual_info_classif

import pennylane as qml

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Deep Hybrid QNN (3Q+3C) comparison")
    p.add_argument("--csv", type=str, default="./cardio_train.csv",
                   help="Path to cardio_train.csv (semicolon-delimited)")
    # Training budgets
    p.add_argument("--epochs_classical", type=int, default=20, help="Epochs for classical baseline")
    p.add_argument("--epochs_enhanced", type=int, default=25, help="Epochs for EnhancedQNN (1Q+3C)")
    p.add_argument("--epochs_deep", type=int, default=25, help="Epochs for DeepHybridQNN (3Q+3C)")
    # Learning rates
    p.add_argument("--lr_classical", type=float, default=5e-3, help="LR for classical baseline")
    p.add_argument("--lr_enhanced", type=float, default=5e-3, help="LR for EnhancedQNN")
    p.add_argument("--lr_deep", type=float, default=1e-3, help="LR for DeepHybridQNN (lower helps stability)")
    # Architecture knobs
    p.add_argument("--n_qubits", type=int, default=4, help="Number of qubits / quantum features (top-k)")
    p.add_argument("--q_reps_enh", type=int, default=8, help="Repetitions for EnhancedQNN's Q block")
    p.add_argument("--q_reps_block", type=int, default=6, help="Repetitions for each Q block in DeepHybridQNN")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for hybrid models")
    # Stability & controls
    p.add_argument("--half_angles", action="store_true",
                   help="Use ±π/2 instead of ±π before Q2 and Q3 to reduce saturation")
    p.add_argument("--dropout_p", type=float, default=0.15, help="Dropout after C2 in deep model")
    p.add_argument("--use_layernorm", action="store_true",
                   help="Add LayerNorm after C1 and C2 (before nonlinearity) for deep model")
    # What to run
    p.add_argument("--model", type=str, default="all",
                   choices=["all", "classical", "enhanced", "deep"],
                   help="Run which model(s)")
    p.add_argument("--amp_embed", action="store_true",
    help="Use PCA+AmplitudeEmbedding instead of MI+AngleEmbedding")

    p.add_argument("--outdir", type=str, default="./outputs", help="Where to save plots/artifacts")
    return p.parse_args()

# -----------------------------
# Data loading (robust to delimiters)
# -----------------------------
CARDIO_FEATURES = [
    "age","gender","height","weight","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active"
]
CARDIO_TARGET = "cardio"

def load_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Pass with --csv.")
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:  # likely semicolon-delimited
        df = pd.read_csv(csv_path, sep=";")

    print("Loaded CSV with shape:", df.shape)
    print(df.head().to_string(index=False))

    missing = [c for c in CARDIO_FEATURES + [CARDIO_TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}. Present columns: {list(df.columns)}")

    X = df[CARDIO_FEATURES].values
    y = df[CARDIO_TARGET].values
    return X, y

# -----------------------------
# Classical baseline
# -----------------------------
class ClassicalModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        hidden = max(6, in_dim)
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.softmax(x)

def train_ce_model(model, Xtr, ytr, Xte, yte, epochs=20, lr=5e-3, label="Classical"):
    print(f"-> [{label}] training for {epochs} epochs...")
    t0 = time.time()
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        opt.zero_grad()
        out = model(Xtr)
        loss = loss_fn(out, ytr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            preds = out.argmax(dim=1)
            acc_tr = (preds.eq(ytr).sum()/ytr.size(0)).item()
        print(f"[{label}] Epoch {ep+1:02d}/{epochs} | Loss={loss.item():.4f} | Train Acc={acc_tr:.4f}")
    with torch.no_grad():
        out_te = model(Xte)
        preds_te = out_te.argmax(dim=1)
        acc = (preds_te.eq(yte).sum()/yte.size(0)).item()
        cm = confusion_matrix(yte.cpu().numpy(), preds_te.cpu().numpy())
        report = classification_report(yte.cpu().numpy(), preds_te.cpu().numpy(), digits=4)
    print(f"-> [{label}] finished in {time.time()-t0:.2f}s")
    return acc, cm, report

# -----------------------------
# Quantum helpers
# -----------------------------
def make_qdevice(n_qubits: int):
    return qml.device("default.qubit", wires=n_qubits)

def q_strong(inputs, weights, n_qubits, amp_embed=False):
    if amp_embed:
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    else:
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def angle_squash(x: torch.Tensor, half_angles: bool) -> torch.Tensor:
    # keep angles stable for AngleEmbedding
    cap = torch.pi / 2 if half_angles else torch.pi
    return torch.tanh(x) * cap

# -----------------------------
# EnhancedQNN (1Q + 3C) — logits out
# -----------------------------
class EnhancedQNN(nn.Module):
    def __init__(self, qnode, weight_shapes, n_qubits):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc1 = nn.Linear(n_qubits, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc_out = nn.Linear(4, 1)  # logits

    def forward(self, x):
        x = self.q_layer(x)            # (B, nq)
        x = torch.relu(self.fc1(x))    # (B, 8)
        x = torch.relu(self.fc2(x))    # (B, 4)
        return self.fc_out(x)          # (B, 1)

# -----------------------------
# DeepHybridQNN (3Q + 3C) — logits out
# -----------------------------
class DeepHybridQNN(nn.Module):
    """
    Q1 -> C1 (nq) -> Q2 -> C2 (nq) -> Q3 -> C3 (4) -> Out (1)
    - squashes activations to angle range before Q2 and Q3
    - optional LayerNorm after C1/C2 to stabilize distributions
    - logits output (use BCEWithLogitsLoss)
    """
    def __init__(self, n_qubits, qnode1, w1, qnode2, w2, qnode3, w3, p_drop=0.15, half_angles=False, use_layernorm=False):
        super().__init__()
        self.nq = n_qubits
        self.half_angles = half_angles
        self.use_layernorm = use_layernorm

        self.q1 = qml.qnn.TorchLayer(qnode1, w1)
        self.q2 = qml.qnn.TorchLayer(qnode2, w2)
        self.q3 = qml.qnn.TorchLayer(qnode3, w3)

        self.c1 = nn.Linear(n_qubits, n_qubits)
        self.c2 = nn.Linear(n_qubits, n_qubits)
        self.c3 = nn.Linear(n_qubits, 4)
        self.out = nn.Linear(4, 1)

        self.ln1 = nn.LayerNorm(n_qubits) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_qubits) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x):
        # Block 1
        x = self.q1(x)                     # (B, nq)
        x = self.ln1(self.c1(x))           # (B, nq)
        x = torch.relu(x)
        x = angle_squash(x, self.half_angles)  # prepare for Q2

        # Block 2
        x = self.q2(x)                     # (B, nq)
        x = self.ln2(self.c2(x))           # (B, nq)
        x = torch.relu(x)
        x = self.drop(x)
        x = angle_squash(x, self.half_angles)  # prepare for Q3

        # Block 3
        x = self.q3(x)                     # (B, nq)
        x = torch.relu(self.c3(x))         # (B, 4)
        x = self.out(x)                    # (B, 1) logits
        return x

# -----------------------------
# Binary training loop (logits)
# -----------------------------
def train_logits_binary_model(model, Xtr, ytr_bin, Xte, yte_bin, epochs=25, lr=5e-3, batch_size=256, label="Hybrid"):
    print(f"-> [{label}] compiling QNodes & starting training for {epochs} epochs...")
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = torch.utils.data.TensorDataset(Xtr, ytr_bin)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            logits_te = model(Xte)
            preds = (torch.sigmoid(logits_te) > 0.5).float()
            acc = (preds.eq(yte_bin).sum()/yte_bin.size(0)).item()
        print(f"[{label}] Epoch {ep+1:02d}/{epochs} | Loss={running/len(ds):.4f} | Test Acc={acc:.4f}")

    with torch.no_grad():
        logits_te = model(Xte)
        proba = torch.sigmoid(logits_te)
        preds_bin = (proba > 0.5).float()
        acc = (preds_bin.eq(yte_bin).sum()/yte_bin.size(0)).item()
        y_true = yte_bin.cpu().numpy().ravel().astype(int)
        y_pred = preds_bin.cpu().numpy().ravel().astype(int)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)

    print(f"-> [{label}] finished in {time.time()-t0:.2f}s")
    return acc, cm, report

# -----------------------------
# Utils
# -----------------------------
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def save_confusion_matrix(cm, title, outpath):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")

# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(42)
    args = parse_args()
    ensure_outdir(args.outdir)

    print("== Loading dataset ==")
    X, y = load_dataset(args.csv)

    # Cleaning & split
    print("== Cleaning & scaling ==")
    t0 = time.time()

    # age: days -> years
    age_idx = CARDIO_FEATURES.index("age")
    X[:, age_idx] = X[:, age_idx] / 365.25

    # clamp blood pressures
    ap_hi_idx = CARDIO_FEATURES.index("ap_hi")
    ap_lo_idx = CARDIO_FEATURES.index("ap_lo")
    X[:, ap_hi_idx] = np.clip(X[:, ap_hi_idx], 60, 240)
    X[:, ap_lo_idx] = np.clip(X[:, ap_lo_idx], 30, 150)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training data shape:", X_train.shape, "Testing data shape:", X_test.shape)
    print(f"Preprocessing done in {time.time()-t0:.2f}s")

    # MI-based top-k feature selection for quantum (k = n_qubits)
    k = args.n_qubits
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    topk_idx = np.argsort(mi)[-k:]
    topk_idx = np.sort(topk_idx)
    topk_names = [CARDIO_FEATURES[i] for i in topk_idx]
    print(f"Top-{k} features for quantum:", topk_names)

    X_train_q = X_train[:, topk_idx]
    X_test_q  = X_test[:, topk_idx]

    # Tensors
    in_dim = X_train.shape[1]
    X_train_torch_k = torch.tensor(X_train_q, dtype=torch.float32)
    X_test_torch_k  = torch.tensor(X_test_q,  dtype=torch.float32)
    y_train_torch   = torch.tensor(y_train,   dtype=torch.long)
    y_test_torch    = torch.tensor(y_test,    dtype=torch.long)

    X_train_classical = torch.tensor(X_train, dtype=torch.float32)
    X_test_classical  = torch.tensor(X_test,  dtype=torch.float32)

    y_train_bin = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    y_test_bin  = torch.tensor(y_test,  dtype=torch.float32).view(-1,1)

    # Sanity checks
    print("\n=== Sanity Checks ===")
    print("Label balance (train):", np.bincount(y_train))
    print("Label balance (test): ", np.bincount(y_test))
    try:
        _ = ClassicalModel(in_dim)(X_train_classical[:5])
        print("Classical forward OK")
    except Exception as e:
        print("Classical forward FAILED:", e)
    print("=== End Sanity Checks ===\n")

    # Device & QNodes
    n_qubits = args.n_qubits
    dev = make_qdevice(n_qubits)

    # EnhancedQNN single Q block
    @qml.qnode(dev, interface="torch")
    def qnode_enh(inputs, weights):
        return q_strong(inputs, weights, n_qubits, amp_embed=args.amp_embed)
    w_enh = {"weights": (args.q_reps_enh, n_qubits, 3)}

    # DeepHybridQNN three Q blocks
    @qml.qnode(dev, interface="torch")
    def qnode_block1(inputs, weights):
        return q_strong(inputs, weights, n_qubits, amp_embed=args.amp_embed)
    @qml.qnode(dev, interface="torch")
    def qnode_block2(inputs, weights):
        return q_strong(inputs, weights, n_qubits, amp_embed=args.amp_embed)
    @qml.qnode(dev, interface="torch")
    def qnode_block3(inputs, weights):
        return q_strong(inputs, weights, n_qubits, amp_embed=args.amp_embed)
    w_block1 = {"weights": (args.q_reps_block, n_qubits, 3)}
    w_block2 = {"weights": (args.q_reps_block, n_qubits, 3)}
    w_block3 = {"weights": (args.q_reps_block, n_qubits, 3)}

    from sklearn.decomposition import PCA

    if args.amp_embed:
        # PCA → 2^n features
        target_dim = 2**args.n_qubits
        print(f"Using PCA → {target_dim} comps for AmplitudeEmbedding")
        pca = PCA(n_components=target_dim, random_state=42)
        X_train_q = pca.fit_transform(X_train)
        X_test_q  = pca.transform(X_test)
    else:
        # MI top-k as before
        k = args.n_qubits
        mi = mutual_info_classif(X_train, y_train, random_state=42)
        topk_idx = np.argsort(mi)[-k:]
        topk_idx = np.sort(topk_idx)
        topk_names = [CARDIO_FEATURES[i] for i in topk_idx]
        print(f"Top-{k} features for quantum:", topk_names)
        X_train_q = X_train[:, topk_idx]
        X_test_q  = X_test[:, topk_idx]

    # ---------------- Run selection ----------------
    summary_rows = []

    if args.model in ("all", "classical"):
        print("\n== Classical baseline (all features) ==")
        classical_model = ClassicalModel(in_dim)
        acc_class, cm_class, rpt_class = train_ce_model(
            classical_model, X_train_classical, y_train_torch, X_test_classical, y_test_torch,
            epochs=args.epochs_classical, lr=args.lr_classical, label="Classical"
        )
        save_confusion_matrix(cm_class, "Classical Model", os.path.join(args.outdir, "cm_classical.png"))
        print(rpt_class)
        summary_rows.append({"Model": "Classical", "Accuracy": acc_class})

    if args.model in ("all", "enhanced"):
        print("\n== EnhancedQNN (1Q + 3C) ==")
        enh_model = EnhancedQNN(qnode=qnode_enh, weight_shapes=w_enh, n_qubits=n_qubits)
        acc_enh, cm_enh, rpt_enh = train_logits_binary_model(
            enh_model, X_train_torch_k, y_train_bin, X_test_torch_k, y_test_bin,
            epochs=args.epochs_enhanced, lr=args.lr_enhanced, batch_size=args.batch_size, label="EnhancedQNN"
        )
        save_confusion_matrix(cm_enh, "EnhancedQNN", os.path.join(args.outdir, "cm_enhanced_qnn.png"))
        print(rpt_enh)
        summary_rows.append({"Model": "EnhancedQNN (1Q+3C)", "Accuracy": acc_enh})

    if args.model in ("all", "deep"):
        print("\n== DeepHybridQNN (3Q + 3C) ==")
        deep_model = DeepHybridQNN(
            n_qubits=n_qubits,
            qnode1=qnode_block1, w1=w_block1,
            qnode2=qnode_block2, w2=w_block2,
            qnode3=qnode_block3, w3=w_block3,
            p_drop=args.dropout_p,
            half_angles=args.half_angles,
            use_layernorm=args.use_layernorm
        )
        acc_deep, cm_deep, rpt_deep = train_logits_binary_model(
            deep_model, X_train_torch_k, y_train_bin, X_test_torch_k, y_test_bin,
            epochs=args.epochs_deep, lr=args.lr_deep, batch_size=args.batch_size, label="DeepHybridQNN"
        )
        save_confusion_matrix(cm_deep, "DeepHybridQNN", os.path.join(args.outdir, "cm_deep_hybrid_qnn.png"))
        print(rpt_deep)
        summary_rows.append({"Model": "DeepHybridQNN (3Q+3C)", "Accuracy": acc_deep})

    # ---------------- Summary ----------------
    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values("Accuracy", ascending=False)
        out_csv = os.path.join(args.outdir, "summary_deep_vs_enhanced.csv")
        summary.to_csv(out_csv, index=False)
        print("\nSummary (saved to {}):".format(out_csv))
        print(summary.to_string(index=False))
    else:
        print("No models were run. Check --model arg.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
