# QML_model_enhanced.py
# VS Code–friendly script for QML baselines + EnhancedQNN on the Kaggle cardio dataset.
# Fixes:
# - Robust CSV loading (auto-detects semicolon delimiter)
# - Uses correct cardio feature/target columns
# - Dynamic classical model input dimension
# - Clear checkpoints + timers + sanity checks
# - Non-interactive plotting; saves figures to ./outputs

import argparse
import os
import random
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

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
    p = argparse.ArgumentParser(description="QML baseline + EnhancedQNN")
    p.add_argument("--csv", type=str, default="./cardio_train.csv",
                   help="Path to cardio_train.csv (semicolon-delimited)")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs for CE models")
    p.add_argument("--enhanced_epochs", type=int, default=15, help="Training epochs for EnhancedQNN (BCE)")
    p.add_argument("--lr", type=float, default=5e-3, help="Learning rate for baselines")
    p.add_argument("--enhanced_lr", type=float, default=1e-2, help="Learning rate for EnhancedQNN")
    p.add_argument("--outdir", type=str, default="./outputs", help="Where to save plots/artifacts")
    return p.parse_args()

# -----------------------------
# Data loading (robust to delimiters)
# -----------------------------
# Kaggle cardio columns
CARDIO_FEATURES = [
    "age","gender","height","weight","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active"
]
CARDIO_TARGET = "cardio"

def load_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Pass with --csv.")

    # Try comma first
    df = pd.read_csv(csv_path)
    # If it's a single messy column, try semicolon
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path, sep=";")

    print("Loaded CSV with shape:", df.shape)
    print(df.head().to_string(index=False))

    # Validate expected columns
    missing = [c for c in CARDIO_FEATURES + [CARDIO_TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}. Present columns: {list(df.columns)}")

    X = df[CARDIO_FEATURES].values
    y = df[CARDIO_TARGET].values
    return X, y

# -----------------------------
# Classical helpers
# -----------------------------
class ClassicalModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        hidden = max(6, in_dim)  # simple heuristic so it scales with feature count
        self.layer1 = nn.Linear(in_dim, hidden)
        self.layer2 = nn.Linear(hidden, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x); x = self.relu(x)
        x = self.layer2(x)
        return self.softmax(x)

# -----------------------------
# Quantum helpers
# -----------------------------
def make_qdevice(n_qubits: int):
    return qml.device("default.qubit", wires=n_qubits)

def qc_basic(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qc_strong(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qc_random(inputs, weights, n_qubits):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.RandomLayers(weights, wires=range(n_qubits), seed=42)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qc_amp(inputs, weights, n_qubits):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QMLModelTemplate(nn.Module):
    def __init__(self, qnode, weight_shapes, n_qubits):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.clayer = nn.Linear(n_qubits, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.qlayer(x)
        x = self.clayer(x)
        return self.softmax(x)

# -----------------------------
# EnhancedQNN (BCE, binary output)
# -----------------------------
class EnhancedQNN(nn.Module):
    def __init__(self, qnode, weight_shapes, n_qubits):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc1 = nn.Linear(n_qubits, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc_out = nn.Linear(4, 1)  # logits
    def forward(self, x):
        x = self.q_layer(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)  # NOTE: no sigmoid here


# -----------------------------
# Training / Eval utilities
# -----------------------------
def train_ce_model(model, Xtr, ytr, Xte, yte, epochs=20, lr=5e-3, label="CE"):
    print(f"-> [{label}] compiling QNode (if any) & starting training for {epochs} epochs...")
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
    return acc, cm, report, preds_te

def train_enhanced_qnn(model, Xtr, ytr_bin, Xte, yte_bin, epochs=25, lr=5e-3, batch_size=256):
    print("-> [EnhancedQNN] compiling QNode & starting training...")
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # numerically stable

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

        # Eval
        model.eval()
        with torch.no_grad():
            logits_te = model(Xte)
            preds = (torch.sigmoid(logits_te) > 0.5).float()
            acc = (preds.eq(yte_bin).sum()/yte_bin.size(0)).item()

        print(f"[EnhancedQNN] Epoch {ep+1:02d}/{epochs} | Loss={running/len(ds):.4f} | Test Acc={acc:.4f}")

    with torch.no_grad():
        logits_te = model(Xte)
        proba = torch.sigmoid(logits_te)
        preds_bin = (proba > 0.5).float()
        acc = (preds_bin.eq(yte_bin).sum()/yte_bin.size(0)).item()
        y_true = yte_bin.cpu().numpy().ravel().astype(int)
        y_pred = preds_bin.cpu().numpy().ravel().astype(int)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
    print(f"-> [EnhancedQNN] finished in {time.time()-t0:.2f}s")
    return acc, cm, report, preds_bin

# -----------------------------
# Plot helpers
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

    # ---------------- Preprocessing ----------------
    from sklearn.feature_selection import mutual_info_classif

    print("== Cleaning & scaling ==")
    t0 = time.time()

    # Convert age (days) -> years
    age_idx = CARDIO_FEATURES.index("age")
    X[:, age_idx] = X[:, age_idx] / 365.25

    # Clamp absurd blood pressures
    ap_hi_idx = CARDIO_FEATURES.index("ap_hi")
    ap_lo_idx = CARDIO_FEATURES.index("ap_lo")
    X[:, ap_hi_idx] = np.clip(X[:, ap_hi_idx], 60, 240)
    X[:, ap_lo_idx] = np.clip(X[:, ap_lo_idx], 30, 150)

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("Training data shape:", X_train.shape, "Testing data shape:", X_test.shape)
    print(f"Preprocessing done in {time.time()-t0:.2f}s")

    # Pick best 4 features for quantum via mutual information
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    topk_idx = np.argsort(mi)[-4:]
    topk_idx = np.sort(topk_idx)
    topk_names = [CARDIO_FEATURES[i] for i in topk_idx]
    print("Top-4 features for quantum:", topk_names)

    X_train_q = X_train[:, topk_idx]
    X_test_q  = X_test[:, topk_idx]

    # ---------------- Torch tensors ----------------
    in_dim = X_train.shape[1]  # 11 for cardio
    X_train_torch_4 = torch.tensor(X_train_q, dtype=torch.float32)
    X_test_torch_4  = torch.tensor(X_test_q,  dtype=torch.float32)
    y_train_torch   = torch.tensor(y_train,   dtype=torch.long)
    y_test_torch    = torch.tensor(y_test,    dtype=torch.long)

    X_train_classical = torch.tensor(X_train, dtype=torch.float32)
    X_test_classical  = torch.tensor(X_test,  dtype=torch.float32)

    y_train_bin = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    y_test_bin  = torch.tensor(y_test,  dtype=torch.float32).view(-1,1)

    # ---------------- Sanity checks ----------------
    print("\n=== Sanity Checks ===")
    print("Label balance (train):", np.bincount(y_train))
    print("Label balance (test): ", np.bincount(y_test))
    try:
        cm = ClassicalModel(in_dim)
        out = cm(X_train_classical[:5])
        print("Classical forward OK:", tuple(out.shape))
    except Exception as e:
        print("Classical forward FAILED:", e)
    try:
        n_qubits = 4
        dev = make_qdevice(n_qubits)
        @qml.qnode(dev, interface="torch")
        def qnode_basic(inputs, weights):
            return qc_basic(inputs, weights, n_qubits)
        w_basic = {"weights": (5, n_qubits)}
        qm = QMLModelTemplate(qnode=qnode_basic, weight_shapes=w_basic, n_qubits=n_qubits)
        out = qm(X_train_torch_4[:5])
        print("Quantum forward OK (BasicEntanglerLayers):", tuple(out.shape))
    except Exception as e:
        print("Quantum forward FAILED:", e)
    print("=== End Sanity Checks ===\n")

    # ---------------- QNodes & weight shapes ----------------
    @qml.qnode(dev, interface="torch")
    def qnode_strong(inputs, weights):
        return qc_strong(inputs, weights, n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode_random(inputs, weights):
        return qc_random(inputs, weights, n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode_amp(inputs, weights):
        return qc_amp(inputs, weights, n_qubits)

    w_strong = {"weights": (8, n_qubits, 3)}  # Deeper StronglyEntanglingLayers
    w_random = {"weights": (5, n_qubits)}
    w_amp    = {"weights": (5, n_qubits, 3)}


    # ---------------- Baselines ----------------
    print("\n== Classical baseline (all features) ==")
    classical_model = ClassicalModel(in_dim)
    acc_class, cm_class, rpt_class, preds_class = train_ce_model(
        classical_model, X_train_classical, y_train_torch, X_test_classical, y_test_torch,
        epochs=args.epochs, lr=args.lr, label="Classical"
    )
    save_confusion_matrix(cm_class, "Classical Model", os.path.join(args.outdir, "cm_classical.png"))
    print(rpt_class)

    print("\n== QML (BasicEntanglerLayers, 4 qubits, AngleEmbedding) ==")
    model_basic = QMLModelTemplate(qnode=qnode_basic, weight_shapes=w_basic, n_qubits=n_qubits)
    acc_basic, cm_basic, rpt_basic, _ = train_ce_model(
        model_basic, X_train_torch_4, y_train_torch, X_test_torch_4, y_test_torch,
        epochs=args.epochs, lr=args.lr, label="QML-Basic"
    )
    save_confusion_matrix(cm_basic, "QML BasicEntanglerLayers", os.path.join(args.outdir, "cm_qml_basic.png"))
    print(rpt_basic)

    print("\n== QML (StronglyEntanglingLayers, 4 qubits, AngleEmbedding) ==")
    model_strong = QMLModelTemplate(qnode=qnode_strong, weight_shapes=w_strong, n_qubits=n_qubits)
    acc_strong, cm_strong, rpt_strong, _ = train_ce_model(
        model_strong, X_train_torch_4, y_train_torch, X_test_torch_4, y_test_torch,
        epochs=args.epochs, lr=args.lr, label="QML-Strong"
    )
    save_confusion_matrix(cm_strong, "QML StronglyEntanglingLayers", os.path.join(args.outdir, "cm_qml_strong.png"))
    print(rpt_strong)

    print("\n== QML (RandomLayers, 4 qubits, AngleEmbedding) ==")
    model_random = QMLModelTemplate(qnode=qnode_random, weight_shapes=w_random, n_qubits=n_qubits)
    acc_random, cm_random, rpt_random, _ = train_ce_model(
        model_random, X_train_torch_4, y_train_torch, X_test_torch_4, y_test_torch,
        epochs=args.epochs, lr=args.lr, label="QML-Random"
    )
    save_confusion_matrix(cm_random, "QML RandomLayers", os.path.join(args.outdir, "cm_qml_random.png"))
    print(rpt_random)

    print("\n== QML (StronglyEntanglingLayers + AmplitudeEmbedding, 4 qubits) ==")
    model_amp = QMLModelTemplate(qnode=qnode_amp, weight_shapes=w_amp, n_qubits=n_qubits)
    acc_amp, cm_amp, rpt_amp, _ = train_ce_model(
        model_amp, X_train_torch_4, y_train_torch, X_test_torch_4, y_test_torch,
        epochs=args.epochs, lr=args.lr, label="QML-Amp"
    )
    save_confusion_matrix(cm_amp, "QML AmplitudeEmbedding", os.path.join(args.outdir, "cm_qml_amp.png"))
    print(rpt_amp)

    # ---------------- EnhancedQNN ----------------
    print("\n== EnhancedQNN (Quantum front-end + Dense(8)->Dense(4)->Out->Sigmoid) ==")
    enhanced = EnhancedQNN(qnode=qnode_strong, weight_shapes=w_strong, n_qubits=n_qubits)
    acc_enh, cm_enh, rpt_enh, _ = train_enhanced_qnn(
        enhanced, X_train_torch_4, y_train_bin, X_test_torch_4, y_test_bin,
        epochs=args.enhanced_epochs, lr=args.enhanced_lr
    )
    save_confusion_matrix(cm_enh, "EnhancedQNN", os.path.join(args.outdir, "cm_enhanced_qnn.png"))
    print(rpt_enh)

    # ---------------- Summary Table ----------------
    summary = pd.DataFrame([
        {"Model": "Classical", "Accuracy": acc_class},
        {"Model": "QML BasicEntanglerLayers", "Accuracy": acc_basic},
        {"Model": "QML StronglyEntanglingLayers", "Accuracy": acc_strong},
        {"Model": "QML RandomLayers", "Accuracy": acc_random},
        {"Model": "QML AmplitudeEmbedding", "Accuracy": acc_amp},
        {"Model": "EnhancedQNN", "Accuracy": acc_enh},
    ]).sort_values("Accuracy", ascending=False)

    out_csv = os.path.join(args.outdir, "summary.csv")
    summary.to_csv(out_csv, index=False)
    print("\nSummary (saved to {}):".format(out_csv))
    print(summary.to_string(index=False))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
