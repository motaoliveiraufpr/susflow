# -*- coding: utf-8 -*-
"""
> [!AML-DOC-FILE]
@file run_temporal.py
@description Two reviewer-requested additions that the per-month ablation
             cannot cover: (a) genuinely spatio-temporal baselines, and
             (b) class-weighted training for the neural models.
@module sibgrapi.rebuttal
@exports main
@created 2026-08-04
@context Answers R3 ("the GCN used is purely spatial ... there is no recurrent
         or temporal convolution module (like in EvolveGCN or T-GCN) ... adding
         a true spatio-temporal baseline would better align with the title")
         and R2 ("severe class imbalance ... affects every neural network").

         On R2: the published comparison was asymmetric. Logistic Regression
         and Random Forest were fitted with class_weight="balanced" while the
         GCN and the MLP minimized an unweighted cross-entropy, so the neural
         models were handicapped on exactly the 5-instance minority class the
         comparison turns on. Every model here is run both ways.

         MODELS
         - MLP      : no graph, no time      (control)
         - GCN      : graph, no time         (the paper's model)
         - GRU      : no graph, time         (isolates the temporal component)
         - T-GCN    : graph + time           (GCNConv -> GRUCell, per node)

         The T-GCN follows the T-GCN/EvolveGCN family: a spatial encoder is
         applied to each monthly snapshot and its output drives a recurrent
         cell whose hidden state is carried across months, per node. Node rows
         are aligned to a fixed municipality order (pipeline.load_aligned),
         without which the carried state would be meaningless.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch_geometric.nn import GCNConv

from pipeline import load_aligned

N_RUNS = 30
HIDDEN = 16
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
N_FEATURES = 7
N_CLASSES = 3
TRAIN_MONTHS = 9          # months 1-9 train, 10-12 test


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def balanced_weights(train_data):
    """
    > [!AML-DOC-UNIT]
    Reproduce sklearn's class_weight="balanced" for cross-entropy:
    w_c = n_samples / (n_classes * count_c).

    @param train_data: list of Data holding the training months.
    @returns FloatTensor of shape [N_CLASSES].
    """
    y = torch.cat([d.y for d in train_data])
    counts = torch.bincount(y, minlength=N_CLASSES).float()
    counts[counts == 0] = 1.0                      # avoid div-by-zero
    return y.numel() / (N_CLASSES * counts)


# --------------------------------------------------------------------- models

class MLP(torch.nn.Module):
    """No message passing, no recurrence."""

    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(N_FEATURES, HIDDEN)
        self.lin2 = torch.nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, data, hidden=None):
        h = F.relu(self.lin1(data.x))
        return self.lin2(h), None


class GCN(torch.nn.Module):
    """The paper's model: two GCNConv layers, no recurrence."""

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(N_FEATURES, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, N_CLASSES)

    def forward(self, data, hidden=None):
        h = F.relu(self.conv1(data.x, data.edge_index))
        return self.conv2(h, data.edge_index), None


class GRUOnly(torch.nn.Module):
    """Recurrence without message passing. Isolates whether the temporal
    dimension carries signal independently of the graph."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(N_FEATURES, HIDDEN)
        self.cell = torch.nn.GRUCell(HIDDEN, HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, data, hidden=None):
        h_in = F.relu(self.lin(data.x))
        hidden = self.cell(h_in, hidden)
        return self.out(hidden), hidden


class TGCN(torch.nn.Module):
    """Spatio-temporal: GCNConv encodes each monthly snapshot, a GRUCell
    carries per-node state across months (T-GCN / EvolveGCN family)."""

    def __init__(self):
        super().__init__()
        self.conv = GCNConv(N_FEATURES, HIDDEN)
        self.cell = torch.nn.GRUCell(HIDDEN, HIDDEN)
        self.out = torch.nn.Linear(HIDDEN, N_CLASSES)

    def forward(self, data, hidden=None):
        h_in = F.relu(self.conv(data.x, data.edge_index))
        hidden = self.cell(h_in, hidden)
        return self.out(hidden), hidden


MODELS = {
    "MLP": (MLP, False),
    "GCN": (GCN, False),
    "GRU (no graph)": (GRUOnly, True),
    "T-GCN": (TGCN, True),
}


# ------------------------------------------------------------------ train/eval

def train_eval(model_cls, recurrent, data_list, seed, class_weight=None):
    """
    > [!AML-DOC-UNIT]
    Temporal hold-out identical to the paper's: fit on months 1-9, predict
    10-12.

    Non-recurrent models see each month independently, exactly as in
    run_ablation.py. Recurrent models unroll month 1..9 within each epoch,
    carrying hidden state; at test time the state carried out of month 9 is
    fed into month 10 and onward, so the test months are a genuine forward
    continuation of the sequence rather than isolated snapshots.

    @param class_weight: FloatTensor[N_CLASSES] or None (unweighted loss).
    @returns (y_true, y_pred) over the 87 test instances.
    """
    set_all_seeds(seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_data, test_data = data_list[:TRAIN_MONTHS], data_list[TRAIN_MONTHS:]

    for _ in range(EPOCHS):
        model.train()
        hidden = None
        for data in train_data:
            opt.zero_grad()
            logits, hidden = model(data, hidden)
            loss = F.cross_entropy(logits, data.y, weight=class_weight)
            loss.backward()
            opt.step()
            # detach so backprop stays within one month (truncated BPTT);
            # full BPTT over 9 months on 261 instances overfits immediately
            hidden = hidden.detach() if hidden is not None else None

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        hidden = None
        if recurrent:                      # replay train months to warm state
            for data in train_data:
                _, hidden = model(data, hidden)
        for data in test_data:
            logits, hidden = model(data, hidden)
            y_true.extend(data.y.numpy())
            y_pred.extend(logits.argmax(1).numpy())
    return np.array(y_true), np.array(y_pred)


def metrics_from_preds(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2],
                         zero_division=0)
    recalls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )[1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_p": p, "macro_r": r, "macro_f1": f1,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "f1_c0": per_class[0], "f1_c1": per_class[1], "f1_c2": per_class[2],
        "recall_c2": recalls[2],
    }


def main(gnn_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, data_list = load_aligned(gnn_dir)
    weights = balanced_weights(data_list[:TRAIN_MONTHS])
    print("balanced class weights (train months):",
          [round(w, 3) for w in weights.tolist()], flush=True)

    rows = []
    for name, (cls, recurrent) in MODELS.items():
        for weighted in (False, True):
            label = f"{name}{' + balanced' if weighted else ''}"
            print(f"running {label} x{N_RUNS} ...", flush=True)
            for run in range(N_RUNS):
                y_true, y_pred = train_eval(
                    cls, recurrent, data_list, seed=run,
                    class_weight=weights if weighted else None,
                )
                m = metrics_from_preds(y_true, y_pred)
                m.update(model=name, balanced=weighted, label=label, run=run)
                rows.append(m)

    raw = pd.DataFrame(rows)
    metric_cols = ["accuracy", "macro_p", "macro_r", "macro_f1",
                   "weighted_f1", "f1_c0", "f1_c1", "f1_c2", "recall_c2"]
    summary = raw.groupby("label")[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()

    raw.to_csv(out_dir / "temporal_results_raw.csv", index=False)
    summary.to_csv(out_dir / "temporal_results.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["label", "accuracy_mean", "accuracy_std", "macro_f1_mean",
            "macro_f1_std", "f1_c2_mean", "f1_c2_std", "recall_c2_mean"]
    print(f"\n=== TEMPORAL + CLASS-WEIGHT (mean +/- std over {N_RUNS} seeds) ===")
    print(summary[show].to_string(index=False))

    print("\n=== EFFECT OF BALANCING (paired over seeds, per model) ===")
    for name in MODELS:
        a = raw[(raw.model == name) & (~raw.balanced)].sort_values("run")
        b = raw[(raw.model == name) & (raw.balanced)].sort_values("run")
        for metric in ("macro_f1", "f1_c2"):
            x, y = a[metric].values, b[metric].values
            if np.allclose(x, y):
                print(f"{name:16s} {metric:9s} unchanged")
                continue
            _, p = stats.ttest_rel(y, x)
            print(f"{name:16s} {metric:9s} "
                  f"{x.mean():.3f} -> {y.mean():.3f} "
                  f"(delta={y.mean() - x.mean():+.3f}, p={p:.4f})")

    print("\n=== DOES TIME HELP? (paired, vs the non-recurrent counterpart) ===")
    for temporal, static in (("T-GCN", "GCN"), ("GRU (no graph)", "MLP")):
        for weighted in (False, True):
            a = raw[(raw.model == static) & (raw.balanced == weighted)].sort_values("run")
            b = raw[(raw.model == temporal) & (raw.balanced == weighted)].sort_values("run")
            tag = "balanced" if weighted else "unweighted"
            x, y = a["accuracy"].values, b["accuracy"].values
            _, p = stats.ttest_rel(y, x)
            print(f"{temporal:16s} vs {static:5s} [{tag:10s}] accuracy "
                  f"{x.mean():.3f} -> {y.mean():.3f} "
                  f"(delta={y.mean() - x.mean():+.3f}, p={p:.4f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
