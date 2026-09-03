# -*- coding: utf-8 -*-
"""
> [!AML-DOC-FILE]
@file run_ablation.py
@description Reviewer-requested experiments: MLP (no-graph) baseline, graph
             ablations (shuffled / reverse / undirected / edge-weighted), and
             repeated runs with seed variation for uncertainty estimates.
             The edge-weight key is the one the data actually carries
             ('dep_curitiba'), and a paired shuffled-vs-real significance test
             is included.
@module sibgrapi.rebuttal
@exports main
@created 2026-08-04
@context Answers R1 ("Important ablations are missing, including weighted
         edges, reverse or undirected edges, shuffled topology, and an MLP
         baseline") and R1/R3 ("results should include repeated runs or
         uncertainty estimates").
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

from pipeline import EDGE_ATTR_KEY, FEATURE_COLS, load_all

N_RUNS = 30
HIDDEN = 16
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4
N_FEATURES = 7
N_CLASSES = 3

# The pickled monthly graphs carry exactly one edge attribute, and it is
# 'dep_curitiba' -- the same quantity that also appears as a node feature and
# as the 0.4-weighted term of Eq. (1). Weighting edges by it is therefore not
# a neutral ablation; see the note printed at the end of main().
EDGE_WEIGHT_ATTR = EDGE_ATTR_KEY


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class GCN(torch.nn.Module):
    """Architecture identical to the paper's (cells 29-30 of the notebook),
    extended only to accept an optional edge_weight for the weighted ablation.
    With edge_weight=None it is numerically identical to the original."""

    def __init__(self, in_channels=N_FEATURES, hidden_channels=HIDDEN,
                 out_channels=N_CLASSES):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class MLP(torch.nn.Module):
    """Same width, depth, activation, optimizer and schedule as the GCN, with
    GCNConv replaced by Linear: no message passing at all. Any GCN-vs-MLP gap
    is therefore attributable to topology, not to capacity or nonlinearity."""

    def __init__(self, in_channels=N_FEATURES, hidden_channels=HIDDEN,
                 out_channels=N_CLASSES):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))


# ---------------------------------------------------------------- edge variants

def edges_original(data, graphs, seed=None):
    return data.edge_index, None


def edges_reversed(data, graphs, seed=None):
    return data.edge_index.flip(0), None


def edges_undirected(data, graphs, seed=None):
    return to_undirected(data.edge_index, num_nodes=data.num_nodes), None


def edges_shuffled(data, graphs, seed=0):
    """Random topology with the same node count and the same edge count.
    The key control: if the real topology is no better than this, the model is
    not extracting information from the observed structure."""
    g = torch.Generator().manual_seed(int(seed) * 1000 + int(data.month))
    n, n_edges = data.num_nodes, data.edge_index.size(1)
    out = torch.empty((2, 0), dtype=torch.long)
    while out.size(1) < n_edges:
        src = torch.randint(0, n, (n_edges * 3,), generator=g)
        dst = torch.randint(0, n, (n_edges * 3,), generator=g)
        m = src != dst
        out = torch.cat([out, torch.stack([src[m], dst[m]])], dim=1)
    return out[:, :n_edges], None


def edges_weighted(data, graphs, seed=None):
    """Weight each edge by its stored 'dep_curitiba' value. Edge order matches
    edge_index because both are built by iterating G.edges() (notebook cell
    21)."""
    G = graphs[int(data.month)]
    w = [float(G[u][v].get(EDGE_WEIGHT_ATTR, 1.0)) for u, v in G.edges()]
    return data.edge_index, torch.tensor(w, dtype=torch.float)


# ---------------------------------------------------------------- train / eval

def train_eval_gcn(train_data, test_data, graphs, edge_fn, seed):
    set_all_seeds(seed)
    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    variants = [edge_fn(d, graphs, seed) for d in train_data]

    for _ in range(EPOCHS):
        model.train()
        for data, (ei, ew) in zip(train_data, variants):
            opt.zero_grad()
            loss = F.cross_entropy(model(data.x, ei, ew), data.y)
            loss.backward()
            opt.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_data:
            ei, ew = edge_fn(data, graphs, seed)
            y_true.extend(data.y.numpy())
            y_pred.extend(model(data.x, ei, ew).argmax(1).numpy())
    return np.array(y_true), np.array(y_pred)


def train_eval_mlp(train_data, test_data, seed):
    set_all_seeds(seed)
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        model.train()
        for data in train_data:
            opt.zero_grad()
            loss = F.cross_entropy(model(data.x), data.y)
            loss.backward()
            opt.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in test_data:
            y_true.extend(data.y.numpy())
            y_pred.extend(model(data.x).argmax(1).numpy())
    return np.array(y_true), np.array(y_pred)


def metrics_from_preds(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2],
                         zero_division=0)
    rec_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )[1]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_p": p, "macro_r": r, "macro_f1": f1,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "f1_c0": per_class[0], "f1_c1": per_class[1], "f1_c2": per_class[2],
        "recall_c2": rec_per_class[2],
    }


def run_sklearn_baselines(monthly_features):
    def stack(months):
        X = pd.concat([monthly_features[m][FEATURE_COLS] for m in months],
                      ignore_index=True)
        y = pd.concat([monthly_features[m]["vulnerability_class"]
                       for m in months], ignore_index=True)
        return X, y

    X_tr, y_tr = stack(range(1, 10))
    X_te, y_te = stack(range(10, 13))

    rows = []
    for name, clf in [
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced")),
        ("Logistic Regression", LogisticRegression(
            max_iter=2000, random_state=42, class_weight="balanced")),
    ]:
        clf.fit(X_tr, y_tr)
        m = metrics_from_preds(y_te.values, clf.predict(X_te))
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows)


def main(gnn_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs, features, data_list = load_all(gnn_dir)
    train_data, test_data = data_list[:9], data_list[9:]

    variants = {
        "GCN (original)": edges_original,
        "GCN (reverse-edge)": edges_reversed,
        "GCN (undirected)": edges_undirected,
        "GCN (shuffled-edge)": edges_shuffled,
        "GCN (edge-weighted)": edges_weighted,
    }

    raw_rows = []
    for name, fn in variants.items():
        print(f"running {name} x{N_RUNS} ...", flush=True)
        for run in range(N_RUNS):
            yt, yp = train_eval_gcn(train_data, test_data, graphs, fn, run)
            m = metrics_from_preds(yt, yp)
            m.update(model=name, run=run)
            raw_rows.append(m)

    print(f"running MLP (no graph) x{N_RUNS} ...", flush=True)
    for run in range(N_RUNS):
        yt, yp = train_eval_mlp(train_data, test_data, run)
        m = metrics_from_preds(yt, yp)
        m.update(model="MLP (no graph)", run=run)
        raw_rows.append(m)

    raw = pd.DataFrame(raw_rows)
    metric_cols = ["accuracy", "macro_p", "macro_r", "macro_f1",
                   "weighted_f1", "f1_c0", "f1_c1", "f1_c2", "recall_c2"]

    summary = raw.groupby("model")[metric_cols].agg(["mean", "std"]).round(3)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()

    baselines = run_sklearn_baselines(features)
    b = baselines.copy()
    for c in metric_cols:
        b[f"{c}_mean"] = b[c].round(3)
        b[f"{c}_std"] = 0.0
    b = b[["model"] + [f"{c}_mean" for c in metric_cols]
          + [f"{c}_std" for c in metric_cols]]

    full = pd.concat([summary, b], ignore_index=True)

    raw.to_csv(out_dir / "ablation_results_raw.csv", index=False)
    full.to_csv(out_dir / "ablation_results.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["model", "accuracy_mean", "accuracy_std", "macro_f1_mean",
            "macro_f1_std", "f1_c2_mean", "f1_c2_std", "recall_c2_mean"]
    print("\n=== COMPARISON TABLE (mean +/- std over "
          f"{N_RUNS} seeds; sklearn baselines deterministic) ===")
    print(full[show].to_string(index=False))

    # Paired test: same seed -> same init, so runs pair naturally.
    print("\n=== PAIRED TESTS vs GCN (original), across seeds ===")
    base = raw[raw.model == "GCN (original)"].sort_values("run")
    for name in variants:
        if name == "GCN (original)":
            continue
        other = raw[raw.model == name].sort_values("run")
        for metric in ("accuracy", "macro_f1"):
            a = base[metric].values
            c = other[metric].values
            diff = c.mean() - a.mean()
            if np.allclose(a, c):
                print(f"{name:24s} {metric:10s} identical to original")
                continue
            t, p = stats.ttest_rel(c, a)
            print(f"{name:24s} {metric:10s} delta={diff:+.4f}  p={p:.4f}")

    print("\n=== TOPOLOGY DIAGNOSTICS ===")
    for m in (1, 6, 12):
        G = graphs[m]
        deg = dict(G.degree())
        leaves = sum(1 for n, d in deg.items() if d == 1 and n != '410690')
        print(f"month {m:2d}: directed={G.is_directed()} "
              f"edges={G.number_of_edges()} "
              f"deg(Curitiba)={deg.get('410690', 0)} "
              f"degree-1 nodes={leaves} "
              f"isolated={sum(1 for d in deg.values() if d == 0)}")
    print("\nNOTE: every edge is incident to Curitiba (410690); the monthly "
          "graphs are stars, and the sole edge attribute is 'dep_curitiba', "
          "which is also a node feature and the 0.4 term of Eq. (1).")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
