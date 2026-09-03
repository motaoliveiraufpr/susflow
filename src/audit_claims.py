# -*- coding: utf-8 -*-
"""
> [!AML-DOC-FILE]
@file audit_claims.py
@description Verifies, against the data, the four load-bearing empirical claims
             made in the revised manuscript and the response letter.
@module sibgrapi.rebuttal
@exports main
@created 2026-08-04
@context Raised in internal review before resubmission:
         A. "Curitiba's feature vector is zero in every month" -- the code
            injects the zero row *conditionally*, so the absolute claim must be
            audited month by month before it can stand.
         B. "every edge is incident to Curitiba" (star topology).
         C. Random Forest and Logistic Regression report identical values in
            three metrics at once; check whether the predictions really
            coincide or only the rounded metrics do.
         D. The MLP reports std 0.000 over 30 seeds; check the seeds actually
            change the initialization.
"""

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from pipeline import (
    ALL_IBGE_CODES,
    COLUMN_RENAME,
    CURITIBA,
    FEATURE_COLS,
    MONTHS,
    load_all,
)
from run_temporal import MLP, N_CLASSES, balanced_weights, train_eval

RULE = "=" * 78


def audit_curitiba(gnn_dir):
    """A + B: is Curitiba zero-filled in EVERY month, and is the graph a star?"""
    print(RULE)
    print("A. CURITIBA: raw presence, zero-filling, and scaled feature vector")
    print(RULE)

    graphs, features, data_list = load_all(gnn_dir)

    print(f"{'month':>5} {'in CSV':>7} {'raw sum':>10} {'scaled sum':>11} "
          f"{'all-zero':>9} {'edges':>6} {'inc.Cur':>8} {'deg':>4}")
    injected = zero_scaled = 0
    non_star_months = []

    for month, stub in MONTHS.items():
        raw = pd.read_csv(f"{gnn_dir}/features_2025_{stub}.csv")
        raw = raw.rename(columns=COLUMN_RENAME)
        raw["ibge_code"] = raw["ibge_code"].astype(float).astype(int).astype(str)
        present = CURITIBA in raw["ibge_code"].values

        if present:
            raw_row = raw.set_index("ibge_code").loc[CURITIBA, FEATURE_COLS]
            raw_sum = float(np.asarray(raw_row, dtype=float).sum())
        else:
            raw_sum = float("nan")
            injected += 1

        df = features[month].set_index("ibge_code")
        scaled = np.asarray(df.loc[CURITIBA, FEATURE_COLS], dtype=float)
        all_zero = bool((scaled == 0).all())
        zero_scaled += all_zero

        graph = graphs[month]
        n_edges = graph.number_of_edges()
        incident = sum(1 for u, v in graph.edges()
                       if u == CURITIBA or v == CURITIBA)
        if incident != n_edges:
            non_star_months.append(month)

        print(f"{month:>5} {str(present):>7} "
              f"{raw_sum:>10.3f} {scaled.sum():>11.3f} "
              f"{str(all_zero):>9} {n_edges:>6} {incident:>8} "
              f"{graph.degree(CURITIBA):>4}")

    print()
    print(f"  months where Curitiba was ABSENT from the CSV (row injected): "
          f"{injected}/12")
    print(f"  months where Curitiba's SCALED vector is all zeros: "
          f"{zero_scaled}/12")
    print(f"  months whose graph is NOT a pure star: "
          f"{non_star_months or 'none'}")

    print()
    print("B. EDGE ORIENTATION AS STORED IN edge_index")
    print(f"{'month':>5} {'edges':>6} {'Cur=source':>11} {'Cur=target':>11}")
    for data in data_list:
        graph = graphs[data.month]
        nodes = list(graph.nodes())
        idx = nodes.index(CURITIBA)
        src, dst = data.edge_index
        print(f"{data.month:>5} {data.edge_index.size(1):>6} "
              f"{int((src == idx).sum()):>11} {int((dst == idx).sum()):>11}")

    return injected, zero_scaled, non_star_months


def audit_rf_vs_lr(gnn_dir):
    """C: do RF and LR actually produce the same 87 predictions?"""
    print()
    print(RULE)
    print("C. RANDOM FOREST vs LOGISTIC REGRESSION -- same predictions or only")
    print("   the same rounded metrics?")
    print(RULE)

    _, features, _ = load_all(gnn_dir)

    def stack(months):
        X = pd.concat([features[m][FEATURE_COLS] for m in months],
                      ignore_index=True)
        y = pd.concat([features[m]["vulnerability_class"] for m in months],
                      ignore_index=True)
        return X, y

    X_tr, y_tr = stack(range(1, 10))
    X_te, y_te = stack(range(10, 13))

    rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                class_weight="balanced").fit(X_tr, y_tr)
    lr = LogisticRegression(max_iter=2000, random_state=42,
                            class_weight="balanced").fit(X_tr, y_tr)
    p_rf, p_lr = rf.predict(X_te), lr.predict(X_te)

    agree = int((p_rf == p_lr).sum())
    print(f"  identical predictions: {agree}/{len(p_rf)} "
          f"({100 * agree / len(p_rf):.1f}%)")
    print(f"  predictions are element-wise identical: {bool((p_rf == p_lr).all())}")
    if agree != len(p_rf):
        diff = np.where(p_rf != p_lr)[0]
        print(f"  disagreement at test indices {diff.tolist()}: "
              f"RF={p_rf[diff].tolist()} vs LR={p_lr[diff].tolist()} "
              f"(true={y_te.values[diff].tolist()})")

    for name, pred in (("Random Forest", p_rf), ("Logistic Reg.", p_lr)):
        acc = accuracy_score(y_te, pred)
        _, _, mf1, _ = precision_recall_fscore_support(
            y_te, pred, average="macro", zero_division=0)
        f1c = f1_score(y_te, pred, average=None, labels=[0, 1, 2],
                       zero_division=0)
        print(f"\n  {name}")
        print(f"    accuracy  = {acc:.6f}   ({int(round(acc * 87))}/87 correct)")
        print(f"    macro-F1  = {mf1:.6f}")
        print(f"    F1 per class = {np.round(f1c, 6).tolist()}")
        print(f"    confusion matrix:\n{confusion_matrix(y_te, pred)}")


def audit_mlp_seeds(gnn_dir):
    """D: do the 30 seeds really change the MLP, despite std = 0.000?"""
    print()
    print(RULE)
    print("D. MLP -- do the seeds change initialization, and are the outputs")
    print("   genuinely identical?")
    print(RULE)

    from pipeline import load_aligned
    _, _, data_list = load_aligned(gnn_dir)
    weights = balanced_weights(data_list[:9])

    inits, preds, accs = [], [], []
    for seed in range(5):
        torch.manual_seed(seed)
        model = MLP()
        inits.append(float(model.lin1.weight.sum()))
        y_true, y_pred = train_eval(MLP, False, data_list, seed,
                                    class_weight=weights)
        preds.append(y_pred)
        accs.append(accuracy_score(y_true, y_pred))

    print(f"  sum of lin1 weights at init, seeds 0-4:")
    for s, v in enumerate(inits):
        print(f"    seed {s}: {v:+.6f}")
    print(f"  distinct initializations: {len(set(inits))}/5")

    same = all((preds[0] == p).all() for p in preds[1:])
    print(f"  all 5 seeds give identical predictions: {same}")
    print(f"  accuracies: {[round(a, 6) for a in accs]}")
    if not same:
        for s, p in enumerate(preds[1:], 1):
            d = int((preds[0] != p).sum())
            print(f"    seed {s} differs from seed 0 in {d}/87 predictions")


def main(gnn_dir):
    injected, zero_scaled, non_star = audit_curitiba(gnn_dir)
    audit_rf_vs_lr(gnn_dir)
    audit_mlp_seeds(gnn_dir)

    print()
    print(RULE)
    print("VERDICT ON THE CLAIMS AS CURRENTLY WRITTEN")
    print(RULE)
    print(f"  'zero-filled in every month'      -> "
          f"{'HOLDS' if zero_scaled == 12 else 'FALSE'} "
          f"({zero_scaled}/12 months all-zero after scaling; "
          f"{injected}/12 rows injected)")
    print(f"  'every edge is incident to Curitiba' -> "
          f"{'HOLDS' if not non_star else 'FALSE'}")


if __name__ == "__main__":
    main(sys.argv[1])
