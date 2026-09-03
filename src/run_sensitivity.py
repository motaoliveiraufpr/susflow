# -*- coding: utf-8 -*-
"""
> [!AML-DOC-FILE]
@file run_sensitivity.py
@description Sensitivity of the vulnerability index to the Eq. (1) weights and
             the Eq. (2) cut points.
@module sibgrapi.rebuttal
@exports main
@created 2026-08-04
@context Answers R1 ("the paper should explain why each attribute was selected,
         how the weights in Eq. (1) were defined, and why thresholds 0.33 and
         0.66 were chosen ... A sensitivity analysis or domain-expert
         justification would strengthen this part") and R2 ("the article should
         mention how the constants in the linear combination were determined").

         Both constants were fixed a priori, so the honest question is not
         whether they are optimal but whether the paper's conclusions survive
         reasonable alternatives. Two things are measured per configuration:
           (i)  label stability -- agreement and Cohen's kappa against the
                published labelling;
           (ii) conclusion stability -- whether the no-graph MLP still beats
                the GCN, which is the paper's central claim.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

from pipeline import INDEX_WEIGHTS, THRESHOLDS, load_all
from run_temporal import (
    GCN,
    MLP,
    N_CLASSES,
    balanced_weights,
    metrics_from_preds,
    train_eval,
)

N_SEEDS = 5               # per configuration; the ranking is what matters here
N_RANDOM = 30             # Dirichlet draws over the four index weights
TRAIN_MONTHS = 9
INDEX_COLS = ["curitiba_dep", "to_curitiba", "time_min", "distance_km"]


def labels_of(data_list):
    """Concatenated labels over the three test months (87 instances)."""
    return torch.cat([d.y for d in data_list[TRAIN_MONTHS:]]).numpy()


def all_labels_of(data_list):
    """Concatenated labels over all twelve months (348 instances)."""
    return torch.cat([d.y for d in data_list]).numpy()


def evaluate_config(gnn_dir, weights, thresholds, reference_all, reference_test):
    """
    > [!AML-DOC-UNIT]
    Rebuild the dataset under one (weights, thresholds) setting and report
    label stability plus the MLP-vs-GCN comparison.

    @param reference_all/reference_test: labels of the published configuration,
           used as the agreement baseline.
    @returns dict of stability and performance figures.
    """
    _, _, data_list = load_all(gnn_dir, weights=weights, thresholds=thresholds)

    y_all, y_test = all_labels_of(data_list), labels_of(data_list)
    counts = np.bincount(y_test, minlength=N_CLASSES)

    class_weight = balanced_weights(data_list[:TRAIN_MONTHS])
    scores = {}
    for name, cls in (("MLP", MLP), ("GCN", GCN)):
        accs, f1s = [], []
        for seed in range(N_SEEDS):
            yt, yp = train_eval(cls, False, data_list, seed,
                                class_weight=class_weight)
            m = metrics_from_preds(yt, yp)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
        scores[f"{name}_acc"] = float(np.mean(accs))
        scores[f"{name}_macro_f1"] = float(np.mean(f1s))

    return {
        "agreement_all": float((y_all == reference_all).mean()),
        "kappa_all": float(cohen_kappa_score(y_all, reference_all)),
        "agreement_test": float((y_test == reference_test).mean()),
        "test_c0": int(counts[0]), "test_c1": int(counts[1]),
        "test_c2": int(counts[2]),
        **scores,
        "mlp_beats_gcn": bool(scores["MLP_acc"] > scores["GCN_acc"]),
    }


def main(gnn_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, reference = load_all(gnn_dir)
    ref_all, ref_test = all_labels_of(reference), labels_of(reference)

    def w(*vals):
        return dict(zip(INDEX_COLS, vals))

    named_weights = {
        "published (.4/.3/.2/.1)": w(0.4, 0.3, 0.2, 0.1),
        "equal (.25 each)":        w(0.25, 0.25, 0.25, 0.25),
        "swap top two":            w(0.3, 0.4, 0.2, 0.1),
        "accessibility-heavy":     w(0.2, 0.2, 0.35, 0.25),
        "dependency only":         w(1.0, 0.0, 0.0, 0.0),
        "demand only":             w(0.0, 1.0, 0.0, 0.0),
    }

    rows = []
    print("=== A. WEIGHTS OF EQ. (1), thresholds fixed at 0.33/0.66 ===",
          flush=True)
    for label, weights in named_weights.items():
        r = evaluate_config(gnn_dir, weights, THRESHOLDS, ref_all, ref_test)
        r.update(kind="weights", config=label)
        rows.append(r)
        print(f"  {label:24s} agree={r['agreement_all']:.3f} "
              f"kappa={r['kappa_all']:.3f} "
              f"test=[{r['test_c0']},{r['test_c1']},{r['test_c2']}] "
              f"MLP={r['MLP_acc']:.3f} GCN={r['GCN_acc']:.3f}", flush=True)

    print(f"\n=== B. {N_RANDOM} RANDOM WEIGHT VECTORS (Dirichlet, alpha=1) ===",
          flush=True)
    rng = np.random.default_rng(0)
    for i in range(N_RANDOM):
        weights = w(*rng.dirichlet(np.ones(4)))
        r = evaluate_config(gnn_dir, weights, THRESHOLDS, ref_all, ref_test)
        r.update(kind="random_weights", config=f"dirichlet_{i:02d}",
                 **{f"w_{c}": weights[c] for c in INDEX_COLS})
        rows.append(r)
    rnd = pd.DataFrame([r for r in rows if r["kind"] == "random_weights"])
    print(f"  agreement (all months): mean={rnd.agreement_all.mean():.3f} "
          f"min={rnd.agreement_all.min():.3f} max={rnd.agreement_all.max():.3f}")
    print(f"  kappa                 : mean={rnd.kappa_all.mean():.3f} "
          f"min={rnd.kappa_all.min():.3f}")
    print(f"  MLP beat GCN in {int(rnd.mlp_beats_gcn.sum())}/{len(rnd)} draws")
    print(f"  high-vulnerability test instances: "
          f"min={int(rnd.test_c2.min())} max={int(rnd.test_c2.max())}")

    print("\n=== C. CUT POINTS OF EQ. (2), published weights ===", flush=True)
    for lo, hi in [(0.25, 0.50), (0.33, 0.66), (0.40, 0.70), (0.30, 0.60)]:
        r = evaluate_config(gnn_dir, INDEX_WEIGHTS, (lo, hi), ref_all, ref_test)
        r.update(kind="thresholds", config=f"({lo}, {hi})")
        rows.append(r)
        print(f"  ({lo:.2f}, {hi:.2f})            agree={r['agreement_all']:.3f} "
              f"kappa={r['kappa_all']:.3f} "
              f"test=[{r['test_c0']},{r['test_c1']},{r['test_c2']}] "
              f"MLP={r['MLP_acc']:.3f} GCN={r['GCN_acc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sensitivity_results.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"MLP > GCN in {int(df.mlp_beats_gcn.sum())}/{len(df)} configurations "
          f"({100 * df.mlp_beats_gcn.mean():.0f}%)")
    print(f"minimum label agreement with the published labelling: "
          f"{df.agreement_all.min():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
