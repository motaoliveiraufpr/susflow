# -*- coding: utf-8 -*-
"""
> [!AML-DOC-FILE]
@file pipeline.py
@description Faithful local replication of the notebook pipeline
             (notebooks/gcn_pipeline.ipynb, code cells 3-28): loads the 12
             monthly NetworkX graphs + feature CSVs, injects the Curitiba row,
             per-month MinMax-scales the 7 features, builds the composite
             vulnerability index and its 3-class discretization, and emits the
             12 PyG Data objects plus the temporal hold-out split.
@module sibgrapi.rebuttal
@exports load_all, MONTHS, FEATURE_COLS, COLUMN_RENAME
@created 2026-08-04
@context Reviewer response. Nothing here may deviate from the notebook: the
         ablation numbers are only comparable to Tables I-II of the paper if
         the data construction is bit-identical. Identifiers and dataframe
         columns are in English (reviewer R2), matching the names used in
         Section III-B of the paper; the CSVs on disk keep the original
         Portuguese headers and are renamed on load via COLUMN_RENAME.
"""

import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

# month index -> filename stub, in the notebook's order (cells 3-4)
MONTHS = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}

# On-disk CSV headers (Portuguese) -> the English names used in the paper.
# Reviewer R2 asked for English feature names; the raw files are left untouched
# so the notebooks that produced them stay reproducible.
COLUMN_RENAME = {
    "codigo_ibge": "ibge_code",
    "fora": "ext",
    "para_curitiba": "to_curitiba",
    "dep_curitiba": "curitiba_dep",
    "distancia_km": "distance_km",
    "tempo_min": "time_min",
}

# cell 15 / cell 20, in paper notation (Section III-B)
FEATURE_COLS = [
    "total", "ext", "export_pct", "to_curitiba",
    "curitiba_dep", "distance_km", "time_min",
]

# The single edge attribute stored inside the pickled graphs. It keeps its
# original Portuguese key because it lives in the pickle, not in our code.
# Note it equals the `curitiba_dep` node feature and the 0.4 term of Eq. (1).
EDGE_ATTR_KEY = "dep_curitiba"

CURITIBA = "410690"

# cell 11 -- the 29 IBGE codes; note Curitiba IS one of them
ALL_IBGE_CODES = [
    '410020', '410030', '410040', '410180', '410230', '410310',
    '410400', '410410', '410420', '410425', '410520', '410580',
    '410620', '410690', '412863', '410765', '411125', '411320',
    '411430', '411910', '411915', '411950', '412080', '412120',
    '412220', '412230', '412550', '412760', '412788',
]

# cell 12 (already in English column names)
CURITIBA_ROW = {
    "ibge_code": CURITIBA, "total": 0, "ext": 0, "export_pct": 0,
    "to_curitiba": 0, "curitiba_dep": 0, "distance_km": 0, "time_min": 0,
}

# Eq. (1) weights and Eq. (2) cut points, exposed so the sensitivity analysis
# can vary them without touching the loader.
INDEX_WEIGHTS = {
    "curitiba_dep": 0.4, "to_curitiba": 0.3, "time_min": 0.2, "distance_km": 0.1,
}
THRESHOLDS = (0.33, 0.66)


def classify(value, thresholds=THRESHOLDS):
    """
    > [!AML-DOC-UNIT]
    cell 17: equal-width tercile discretization of the vulnerability index.

    @param value: the composite index V_{i,t}.
    @param thresholds: (low, high) cut points; defaults to Eq. (2).
    @returns 0 (low), 1 (medium) or 2 (high).
    """
    low, high = thresholds
    if value < low:
        return 0
    elif value < high:
        return 1
    return 2


def load_all(gnn_dir, weights=None, thresholds=THRESHOLDS):
    """
    > [!AML-DOC-UNIT]
    Rebuild monthly_graphs / monthly_features / data_list exactly as the
    notebook does.

    @param gnn_dir: Path to the directory holding graph_2025_*.pkl and
                    features_2025_*.csv for all 12 months.
    @param weights: optional override of the Eq. (1) weights, for the
                    sensitivity analysis. None uses INDEX_WEIGHTS.
    @param thresholds: optional override of the Eq. (2) cut points.
    @returns (monthly_graphs, monthly_features, data_list) where data_list is
             ordered month 1..12 and each Data carries .month.
    @raises FileNotFoundError if any of the 24 monthly files is missing.
    @sideEffects none (pure read).
    """
    gnn_dir = Path(gnn_dir)
    weights = weights or INDEX_WEIGHTS

    monthly_graphs, monthly_features = {}, {}
    for month, stub in MONTHS.items():
        with open(gnn_dir / f"graph_2025_{stub}.pkl", "rb") as fh:
            monthly_graphs[month] = pickle.load(fh)
        df = pd.read_csv(gnn_dir / f"features_2025_{stub}.csv")
        df = df.rename(columns=COLUMN_RENAME)
        # cell 7: codes come out of CSV as floats; normalize to bare strings
        df["ibge_code"] = df["ibge_code"].astype(float).astype(int).astype(str)
        monthly_features[month] = df

    # cell 11: guarantee all 29 nodes exist in every monthly graph
    for month in range(1, 13):
        monthly_graphs[month].add_nodes_from(ALL_IBGE_CODES)

    # cell 13: Curitiba is a destination-only hub, so in some months it has no
    # outgoing-flow row of its own -> inject a zero row
    for month in range(1, 13):
        if CURITIBA not in monthly_features[month]["ibge_code"].values:
            monthly_features[month] = pd.concat(
                [monthly_features[month], pd.DataFrame([CURITIBA_ROW])],
                ignore_index=True,
            )

    # cell 15: MinMax scaling, refit independently per month
    scaler = MinMaxScaler()
    for month in range(1, 13):
        df = monthly_features[month].copy()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
        monthly_features[month] = df

    # cells 16 & 18: composite index (Eq. 1) and its discretization (Eq. 2)
    for month in range(1, 13):
        df = monthly_features[month]
        df["vulnerability_index"] = sum(
            w * df[col] for col, w in weights.items()
        )
        df["vulnerability_class"] = df["vulnerability_index"].apply(
            lambda v: classify(v, thresholds)
        )
        monthly_features[month] = df

    # cell 21: PyG Data objects, node order = G.nodes() order
    data_list = []
    for month in range(1, 13):
        graph = monthly_graphs[month]
        df = monthly_features[month].copy().set_index("ibge_code")
        nodes = list(graph.nodes())

        x = torch.tensor(df.loc[nodes, FEATURE_COLS].values, dtype=torch.float)
        y = torch.tensor(
            df.loc[nodes, "vulnerability_class"].values, dtype=torch.long
        )
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_index = torch.tensor(
            [[node_to_idx[u], node_to_idx[v]] for u, v in graph.edges()],
            dtype=torch.long,
        ).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.month = month
        data_list.append(data)

    return monthly_graphs, monthly_features, data_list


def load_aligned(gnn_dir, **kwargs):
    """
    > [!AML-DOC-UNIT]
    Same as load_all, but every month is reindexed to the fixed node order of
    ALL_IBGE_CODES instead of each graph's own `list(G.nodes())` order.

    RISK:HIGH if skipped -- `list(G.nodes())` depends on the order edges were
    inserted, so it differs between months. Any model that carries per-node
    state across months (the recurrent baselines answering R3) would silently
    associate month t's hidden state for node k with a different municipality
    at t+1. Per-month models are unaffected, which is why load_all keeps the
    original order and stays bit-identical to the notebook.

    @param gnn_dir: directory with the 24 monthly files.
    @param kwargs: forwarded to load_all (weights, thresholds).
    @returns (monthly_graphs, monthly_features, aligned_data_list) with node
             row i of every month corresponding to ALL_IBGE_CODES[i].
    @sideEffects none.
    """
    graphs, features, _ = load_all(gnn_dir, **kwargs)
    position = {code: i for i, code in enumerate(ALL_IBGE_CODES)}

    aligned = []
    for month in range(1, 13):
        graph = graphs[month]
        df = features[month].copy().set_index("ibge_code").loc[ALL_IBGE_CODES]

        x = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float)
        y = torch.tensor(df["vulnerability_class"].values, dtype=torch.long)
        edge_index = torch.tensor(
            [[position[u], position[v]] for u, v in graph.edges()],
            dtype=torch.long,
        ).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.month = month
        aligned.append(data)

    return graphs, features, aligned


if __name__ == "__main__":
    import sys

    graphs, features, data_list = load_all(sys.argv[1])
    print(f"{'month':>6} {'nodes':>6} {'edges':>6} {'c0':>4} {'c1':>4} {'c2':>4}")
    for data in data_list:
        counts = torch.bincount(data.y, minlength=3).tolist()
        print(f"{data.month:>6} {data.num_nodes:>6} "
              f"{data.edge_index.size(1):>6} "
              f"{counts[0]:>4} {counts[1]:>4} {counts[2]:>4}")

    test_y = torch.cat([d.y for d in data_list[9:]])
    print("\ntest set (months 10-12):", test_y.numel(), "instances")
    print("class distribution:", torch.bincount(test_y, minlength=3).tolist())
    print("\nedge attribute of month 1, first edge:")
    for u, v, attrs in list(graphs[1].edges(data=True))[:1]:
        print("  ", attrs)
