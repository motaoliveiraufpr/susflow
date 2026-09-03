# Graph Convolutional Networks for Temporal Vulnerability Classification in Obstetric Hospitalization Flows from SUS

This repository contains the code and reproducibility materials associated with the SIBGRAPI 2026 paper:

**“Graph Convolutional Networks for Temporal Vulnerability Classification in Obstetric Hospitalization Flows from SUS.”**

The study models intermunicipal obstetric hospitalization flows in the 2nd Health Region of Paraná, Brazil, as monthly graphs and evaluates Graph Convolutional Networks (GCNs) for municipality-level vulnerability classification. The experiments also include non-graph and temporal baselines, graph ablations, class balancing, and sensitivity analysis.

## Repository contents

```text
.
├── notebooks/        Data preparation and exploratory notebooks
├── src/              Experimental and reproducibility scripts
├── results/          Experiment outputs
└── README.md
```

Main scripts:

- `src/pipeline.py` — loads and prepares the monthly graph data.
- `src/run_ablation.py` — evaluates graph variants and non-graph baselines.
- `src/run_temporal.py` — evaluates temporal models and class-balanced variants.
- `src/run_sensitivity.py` — evaluates sensitivity to vulnerability-index configurations.

## Data

The experiments use publicly available administrative health data from **SIH/SUS (DATASUS)** and geographic/reference data from public Brazilian data sources.

Raw datasets are not distributed in this repository. The experimental scripts expect the processed monthly graph and feature files to be available locally.

The study covers:

- 29 municipalities from the 2nd Health Region of Paraná;
- monthly observations from January to December 2025;
- 35,022 obstetric hospitalizations;
- seven node features;
- three vulnerability classes.

No individual-level patient information is published in this repository.

## Experimental protocol

The primary evaluation uses a temporal hold-out:

- **Training:** months 1–9
- **Testing:** months 10–12

Stochastic neural models are evaluated across 30 random seeds.

The repository includes experiments with:

- Graph Convolutional Network (GCN)
- Logistic Regression
- Random Forest
- Multilayer Perceptron (MLP)
- Gated Recurrent Unit (GRU)
- GCN–GRU
- graph-orientation and topology ablations
- class-balanced training
- vulnerability-index sensitivity analysis

## Requirements

A Python environment with the following main packages is required:

```bash
pip install torch torch-geometric scikit-learn pandas networkx scipy
```

## Running the experiments

Assuming the processed graph data are available in `data/gnn/`:

```bash
python src/run_ablation.py data/gnn results
python src/run_temporal.py data/gnn results
python src/run_sensitivity.py data/gnn results
```

The scripts write their outputs to the specified results directory.

## Reproducibility

For the temporal test set, the expected evaluation set contains **87 municipality-month instances** (29 municipalities × 3 months).

The repository is intended to support verification and reproduction of the computational experiments reported in the paper.

## Research use

This work is intended for research on graph-based learning, healthcare referral networks, spatio-temporal analysis, and public-health data analytics. The vulnerability classes used in the study are analytical categories derived from a composite index and should not be interpreted as clinically validated risk categories.

## Citation

If you use this repository, please cite the associated SIBGRAPI 2026 paper:

> *Graph Convolutional Networks for Temporal Vulnerability Classification in Obstetric Hospitalization Flows from SUS.*  
> SIBGRAPI 2026, Main Track.

A complete bibliographic entry can be added after the proceedings metadata and DOI become available.
