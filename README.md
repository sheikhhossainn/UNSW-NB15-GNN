# UNSW-NB15 Intrusion Detection: Controlled GNN and Tabular Experiments

CSE475 group project. We classify network flows from UNSW-NB15 into 10 classes (Normal and nine attack types) and ask one main question:

> Does the graph help a flow-level Graph Neural Network (GNN) for multi-class intrusion detection?

We compare tabular models (LightGBM, with and without a two-stage design) with two GNN designs, and run controls (no graph, shuffled edges, matched graph density) before drawing conclusions. All experiments were run on Kaggle (GPU for the GNNs). The main metric is macro F1, because Normal is 95.4% of the test rows and weighted F1 hides the rare classes.

## What we found (short version)

These are observations from one dataset and one split. The full evidence and the limits are in `Our_Work/documents/`.

| Finding | Evidence |
|---|---|
| Class weighting made the largest difference among the tabular choices | +0.076 macro F1 (single-stage LightGBM) |
| A two-stage design added a smaller gain; synthetic (SMOTE) rows did not help | +0.011; removing them gave +0.011 to +0.015 |
| Best score: two-stage weighted LightGBM on the original data | 0.670 macro F1 (95% interval 0.658 to 0.679); picked on the test set among five variants, so possibly slightly optimistic |
| Analysis and Backdoor stay near F1 0.17 in every model | About 80% of their training rows have another row with identical features and a different label. This lowers the achievable score but does not explain all of it: a rule that memorises the most common label of each group reaches only about F1 0.36 on them |
| The graph gave our first GNN a small gain once train and test graph density were matched | +0.017 macro F1 over the same network without the graph |
| A paper-style E-GraphSAGE showed no gain from the graph | -0.008 to -0.010 with matched density |
| Graph features did not help LightGBM | 0.6709 vs 0.6695 |
| Both GNNs stay below the best tree model | 0.07 to 0.08 macro F1; cause not isolated |
| On a 5:2:3 split like the GTCN-G paper's, every model we trained scores a weighted F1 of 0.980 to 0.984, above the 0.9512 that paper reports | Not like-for-like: the paper leaves out its preprocessing, split type and loss, its data has 700,001 flows against our 2,042,340, and our version of its E-GraphSAGE-M baseline scores 0.981 where it reports 0.8934. A tree model without a graph is among those above 0.9512 |
| On the 5:2:3 split the best macro F1 is 0.667 (class-weighted LightGBM with class scales tuned on validation); the graph again did not help | Graph minus no graph -0.010 macro F1 (-0.018 to -0.002); two-stage design no gain here (-0.002) |
| Weighted F1 was 0.978 to 0.9825 for the paper-style GNN conditions and the best LightGBM model, so it cannot tell them apart | Macro F1 ranges from 0.51 to 0.67 |

## Folder layout

```
README.md
LICENSE
.gitignore
Our_Work/
  notebooks/              14 notebooks that produce the results, grouped by purpose
    01_preprocessing/     data cleaning and export
    02_tabular_models/    baselines and the LightGBM experiments
    03_graph_models/      the GNN experiments and graph features for LightGBM
    04_evaluation/        bootstrap intervals
    05_paper_replication/ the 5:2:3 split of the GTCN-G paper: baseline, improvements, LightGBM check
    superseded/           2 earlier notebooks kept for reference
  documents/              report, comparison with GTCN-G, literature comparison, trial log
  figures/                charts and per-run CSVs, one folder per notebook
```

## Files

### `Our_Work/notebooks/`

Run in the order of the folder numbers (every notebook uses the output of `01_preprocessing`). All were run on Kaggle with the dataset `shahiismyname/unsw-nb15-preprocessed-dataset` attached.

#### `01_preprocessing/`

| Notebook | What it does |
|---|---|
| `unsw-nb15-preprocessing.ipynb` | Cleans the raw CSVs (removes duplicates), selects and transforms features using training data only, checks label overlap, builds the SMOTE training set and the graph endpoint tables, and exports the model-ready files used by every other notebook (see Reproducing the results). |

#### `02_tabular_models/`

| Notebook | What it does |
|---|---|
| `unsw-nb15-baseline.ipynb` | First baselines: Random Forest, LightGBM, XGBoost, MLP; the two-stage model and an ensemble. These first numbers had two weaknesses (an effectively unweighted LightGBM and a leaky SMOTE validation set) that later notebooks correct. |
| `unsw-nb15-design-controls.ipynb` | Controlled comparison of five LightGBM variants (class weights, single vs two-stage, with and without SMOTE rows), 3 seeds. Source of the 0.670 model. |
| `unsw-nb15-ttl-ablation.ipynb` | Best tabular model with and without the three TTL features; saves its test predictions. |
| `unsw-nb15-class-merge-check.ipynb` | Re-scores the predictions with overlapping classes merged or removed, to show how much the overlap limits macro F1. A view of a changed task, not a result. |

#### `03_graph_models/`

| Notebook | What it does |
|---|---|
| `unsw-nb15-gnn.ipynb` | Builds the flow graph ((IP, port) nodes, flows as edges), draws graph visuals, trains the first E-GraphSAGE (neighbour sampling) and fuses it with the tabular model. |
| `unsw-nb15-gnn-density.ipynb` | Matches train graph density to the test graph and runs the no-graph and shuffled-edge controls (5 seeds). |
| `unsw-nb15-gnn-weights.ipynb` | Retrains the GNN with the full balanced class weights instead of the square-root weights. |
| `unsw-nb15-egraphsage.ipynb` | E-GraphSAGE closer to the original paper (full-batch, full neighbourhood, linear classifier on node embeddings), with plain and weighted loss and the same controls (3 seeds). Also reports weighted F1. |
| `unsw-nb15-graph-features.ipynb` | Gives LightGBM graph-derived features (endpoint degrees and neighbour feature means) and a shuffled-graph control. |

#### `04_evaluation/`

| Notebook | What it does |
|---|---|
| `unsw-nb15-bootstrap.ipynb` | Bootstrap intervals over the test set and paired differences between conditions, using the saved predictions of the notebooks above. |

#### `05_paper_replication/`

Uses a 5:2:3 stratified random split (train/validation/test) built from the same preprocessed data; choices are made on validation and the test set is used once.

| Notebook | What it does |
|---|---|
| `unsw-nb15-paper-split-replication.ipynb` | The paper's baseline on our data: an E-GraphSAGE-M style network (sampled neighbours, batches of 500), trained past the paper's 10 epochs with training and validation curves. Reports weighted and macro F1 after epoch 10 and at the best validation epoch. |
| `unsw-nb15-paper-split-improved.ipynb` | What improves on it: four LightGBM variants (plain, class weights, two-stage, class weights with class scales tuned on validation) and an improved E-GraphSAGE-M with and without the graph. Bootstrap intervals and paired differences. |
| `unsw-nb15-plain-lightgbm-check.ipynb` | Test F1 of LightGBM against the number of trees, with and without class weights, to check why the unweighted model stops after one tree. |

#### `superseded/`

| Notebook | Why it is kept |
|---|---|
| `unsw-nb15-gnn-ablation.ipynb` | Earlier 3-seed graph ablation on a denser train graph; replaced by `gnn-density`. |
| `unsw-nb15-gnn-vs-tabular.ipynb` | Earlier analysis of why the GNN scores below the tabular model; its questions were re-answered with controls in later notebooks. |

### `Our_Work/documents/`

| File | Contents |
|---|---|
| `UNSW-NB15-Report.docx` | Report for the professor: preprocessing, baselines, model settings and why, metrics, the GNN, results and limits, written in plain English. |
| `UNSW-NB15-GTCN-G-Comparison.docx` | Comparison with the one paper we targeted (GTCN-G, Xu et al., https://arxiv.org/abs/2510.07285): what the paper states and leaves out, how closely we matched it, our results next to its reported numbers, what the comparison does and does not show, and suggested wording. |
| `UNSW-NB15-Literature-Comparison.docx` | 14 related papers in a table (dataset, split, method, metric, result, how much we read), how close each is to our setup, and our findings matched against them. Half of the papers are marked "Abstract only" and need their full text opened before citing. |
| `analysis.md` | Chronological trial log with every experiment, number and correction. The most detailed record. |

### `Our_Work/figures/`

Each folder holds the charts (`fig_NN.png`) and per-run CSVs saved by the notebook of the same name.

| Folder | From notebook | Contents |
|---|---|---|
| `design_controls_figures/` | design-controls | `fig_01.png` macro F1 of the five LightGBM variants; `design_controls_runs.csv` per-seed scores. |
| `ablation_figures/` | ttl-ablation | `fig_01.png`, `fig_02.png` with and without TTL features. |
| `merge_check_figures/` | class-merge-check | `fig_01.png`, `fig_02.png` merged-class views. |
| `gnn_figures/` | gnn | `fig_01.png` to `fig_11.png`: graph visuals (for example who attacks whom), training curves and results of the first GNN. |
| `gnn_density_figures/` | gnn-density | `fig_01.png` macro F1 by condition; `gnn_density_runs.csv` per-run scores. |
| `gnn_weights_figures/` | gnn-weights | `fig_01.png`; `gnn_weights_runs.csv`. |
| `egraphsage_figures/` | egraphsage | `fig_01.png` macro and weighted F1 by condition; `egraphsage_runs.csv` per-run scores incl. per-class F1; `egraphsage_summary.csv` means and standard deviations; `boot_local.out` bootstrap output for these runs. |
| `graph_features_figures/` | graph-features | `fig_01.png`, `fig_02.png`; `graph_feature_runs.csv`. |
| `paper_split_figures/` | paper-split-replication | `fig_01.png`, `fig_02.png` training and validation curves; `fig_03.png` weighted F1 against the paper's reported values; `fig_04.png` per-class F1; per-run, per-class and history CSVs. |
| `paper_split_improved_figures/` | paper-split-improved | `fig_01.png`, `fig_02.png` GNN curves; `fig_03.png` weighted and macro F1 of every condition; `fig_04.png` per-class F1 heatmap; run, summary, bootstrap and paired-difference CSVs. |
| `bootstrap_figures/` | bootstrap | `fig_01.png` every condition with its interval; `fig_02.png` paired differences; `bootstrap_scores.csv`, `bootstrap_differences.csv`. |
| `gnn_ablation_figures/`, `gnn_vs_tabular_figures/` | superseded notebooks | Figures and CSVs of the two earlier analyses. |

## Reproducing the results

1. Download the raw UNSW-NB15 CSV files (see the dataset's project page, https://research.unsw.edu.au/projects/unsw-nb15-dataset) and run `unsw-nb15-preprocessing` on them. It writes `train.parquet`, `test.parquet`, `train_smote.parquet`, `train_graph.csv`, `test_graph.csv` and `preprocessing.pkl`. These files are not in this repository. Upload them as a Kaggle dataset; we used `shahiismyname/unsw-nb15-preprocessed-dataset`, and the notebooks read it from `/kaggle/input/datasets/shahiismyname/unsw-nb15-preprocessed-dataset`, so change that path if your dataset has another name.
2. Run a notebook with the dataset attached, for example with the Kaggle CLI: `kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4` (GPU needed for the GNN notebooks; `unsw-nb15-egraphsage` takes a few hours for 24 runs).
3. Notebooks that save test predictions (`design-controls`, `gnn-density`, `gnn-weights`, `egraphsage`) feed `unsw-nb15-bootstrap` and the class-merge check.

Each GNN notebook has a `DEBUG` switch at the top for a quick smoke test on a small sample.

## Limits

One dataset and one random split (80/20, plus a 5:2:3 split for the paper replication, which is not an exact replication because several settings of the paper are not stated and its data has 700,001 flows against our 2,042,340); 3 to 5 seeds (the LightGBM seeds differ only through the binning sample); LightGBM is unstable with many trees and relies on early stopping; the final tabular model was chosen using the test set; Worms has only 34 test rows; the GNNs were not tuned and the paper-style one was trained for at most 500 full-batch epochs; the literature check is partly abstract-only. See `analysis.md` for the details.
