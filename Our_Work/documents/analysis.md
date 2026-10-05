# UNSW-NB15 Project: Analysis and Trial Log

A record of what we tried, what we observed, and why each next step followed. Everything was run on Kaggle and every notebook is in this folder. Results are macro F1 on one fixed held-out test set (408,468 flows, 10 classes). "±" is the standard deviation over training seeds. "95% CI" is a bootstrap interval over the test flows (1,000 resamples, notebook `unsw-nb15-bootstrap`); it reflects test-set sampling noise, not a different train/test split. Where we draw a reading from a result, we say how far we think it goes.

## Results at a glance

| Model | Macro F1 | 95% CI |
|---|---|---|
| Two-stage LightGBM, original data, balanced class weights | 0.670 (± 0.004) | 0.658 - 0.679 |
| Single-stage LightGBM, original data, balanced class weights | 0.658 (± 0.002) | 0.646 - 0.669 |
| Two-stage LightGBM, original data + SMOTE rows | 0.654 (± 0.002) | 0.641 - 0.667 |
| Single-stage LightGBM, original data + SMOTE rows | 0.648 (± 0.001) | 0.634 - 0.659 |
| Two-stage LightGBM without the 3 TTL features | 0.653 (± 0.003) | |
| Single-stage LightGBM, no class weights (like the first baseline) | 0.582 (± 0.004) | 0.573 - 0.591 |
| Two-stage LightGBM + graph features | 0.671 (± 0.007) | |
| GNN (E-GraphSAGE), matched graph density | 0.585 (± 0.010) | 0.576 - 0.595 |
| Same GNN, graph switched off | 0.569 (± 0.015) | 0.556 - 0.581 |
| Same GNN, shuffled edges | 0.562 (± 0.016) | 0.552 - 0.571 |

What the evidence suggests (details and caveats in the sections below):
1. Class weighting made the largest difference we measured (+0.076 for single-stage LightGBM). The two-stage design added a smaller gain (+0.011), and adding synthetic SMOTE rows lowered the score slightly (-0.011 to -0.015).
2. Analysis and Backdoor stayed near F1 0.17 in every model. About 80% of their rows have another row with identical features and a different label, which is probably a major reason.
3. With train and test graphs of equal density, the real graph gave the GNN a small gain over the same network without the graph (+0.017). The same kind of information gave LightGBM no measurable gain. The GNN stayed about 0.08 below the best tree model.

---

## 1. Preprocessing

We loaded the 4 raw CSVs, ran EDA, removed duplicates (2,540,047 to 2,042,340 rows), made a stratified 80/20 split, cleaned and selected features (fit on train only), transformed, and exported. Outputs: `train.parquet`, `test.parquet` (original balance), `train_smote.parquet`, `train_graph.csv` / `test_graph.csv` (endpoints), `preprocessing.pkl`.

- **Leakage control:** every learned value (medians, rare levels, feature ranking, scaler, SMOTE) is fit on train only; test missing values use the train median; duplicates are removed before the split.
- **Features:** we dropped IP addresses, sequence numbers, timestamps and the binary `Label`. We dropped redundant columns by measured correlation: `dwin`/`swin`, `dloss`/`dbytes`, `dloss`/`Dpkts` (≥ 0.99), `sloss` (0.960 with `sbytes`; first described as ≥ 0.99 and corrected), `tcprtt` (sum of `synack` and `ackdat`), `ct_ftp_cmd`, one near-constant column, and `state`, `is_ftp_login` (bottom quartile on all four rankings). 41 columns became 33; after one-hot encoding the models see 45 features. A model on the 33 selected features scored 0.5954 validation macro F1 against 0.5924 for all candidates.
- Skewed columns: `log1p` then standardise; other numeric columns standardised; source port in 4 bins; rare levels (< 0.5%) merged into `Other`.
- Shapes: train (1,633,872, 45), test (408,468, 45), train_smote (1,645,669, 45). The first 1,633,872 rows of `train_smote` are exactly `train.parquet`, followed by 11,797 synthetic rows.

## 2. First baselines

We trained Random Forest, LightGBM, XGBoost and an MLP on `train_smote.parquet`: macro F1 0.596, 0.600, 0.636 and 0.559 (accuracy 0.97 to 0.98). Accuracy is high because Normal is 95% of the test set. The low macro F1 came mainly from precision on rare classes (Analysis 0.09 to 0.13, Backdoor 0.11 to 0.17, DoS 0.27 to 0.44), with many rows predicted as Exploits.

Two weaknesses of this baseline were found later (section 5):
- The LightGBM was trained with `lgb.train` and a `class_weight` key in the params dict. That key only exists in the scikit-learn wrapper, so the model was effectively unweighted. Re-running it without weights gives 0.582, in line with the original 0.57 to 0.60. RF, XGBoost and the MLP did use weights.
- Early stopping used a validation slice cut from the oversampled data, so validation rows could have synthetic neighbours in training.

## 3. Two-stage model and ensemble

Stage 1 (LightGBM, binary) separates attack from Normal. Stage 2 (LightGBM, 9-way) is trained on attack rows only, so Normal does not compete for its splits. On the oversampled data it scored 0.648, above each single model; a soft-vote ensemble of RF + LightGBM + XGBoost scored 0.627. Because the single-stage LightGBM was unweighted, part of this early gain may have been class weighting; section 5 separates the two.

## 4. Changing the oversampling method

We replaced SMOTENC with BorderlineSMOTE and a 5x cap. Macro F1 was identical to three decimals for every tree model (two-stage 0.648 both times). The change touched about 0.26% of the training rows, so this test mainly shows that the choice between these two methods did not matter. It did not test training without synthetic rows (section 5).

## 5. Controlled comparison: class weights, two-stage design, synthetic rows

(`unsw-nb15-design-controls`, 3 seeds.) Five LightGBM variants with identical settings (learning rate 0.08, 63 leaves, up to 300 rounds, early stopping 20). The validation slice is 15% of the original training rows, real rows only, and is identical across variants for a given seed; synthetic rows, when used, are added to the training part only.

| Variant | macro F1 | 95% CI |
|---|---|---|
| single-stage, unweighted | 0.582 ± 0.004 | 0.573 - 0.591 |
| single-stage, weighted | 0.658 ± 0.002 | 0.646 - 0.669 |
| two-stage, weighted | 0.670 ± 0.004 | 0.658 - 0.679 |
| single-stage, weighted, + SMOTE rows | 0.648 ± 0.001 | 0.634 - 0.659 |
| two-stage, weighted, + SMOTE rows | 0.654 ± 0.002 | 0.641 - 0.667 |

Paired differences (95% bootstrap CI):
- Class weights, single-stage: +0.076 (+0.065 to +0.087).
- Two-stage instead of single-stage, both weighted: +0.011 (+0.005 to +0.019); +0.014, +0.007, +0.012 in the three seeds.
- Original data instead of original + SMOTE rows: +0.015 two-stage (+0.007 to +0.024), +0.011 single-stage (+0.004 to +0.018).
- Without Worms the same four differences are +0.036, +0.005, +0.007 and +0.004: smaller, same sign.

In these runs, class weighting seems to account for most of the gap we first attributed to the two-stage design, and the synthetic rows did not help. A possible reason is that balanced weights already handle the imbalance and the synthetic rows add some noise; we did not test this.

**Selection note.** We picked the two-stage weighted variant because it had the highest test macro F1 among these five. No hyper-parameter was tuned on test, but the choice used test scores, so 0.670 may be slightly optimistic. No separate untouched test set was kept.

## 6. Where the remaining errors are

**Identical features, different labels** (`unsw-nb15-preprocessing`). After exact duplicates are removed, any remaining group of rows with identical features carries different labels, and a model that sees only these features has no basis to prefer one label for them. Share of each class's rows with such a twin from another class: Backdoor 81.6%, Analysis 78.3%, DoS 33.9%, Reconnaissance 11.9%, Fuzzers 8.3%, Generic 8.2%, Exploits 7.4%, Worms 7.0%, Shellcode 0%, Normal 0%. The same holds on the selected features (82.5%, 78.2%, 33.2% for the top three). The groups are mixes of Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic and Reconnaissance. The two classes with the highest overlap are the two with the lowest F1 (about 0.17) in every model, so we read overlap as a likely major reason; we did not rule out other causes. Identifiers and timestamps were excluded on purpose, so the check says nothing about whether they would separate these rows. Earlier papers report the same group of confused classes.

**TTL features** (`unsw-nb15-ttl-ablation`, two-stage model, 3 seeds). Sarhan et al. drop `sttl`, `dttl` and `ct_state_ttl` as possible testbed artifacts; we keep them.

| Features | macro F1 | excluding Normal |
|---|---|---|
| all 45 | 0.6695 ± 0.0036 | 0.6335 |
| without the 3 TTL features | 0.6526 ± 0.0030 | 0.6148 |

The drop is about 0.017, the largest per-class drops being DoS -0.04, Worms -0.08 (34 rows) and Exploits -0.014. The model still works without them, but part of its score depends on them, so we think both numbers should be reported. (An earlier run on the oversampled data gave 0.653 vs 0.632.)

**Merged-class view** (`unsw-nb15-class-merge-check`, no retraining, predictions of the best model relabelled): 10 classes 0.670; Analysis + Backdoor merged (9) 0.750; folded into Exploits (8) 0.787; removed from scoring (8) 0.795. This changes the task, so it only shows how much two classes affect the average (roughly 0.08 to 0.13); the other eight classes average about 0.79. All results in this log use the 10-class task.

Per-class F1 of the best tabular model: Normal 0.993, Generic 0.930, Shellcode 0.879, Reconnaissance 0.860, Exploits 0.827, Worms 0.787 (34 rows), Fuzzers 0.611, DoS 0.458, Backdoor 0.179, Analysis 0.172.

## 7. GNN

**Design** (`unsw-nb15-gnn`). Edge classification with a 2-layer E-GraphSAGE in plain PyTorch. Each flow is an edge and each (IP, port) pair is a node; the data has only 49 distinct IPs, so an IP-only graph would be 49 nodes with millions of edges each. An edge is classified from the two node embeddings plus its own 45 features. Settings: hidden size 64 (about 46,000 parameters), neighbours sampled 8 and 5 with replacement, Adam (learning rate 0.002, weight decay 1e-5), batch 2,048, up to 12 epochs with early stopping (patience 3) on validation macro F1, 15% of training edges held out, square-root balanced class weights, 5 sampling passes at test time. The train graph is built from train flows only and the test graph from test flows only; labels are not used in message passing; the GNN trains on `train.parquet`. Graph visuals are in `gnn_figures/`: attacks come from 4 source IPs to 10 targets and almost every attacker reaches every target, so IPs alone do not separate attack types.

**First results.** One GPU run gave 0.571 while the tabular model scored 0.654 at the time. Averaging both 50/50 (weight fixed beforehand) gave 0.651, and no weight in a sweep beat the tabular model alone.

**A density mismatch we noticed late** (`unsw-nb15-gnn-density`, GPU, 5 seeds). The train graph holds all training flows and is denser than the test graph: 43.1% of train nodes have a single flow against 78.0% in the test graph (mean degree 3.2 vs 1.9). We split the train graph into 4 random test-sized chunks, stored as one graph of disconnected pieces; this gives 77.9% single-flow nodes and mean degree 1.9, matching the test graph.

| Condition | test macro F1 | validation macro F1 |
|---|---|---|
| no graph (flow features only) | 0.569 ± 0.015 | 0.576 |
| shuffled edges, matched density | 0.562 ± 0.016 | 0.567 |
| real graph, dense (full) train graph | 0.579 ± 0.009 | 0.588 |
| real graph, matched density | 0.585 ± 0.010 | 0.594 |

- Real graph vs no graph (matched density): +0.017 (CI +0.008 to +0.024), positive in all 5 seeds on test (+0.010, +0.025, +0.004, +0.020, +0.024) and on validation; +0.010 without Worms. Real graph vs shuffled edges: +0.024 (CI +0.018 to +0.029).
- Matched density vs the dense train graph: +0.006 (CI +0.001 to +0.013), no difference without Worms. The earlier 3-seed comparison (+0.009, one negative seed, dense graph) was in hindsight too small a test to be conclusive.
- Our reading: the real connections carry a small amount of useful information for this network, on the order of +0.01 to +0.02. We cannot say how this changes with a different graph definition. A second design that follows the original E-GraphSAGE paper more closely (section 8) did not show this gain.

**Why the GNN scores below the tree model.** The best tree model is 0.084 above the GNN (CI 0.073 to 0.095; 0.040 without Worms). We checked three explanations:
- *Two-stage design:* a weighted single-stage tree model is 0.090 above the GNN without graph (CI 0.076 to 0.103), so the design does not appear to explain the gap.
- *Class-weight strength:* with the full balanced weights that LightGBM uses (`unsw-nb15-gnn-weights`, 5 seeds), the GNN scored lower (0.544 with graph, 0.521 without); the graph still added +0.023.
- *Graph density:* worth +0.006, far less than the gap.

Worms accounts for much of the difference: 0.79 for the tree model, 0.31 for the GNN with graph (0.23 without), on 34 test rows. What remains is probably the network being less suited than boosted trees to these tabular features, but we did not test this directly (architecture, optimiser and per-feature tuning were not varied).

GNN per-class F1 (matched density, mean of 5 seeds): Normal 0.992, Generic 0.878, Reconnaissance 0.849, Shellcode 0.790, Exploits 0.785, Fuzzers 0.582, Worms 0.305, Backdoor 0.232, DoS 0.232, Analysis 0.211. The GNN is above the tree model on Backdoor and Analysis. We first read this as an effect of neighbour information, but the same network without a graph already scores 0.21 on Backdoor, so most of the difference seems to come from how the network trades recall and precision. Merging or removing Analysis and Backdoor from the scores did not narrow the gap to the tree model (0.10 for 10 classes, about 0.13 for the merged or removed views).

## 8. E-GraphSAGE closer to the original paper

(`unsw-nb15-egraphsage`, GPU, 3 seeds.) Lo et al. (2022) report E-GraphSAGE on four IoT datasets, not on UNSW-NB15. In the text we read they compare with classifiers from other papers but do not run a version of their own model without the graph. To see how that recipe behaves here we implemented it more closely than the network of section 7: the whole graph is processed at once (no neighbour sampling), each node averages over its full neighbourhood, 128 hidden units, dropout 0.2, constant node feature, and a single linear classifier on the two node embeddings only (the flow's own features reach the classifier only through the embeddings). Differences from the paper: at most 500 full-batch epochs with early stopping on validation macro F1 (the paper trains much longer), a learning rate of 0.003 that we chose without tuning, and our own split. The graph conditions are the same as in section 7; each is run with plain cross-entropy (as in the paper) and with square-root class weights. "No graph" means every flow is isolated, so its embedding depends on its own features only.

| Condition | macro F1, plain loss | macro F1, sqrt class weights | weighted F1 (both ~) |
|---|---|---|---|
| no graph | 0.523 ± 0.001 | 0.600 ± 0.004 | 0.981 / 0.980 |
| real graph, full train graph | 0.528 ± 0.006 | 0.594 ± 0.008 | 0.981 / 0.980 |
| real graph, matched density | 0.514 ± 0.007 | 0.592 ± 0.010 | 0.981 / 0.979 |
| shuffled edges, matched density | 0.511 ± 0.006 | 0.564 ± 0.019 | 0.980 / 0.978 |
| two-stage LightGBM (for reference) | 0.670 | 0.670 | 0.9825 |

- **Real graph vs no graph.** We saw no gain in this design. With class weights, matched density minus no graph is -0.008 (95% CI -0.016 to -0.001; -0.003 without Worms, CI -0.008 to +0.001); with the plain loss it is -0.010 (CI -0.014 to -0.005). The full (denser) train graph minus no graph is -0.006 (CI -0.014 to +0.001) with weights and +0.005 (CI -0.001 to +0.009) without. Section 7 found +0.017 for a network whose classifier also receives the flow's own features and uses sampled neighbours. We did not test which of these differences matters.
- **Real vs shuffled edges.** +0.029 with class weights (CI +0.014 to +0.041; +0.019 without Worms) and +0.002 with the plain loss (CI -0.001 to +0.007). The shuffled runs were still improving when training stopped (validation F1 was rising until epoch 450 to 500), so part of this gap may be slower training on a graph with misleading neighbours rather than lower information.
- **Class weights.** For the no-graph network, square-root weights give +0.076 macro F1 over the plain loss (CI +0.065 to +0.088; +0.047 without Worms), about the same size as the effect we measured for the tree model in section 5.
- **Weighted F1 does not separate these models.** It is between 0.978 and 0.982 for every condition and 0.9825 for the tree model, while macro F1 differs by up to 0.16 across the same models. The GNN papers we found for UNSW-NB15 report weighted F1, so our macro F1 values cannot be compared with theirs.
- **Gap to the tree model.** Two-stage LightGBM is 0.070 above the weighted no-graph network (CI +0.060 to +0.081; +0.028 without Worms) and 0.077 above the weighted real-graph network (CI +0.067 to +0.088).

Our reading: with this recipe and training budget the real neighbours did not help beyond a network that sees each flow alone, and random neighbours did not help either. Many runs chose an epoch between 400 and 500 (the limit was 500), so longer training could raise all scores; we did not test that.

## 9. Graph features for the tree model

(`unsw-nb15-graph-features`, 3 seeds.) We gave LightGBM the graph information directly: the number of flows at each endpoint and the mean of the other flows' features at each endpoint (92 extra columns, no labels). Train features were computed inside 4 random test-sized chunks so train and test graphs have the same density. The same columns on a shuffled graph are the control.

| Condition | macro F1 |
|---|---|
| flow features only (45) | 0.6695 ± 0.0036 |
| + graph features | 0.6709 ± 0.0069 |
| + graph features, shuffled edges | 0.6649 ± 0.0058 |

Real minus flow-only: -0.005, +0.003, +0.006 (mean +0.001), so we saw no measurable benefit. Real minus shuffled: +0.007, +0.002, +0.009. In our experiments the graph helped the neural network a little and the tree model not at all.

## 10. Relation to the literature

Our 10-class macro F1 of about 0.67 is in a plausible range, but protocols differ (we remove duplicates, use all 2.5M flows and a random split), so it is not a leaderboard comparison. The overlap finding is in line with Zoghi and Serpen and Xie et al., who report the same group of confused classes. Our GNN (0.57 to 0.59) is above the one published 10-class GNN on UNSW-NB15 we found (0.36 to 0.41, 700k sample, 2 epochs) and below our tabular model, which fits the suspicion that graph models suit this dataset poorly. Published GNN results on UNSW-NB15 (for example GTCN-G) are weighted F1, which was 0.98 for all our models and so cannot show these differences; none of the GNN papers we read ran a no-graph control of their own model. Documented options for the overlap are removing the classes, multi-label relabelling and post-hoc probability correction; we did not try them.

## 11. Limits

- One train/test split; 3 seeds for tree models, 5 for the GNN. Bootstrap intervals show test-set sampling noise (and average over seeds), not the effect of a different split.
- The final model was chosen using test macro F1 among five variants (section 5), so 0.670 may be slightly optimistic.
- Worms has 34 test rows and moves macro F1 by several hundredths; we give results without Worms where it matters.
- One graph definition ((IP, port) nodes, no time information) and two GNN architectures without tuning. The paper-style network (section 8) was trained for at most 500 full-batch epochs, many runs stopped near that limit, and its learning rate was not tuned. We ruled out three explanations for the gap to the tree model but did not identify the remaining cause.
- The merged-class numbers are a view of a changed task, not results.
- Sections 2 to 4 quote numbers measured at the time, with the two weaknesses described in section 2; later sections use the corrected pipelines.

## Files

Notebooks: `unsw-nb15-preprocessing`, `unsw-nb15-baseline`, `unsw-nb15-gnn`, `unsw-nb15-design-controls`, `unsw-nb15-ttl-ablation`, `unsw-nb15-class-merge-check`, `unsw-nb15-gnn-density`, `unsw-nb15-gnn-weights`, `unsw-nb15-egraphsage`, `unsw-nb15-graph-features`, `unsw-nb15-bootstrap`, and the earlier `unsw-nb15-gnn-vs-tabular` and `unsw-nb15-gnn-ablation` (the latter superseded by `gnn-density`). Notebooks are in `Our_Work/notebooks/` (the two earlier ones in `superseded/`). Figures and per-run CSVs are in the matching `*_figures/` folders under `Our_Work/figures/`. The Word report and the literature comparison are in `Our_Work/documents/` (`UNSW-NB15-Report.docx`, `UNSW-NB15-Literature-Comparison.docx`).
