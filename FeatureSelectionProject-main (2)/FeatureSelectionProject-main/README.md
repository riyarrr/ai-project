# Feature Selection Project

## 1. Project Overview

This project compares six feature selection methods on two high-dimensional datasets: Colon Cancer and Leukemia.

The goal is to reduce the number of features while evaluating classification performance, feature reduction, and execution time.

## 2. Feature Selection Methods

The project includes six methods:

1. **CFS (Correlation-Based Feature Selection):** Uses feature–class correlation and feature redundancy to guide greedy feature selection.
2. **FCBF (Fast Correlation-Based Filter):** The current implementation ranks features using symmetrical uncertainty and selects the top-k features.
3. **FAST:** The current implementation uses mutual information for relevance ranking and correlation-based redundancy filtering.
4. **GA (Genetic Algorithm):** Uses a population-based search with crossover/mutation-style optimization to find a feature subset.
5. **PSO (Particle Swarm Optimization):** Uses a particle-based search to identify a feature subset.
6. **Hybrid:** Combines mutual-information-based filtering with a genetic-algorithm-based search.

Note: FAST, FCBF, and Hybrid are descriptions of the current project implementations. They should not be assumed to reproduce every step of the corresponding published algorithms.

## 3. Datasets

| Dataset | Samples | Original Features |
|---|---:|---:|
| Colon Cancer | 62 | 2,000 |
| Leukemia | 72 | 7,128 |

The Colon Cancer dataset has 40 samples in one class and 22 in the other. The prepared Leukemia dataset has 47 ALL samples and 25 AML samples.

Dataset files are stored in the `datasets/` directory.

## 4. Evaluation

The current experiments use an 80/20 stratified train/test split with random state 42 and a K-Nearest Neighbors classifier (k = 5).

Standard scaling is fitted using the training data. Feature selection is performed on the training portion before evaluating on the held-out test set.

The reported metrics are:

- **Accuracy:** Proportion of correctly classified test samples.
- **Matthews Correlation Coefficient (MCC):** A classification metric that accounts for both classes.
- **Selected Features:** Number of features retained by the method.
- **Feature Reduction (%):** `((Total Features - Selected Features) / Total Features) * 100`.
- **Runtime (s):** Recorded execution time in seconds.

GA, PSO, and Hybrid additionally use three-fold cross-validation within the training data during their search. The filter-based methods do not all use the same internal optimization procedure.

`comparison.py` checks dataset missing/duplicate samples, validates all six result files, checks selected feature names and counts against the datasets, and re-evaluates Accuracy and MCC using the stated common held-out split and classifier. It preserves the recorded method runtime (feature selection plus classifier evaluation); runtime is environment-dependent and should not be read as a precise hardware-independent benchmark.

Run the validated comparison and recommendation with:

```bash
python comparison.py
python recommendation.py
```

The comparison is written to `final_comparison.csv`; the dataset-specific recommendations are written to `recommendations.csv`.

### Recommendation rule

Recommendations are weight-free. For each dataset, methods dominated in both Accuracy and MCC are excluded. Among the remaining Accuracy/MCC Pareto methods, the recommendation retains the fewest features (equivalently, the greatest feature reduction); runtime breaks feature-count ties, and method name breaks any exact remaining tie. This separates the predictive frontier from the overall compactness/runtime choice instead of combining unlike quantities with arbitrary weights. Missing or invalid metrics stop recommendation generation with an explicit error.

Users can evaluate custom CSV datasets directly in the interactive web dashboard via **Dataset Upload & Inspection**, or via the standalone command-line workflow:


```bash
python upload_recommendation.py path/to/uploaded_dataset.csv --target diagnosis
```

Replace `diagnosis` with the exact target/class column name from the CSV. If `--target` is omitted, the script displays the CSV columns and prompts for the target. Use `--dataset-name "My dataset"` to change the label shown in output. By default, results are displayed in the terminal and are not saved; optional `--results-output results.csv` and `--recommendation-output recommendation.csv` save the newly computed tables.

The terminal output includes a status for every method, successful method metrics and selected feature names, the recommended method and its metrics, and the recommended feature names.

Only CSV files with unique, non-empty column names are supported. The target must be present, have no missing values, and contain at least two classes. Predictor missing values are imputed (median for numeric and most-frequent for categorical data); categorical values are one-hot encoded, with rare categories grouped by the encoder. Imputers and encoders are fit on the stratified training split only. Feature selectors and the final scaled 5-nearest-neighbor evaluation use training data; the held-out test set is used only for final metrics. The split and stochastic selector seeds use 42.

All six methods are reported as completed, unsupported, or errored. The current CFS implementation is limited to binary data because it factorizes class labels and measures numeric correlation; its dense feature-correlation matrix is also limited to at most 8,000 encoded features. FAST considers up to 100 candidate features, and GA/PSO/Hybrid use their existing three-fold KNN search procedures. The workflow requires enough class-balanced samples for a stratified 80/20 split, three-fold search, and 5-NN. It rejects matrices above 10 million training cells and caps one-hot categories per input column. These safeguards do not make every CSV appropriate or guarantee that every selector will complete; method-level failures are shown and are excluded from the recommendation. If every method fails, the command exits with an error rather than producing a recommendation. The recommendation uses only successful methods' newly measured accuracy, MCC, selected-feature counts, and runtime.

The reported test metrics are still estimates from one small holdout split. They are not evidence that the selected method or model will generalize to new populations; larger datasets and nested/repeated validation are preferable.

## 5. Result Files

Each method writes a CSV result file:

| Method | Result File |
|---|---|
| CFS | `cfs_classical_results.csv` |
| FCBF | `fcbf_results.csv` |
| FAST | `fast_results.csv` |
| GA | `ga_results.csv` |
| PSO | `pso_results.csv` |
| Hybrid | `hybrid_results.csv` |

The comparison script combines these results into `final_comparison.csv`, including dataset, method, total features, selected feature count, accuracy, MCC, runtime, and selected feature names.

| Output | Purpose |
|---|---|
| `final_comparison.csv` | Validated results for each method/dataset, including feature reduction |
| `recommendations.csv` | Recommended method, predictive frontier, metrics, and rationale for each dataset |

## 6. Running the Project

### Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Interactive Streamlit Dashboard (Member 3)

Launch the interactive feature selection and explainability dashboard:

```bash
streamlit run app.py
```

The web dashboard provides six integrated sections:
1. **Overview:** High-level metrics, method categorization (Filter, Wrapper, Hybrid), Pareto recommendation principles, and comprehensive methodology & limitations.
2. **Dataset Upload & Inspection:** In-memory CSV validation, interactive data preview, schema inspection, target class distribution validation, duplicate/missing checks, loading status indicator, and run identifier generation.
3. **Algorithm Comparison:** Interactive KPI cards, Plotly charts for Accuracy, MCC, Feature Reduction, Runtime, Accuracy vs Feature Reduction scatter plots, and filtered exportable tables.
4. **Feature Explorer & Consensus:** Method selection, search, pagination, probe ID/encoded provenance decoding, UpSet-style intersection matrix, selection frequency charts, and CSV downloads.
5. **Recommendation & Explainability:** Step-by-step 4-stage pipeline explainability, Pareto dominance verification table with mathematical elimination proofs, multi-objective trade-off explorer, head-to-head pairwise comparator with metric deltas and Jaccard similarity, and plain-language guide to all six algorithms.
6. **Download Results & Provenance:** Export comparison matrices, Pareto recommendation summaries, per-method feature CSVs, consolidated all-methods feature CSVs, and structured JSON payloads with execution timestamps and run IDs.

### Running Automated Tests

Run the complete project test suite:

```bash
python -m unittest discover -p "test_*.py"
```

### Command-Line Execution

Run individual feature selection scripts or the CLI benchmark pipelines:

```bash
# Generate benchmark comparison and recommendations
python comparison.py
python recommendation.py

# Evaluate a custom uploaded CSV from the CLI
python upload_recommendation.py path/to/dataset.csv --target target_column_name
```

The combined comparison is saved as `final_comparison.csv` and recommendations as `recommendations.csv`.

## 7. Current Results

The validated single-split results recommend Hybrid for both datasets:

- **Colon Cancer — Hybrid:** 92.31% accuracy, 0.8433 MCC, 4 of 2,000 features (99.80% reduction), and 5.0921 s recorded runtime. It is the sole Accuracy/MCC Pareto method in this result set. CFS is faster (0.0267 s), but has lower accuracy and MCC and retains more features.
- **Leukemia — Hybrid:** tied with CFS at 100% accuracy and 1.0000 MCC, while selecting 3 rather than 5 of 7,128 features (99.96% versus 99.93% reduction) and recording 6.2230 s.

The Leukemia methods FCBF and FAST retain just one feature but have lower observed accuracy/MCC (86.67%, 0.7303); higher reduction alone does not outweigh the weaker predictive metrics under the documented recommendation rule. Full per-method results and reasoned recommendations are in `final_comparison.csv` and `recommendations.csv`.

These results describe a single stratified 80/20 train/test split, not repeated cross-validation estimates. The test sets contain only 13 Colon Cancer samples and 15 Leukemia samples, so one or two predictions substantially change the reported metrics. Perfect Leukemia accuracy is not proof of generalization; the very small sample counts and high dimensionality make performance estimates unstable. Feature selection is performed on the training portion, and the held-out test set is not used to select features.

## 8. Limitations

- Results are based on a single stratified 80/20 train/test split snapshot; held-out accuracy does not prove generalization to external clinical populations.
- The small sample sizes ($n=62$ Colon Cancer, $n=72$ Leukemia) and high dimensionality ($p=2,000$ and $p=7,128$) make performance estimates subject to sampling variance.
- Internal search procedures differ between methods (stochastic genetic algorithms vs deterministic information-theoretic filters).
- Feature selection identifies predictive statistical associations, not biological causation or molecular etiology.
- Runtime measurements depend on local hardware and execution environment.

