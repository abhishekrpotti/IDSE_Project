# Bank Marketing Subscription Prediction

This project predicts whether a client subscribed to a term deposit (`y = yes`) using the UCI Bank Marketing dataset. The intended prediction is made before a marketing call, so `duration` is excluded because it is known only after the call.

## Objectives

- Audit the source data and describe its distributions, special values, and class imbalance.
- Compare a majority-class baseline, Logistic Regression, Random Forest, and HistGradientBoosting.
- Select and evaluate a tuned Random Forest with a validation-selected decision threshold.
- Keep preprocessing inside scikit-learn pipelines and reserve the test partition for one final evaluation.

## Dataset

The legacy `bank-full.csv` file contains 45,211 records and 17 columns (16 inputs plus target `y`). It is semicolon-separated and has no actual missing values. The target is imbalanced: 5,289 `yes` and 39,922 `no` records (11.70% positive). See [data acquisition and attribution](data/README.md) and the [data dictionary](docs/DATA_DICTIONARY.md).

The source is the [UCI Machine Learning Repository Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), licensed under CC BY 4.0. Cite Moro, Rita, and Cortez (2014) when reusing it.

## Methods and findings

The audit found zero blank cells and zero exact duplicate rows. The source's `unknown` labels are retained as categories in the primary workflow. `pdays = -1` denotes no previous contact and is treated as a special sentinel; it is not an ordinary negative elapsed-time value. Numeric distributions are skewed, especially `balance` and `campaign`; outliers are retained. The report includes EDA, model comparison, threshold analysis, test confusion matrix, and validation permutation importance figures.

The primary features are `age`, `balance`, `day`, `campaign`, `pdays`, `previous`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, and `poutcome`. The target is `y`. `duration` is excluded from all primary models due to post-call leakage. The availability of `campaign`, `contact`, `day`, and `month` at a real scoring time should be confirmed before deployment.

Initial model comparison used a stratified 70/15/15 train/validation/test split (seed 42) and five-fold shuffled StratifiedKFold on training data. A 20-combination RandomizedSearchCV tuned the Random Forest using training-only cross-validation and Average Precision. Model and threshold selection used validation data; preprocessing was fitted within pipelines. The tuned Random Forest was selected, with threshold **0.25** chosen by maximum validation F1.

### Final test results

| Metric | Test result |
|---|---:|
| Average Precision | 0.4624 |
| ROC-AUC | 0.7926 |
| Precision | 0.4647 |
| Recall | 0.5057 |
| F1-score | 0.4843 |
| Balanced accuracy | 0.7143 |
| Accuracy | 0.8741 |

Test confusion matrix (rows: actual `no`, `yes`; columns: predicted `no`, `yes`):

```text
[[5527, 462],
 [ 392, 401]]
```

Accuracy alone obscures minority-class performance; the threshold reflects a validation-set precision/recall trade-off, not a universal optimum. The held-out results do not establish causal effects or guarantee performance on future campaigns.

## Repository contents

```text
README.md                         Project overview and run instructions
requirements.txt                  Runtime dependencies
data/README.md                    Dataset download and license
docs/DATA_DICTIONARY.md           Feature definitions and timing notes
src/                              Audit, EDA, comparison, optimization scripts
reports/FINAL_PROJECT_REPORT.md   Final academic report
reports/figures/                 Figures referenced by the report
presentation/                     Final 14-slide presentation
```

Raw data and generated model checkpoints/tables are not committed. Each script recreates its corresponding intermediate tables, plots, and reports. Run scripts in order because Phase 4 consumes split indices and candidate artifacts produced by Phase 3.

## Installation and reproduction

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the legacy dataset as described in [data/README.md](data/README.md), and place both `bank-full.csv` and `bank-names.txt` in `data/raw/`. Preserve the source CSV unchanged. Then run, from the repository root:

```bash
python src/audit_dataset.py
python src/phase2_eda.py
python src/train_models.py
python src/phase4_optimization.py
```

All scripts resolve paths relative to the repository. Random seed 42 is used for splitting, cross-validation, and stochastic estimators. Phase 3 creates the saved row indices and candidate artifacts needed by Phase 4. Phase 4 performs the targeted search and its one final test evaluation; do not use test results to revise the model or threshold. Re-running Phase 4 repeats that evaluation, so use the supplied reports for the recorded project result unless intentionally reproducing the complete workflow.

The scripts write reproducible intermediate CSVs, plots, model artifacts, and phase reports under `reports/` and `models/`; these generated outputs are excluded from the clean submission. The final report and presentation are included as completed deliverables.

## References

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306).

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
