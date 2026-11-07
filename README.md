# Esquema ML PySpark Databricks

A reference machine-learning pipeline for Spark and Databricks, on real data: predicting which customers of a telephone and internet company will leave (churn), using IBM's [Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) sample (7,043 customers, 19 attributes, 26.5% churn).

| model | AUC ROC, cross-validation | AUC ROC, test | churners among the riskiest 20% |
|---|---|---|---|
| **Logistic regression** (chosen) | **0.847** | 0.843 | 52% |
| Gradient-boosted trees | 0.846 | 0.843 | 50% |
| Random forest | 0.843 | 0.838 | 49% |

Each model is tuned by 3-fold cross-validation on 80% of the customers. The best is chosen by its cross-validated AUC, and only then measured on the other 20% (1,345 customers). The three are practically tied, so the readable linear model is kept.

Calling the 20% of customers the model rates as most at risk reaches 52% of the customers who actually leave, 2.6 times as many as a random selection.

## What the pipeline does

1. **Load** the CSV with explicit types. `TotalCharges` is blank for 11 customers who joined this month; they have not been billed yet, so it is set to 0.
2. **Encode**: `StringIndexer` and then `OneHotEncoder` for the 16 categorical attributes. Contract type or payment method have no order, so they must not become single numbers.
3. **Scale**: `tenure`, `MonthlyCharges` and `TotalCharges` are assembled and standardised.
4. **Compare** logistic regression, random forest and gradient-boosted trees with `CrossValidator`.
5. **Evaluate** on the test set: AUC ROC, AUC PR, precision and recall, and the share of churners among the riskiest 20%.
6. **Export** predictions as CSV (customer, label, prediction, churn probability) and save the fitted pipeline.

## The notebook

[`churn_pipeline.ipynb`](churn_pipeline.ipynb) is saved with its outputs, so it can be read on GitHub without running anything:

1. **Exploration.** Churn rate by contract type, internet service and tenure. Month-to-month contracts churn 43%, two-year contracts 3%.
2. **The previous version.** It used 20 invented customers and left 8 rows for testing, so its AUC measured nothing.
3. **The pipeline.** The steps listed above.
4. **The model comparison.**
5. **A cumulative gains chart** and the logistic regression coefficients. Month-to-month contracts, fiber-optic service and electronic-check payment raise the risk; long contracts and long tenure lower it.
6. **The export schema.**

## Running it

Locally, with Java 17 installed:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
spark-submit src/churn_pipeline.py                                  # or: python src/churn_pipeline.py
spark-submit src/churn_pipeline.py --predictions-output outputs/predictions --model-output outputs/model
```

The data is downloaded on first use from IBM's repository, pinned to a fixed commit, and cached in `data/` (not versioned). A full run took about 5 minutes on a local Spark session with 20 CPU threads.

**On Databricks:**
- Import the notebook and run it on any cluster; the existing `spark` session is used.
- To work with your own table, upload the CSV to a volume and pass its path with `--input`, or replace `load_customers` with `spark.read.table(...)`.
- The output paths can also be volumes.

On Windows, Spark cannot write local files unless Hadoop's `winutils.exe` is installed. Training and evaluation work without it; the two output options need it.

## Adapting it to other data

- Change `CATEGORICAL`, `NUMERIC`, `ID` and `TARGET` in `src/churn_pipeline.py`.
- Adapt `load_customers` to the new source.

The encoding, scaling, model comparison, evaluation and export work unchanged.

## Files

```text
src/churn_pipeline.py      loading, feature stages, models, cross-validation, evaluation, export
churn_pipeline.ipynb       the experiment, executed
requirements.txt
```

The Telco Customer Churn data is © IBM, published under the Apache-2.0 license.
