# PySpark ML Pipeline for Databricks

Reference machine learning project using PySpark for binary churn classification. The repository provides a small but complete workflow: sample data creation, preprocessing, MLlib pipeline construction, training, evaluation and optional prediction export.

## Objective

Predict whether a customer is likely to churn from basic demographic and behavioral variables:

- Gender
- Age
- Income
- Country

The included dataset is synthetic and is meant to document the workflow structure. In a real project, the same architecture can be adapted to Delta, Parquet, CSV or managed Databricks tables.

## Structure

```text
.
|-- Esquema ML PySpark Databricks.ipynb
|-- src/
|   `-- churn_pipeline.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Pipeline

1. Create or load a Spark DataFrame.
2. Inspect the schema.
3. Encode categorical variables with `StringIndexer`.
4. Assemble features with `VectorAssembler`.
5. Split the data into train and test sets.
6. Train a `LogisticRegression` model.
7. Evaluate with AUC-ROC.
8. Optionally export predictions.

## Usage

Local execution with Spark installed:

```bash
pip install -r requirements.txt
spark-submit src/churn_pipeline.py
```

Databricks execution:

1. Import the notebook into a Databricks workspace.
2. Run the cells in order.
3. Replace the synthetic dataset with a real table or file path when needed.

To save predictions:

```bash
spark-submit src/churn_pipeline.py --predictions-output outputs/churn_predictions
```

## Adapting to Real Data

1. Replace the sample data builder with a `spark.read` source.
2. Adjust the target column.
3. Review categorical and numerical feature columns.
4. Add data quality checks before training.
5. Save the trained model with `model.write().overwrite().save(...)` if deployment is required.

## Security

The example does not require credentials. Generated artifacts, models and local outputs are excluded from the repository with `.gitignore`.
