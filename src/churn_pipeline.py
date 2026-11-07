"""Reference PySpark MLlib pipeline for customer churn, trained on the IBM Telco Customer Churn dataset."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

# IBM's public sample (Apache-2.0), pinned to the commit that last changed the file.
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
SEED = 42
ID = "customerID"
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]


def download_data(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "telco_customer_churn.csv"
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_customers(spark: SparkSession, path: str) -> DataFrame:
    """Read the CSV with explicit types and a 0/1 label.

    TotalCharges is blank for 11 customers who joined this month (tenure 0); they have paid nothing yet.
    """
    raw = spark.read.csv(path, header=True, inferSchema=False)
    return (
        raw.withColumn("tenure", F.col("tenure").cast("double"))
        .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
        .withColumn("TotalCharges", F.coalesce(F.trim("TotalCharges").cast("double"), F.lit(0.0)))
        .withColumn("SeniorCitizen", F.when(F.col("SeniorCitizen") == "1", "Yes").otherwise("No"))
        .withColumn("label", (F.col(TARGET) == "Yes").cast("double"))
    )


def feature_stages() -> list:
    """Categories become one-hot vectors (country, contract type... have no order); numbers are scaled."""
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL],
                            outputCols=[f"{c}_ohe" for c in CATEGORICAL], handleInvalid="keep")
    numeric = VectorAssembler(inputCols=NUMERIC, outputCol="numeric_raw")
    scaler = StandardScaler(inputCol="numeric_raw", outputCol="numeric_scaled", withMean=True)
    features = VectorAssembler(inputCols=["numeric_scaled"] + [f"{c}_ohe" for c in CATEGORICAL],
                               outputCol="features")
    return [*indexers, encoder, numeric, scaler, features]


def candidate_pipelines() -> dict[str, tuple[Pipeline, list]]:
    """Each model with the parameter grid explored by cross-validation."""
    lr = LogisticRegression(maxIter=100)
    rf = RandomForestClassifier(numTrees=200, seed=SEED)
    gbt = GBTClassifier(seed=SEED)
    return {
        "Logistic regression": (
            Pipeline(stages=feature_stages() + [lr]),
            ParamGridBuilder().addGrid(lr.regParam, [0.001, 0.01, 0.1]).build(),
        ),
        "Random forest": (
            Pipeline(stages=feature_stages() + [rf]),
            ParamGridBuilder().addGrid(rf.maxDepth, [6, 10]).build(),
        ),
        "Gradient-boosted trees": (
            Pipeline(stages=feature_stages() + [gbt]),
            ParamGridBuilder().addGrid(gbt.maxDepth, [3]).addGrid(gbt.maxIter, [50, 100]).build(),
        ),
    }


def fit_with_cv(pipeline: Pipeline, grid: list, train: DataFrame) -> tuple[PipelineModel, float]:
    """Return the best model refitted on all of train, and its mean cross-validated AUC ROC."""
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
                        numFolds=3, seed=SEED, parallelism=4)
    fitted = cv.fit(train)
    return fitted.bestModel, max(fitted.avgMetrics)


def churn_score(predictions: DataFrame) -> DataFrame:
    return predictions.withColumn("churn_probability", vector_to_array("probability")[1])


def evaluate(predictions: DataFrame) -> dict[str, float]:
    """Ranking quality, metrics at the 0.5 threshold, and how many churners the riskiest 20% contain."""
    scored = churn_score(predictions)
    roc = BinaryClassificationEvaluator(metricName="areaUnderROC").evaluate(predictions)
    pr = BinaryClassificationEvaluator(metricName="areaUnderPR").evaluate(predictions)
    counts = {(r["label"], r["prediction"]): r["count"]
              for r in predictions.groupBy("label", "prediction").count().collect()}
    tp, fp, fn = counts.get((1.0, 1.0), 0), counts.get((0.0, 1.0), 0), counts.get((1.0, 0.0), 0)
    tn = counts.get((0.0, 0.0), 0)
    ranked = scored.withColumn("pct", F.percent_rank().over(Window.orderBy(F.desc("churn_probability"))))
    total_churners = tp + fn
    top20 = ranked.filter("pct < 0.2").agg(F.sum("label")).first()[0] or 0
    return {
        "auc_roc": roc,
        "auc_pr": pr,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / total_churners if total_churners else 0.0,
        "churners_in_top_20pct": top20 / total_churners if total_churners else 0.0,
    }


def predictions_for_export(predictions: DataFrame) -> DataFrame:
    """Plain columns only: CSV cannot store the Spark probability vector."""
    return churn_score(predictions).select(ID, "label", "prediction", "churn_probability")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Telco churn CSV (local, DBFS or Volumes path). Downloaded if omitted.")
    parser.add_argument("--data-dir", default="data", help="Where the downloaded CSV is cached.")
    parser.add_argument("--predictions-output", help="Folder to write test predictions as CSV.")
    parser.add_argument("--model-output", help="Folder to save the best fitted pipeline.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    path = args.input or str(download_data(Path(args.data_dir)))
    customers = load_customers(spark, path)
    train, test = customers.randomSplit([0.8, 0.2], seed=SEED)
    train = train.cache()
    print(f"train: {train.count()} customers, test: {test.count()} customers")

    results = {}
    for name, (pipeline, grid) in candidate_pipelines().items():
        model, cv_auc = fit_with_cv(pipeline, grid, train)
        results[name] = (model, cv_auc)
        test_metrics = evaluate(model.transform(test))
        print(f"{name:>24}: cv_auc_roc {cv_auc:.3f}, " + ", ".join(f"{k} {v:.3f}" for k, v in test_metrics.items()))

    # Chosen on cross-validation, never on the test set.
    best_name = max(results, key=lambda n: results[n][1])
    best_model = results[best_name][0]
    print(f"Best model by cross-validated AUC ROC: {best_name}")

    if args.predictions_output:
        predictions_for_export(best_model.transform(test)).write.mode("overwrite").csv(
            args.predictions_output, header=True)
    if args.model_output:
        best_model.write().overwrite().save(args.model_output)
    spark.stop()


if __name__ == "__main__":
    main()
