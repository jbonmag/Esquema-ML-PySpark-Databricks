"""Reference PySpark ML pipeline for customer churn classification."""

from __future__ import annotations

import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small PySpark churn classification pipeline.")
    parser.add_argument("--predictions-output", default=None, help="Optional CSV output path for predictions.")
    return parser


def build_sample_data(spark: SparkSession):
    rows = [
        (1, "mujer", 25, 32000, "Espana", 0),
        (2, "hombre", 45, 57000, "Mexico", 1),
        (3, "mujer", 31, 43000, "Espana", 0),
        (4, "hombre", 29, 25000, "Chile", 0),
        (5, "mujer", 38, 72000, "Mexico", 1),
        (6, "hombre", 35, 50000, "Argentina", 0),
        (7, "mujer", 22, 28000, "Chile", 1),
        (8, "hombre", 39, 61000, "Espana", 0),
        (9, "mujer", 33, 46000, "Mexico", 1),
        (10, "hombre", 28, 34000, "Argentina", 0),
        (11, "mujer", 51, 83000, "Mexico", 1),
        (12, "hombre", 41, 69000, "Chile", 1),
        (13, "mujer", 27, 37000, "Argentina", 0),
        (14, "hombre", 48, 76000, "Espana", 1),
        (15, "mujer", 36, 52000, "Chile", 0),
        (16, "hombre", 24, 29000, "Mexico", 0),
        (17, "mujer", 44, 65000, "Argentina", 1),
        (18, "hombre", 32, 41000, "Espana", 0),
        (19, "mujer", 55, 91000, "Mexico", 1),
        (20, "hombre", 30, 39000, "Chile", 0),
    ]
    columns = ["id_cliente", "genero", "edad", "ingresos", "pais", "churn"]
    return spark.createDataFrame(rows, columns)


def build_pipeline() -> Pipeline:
    genero_indexer = StringIndexer(inputCol="genero", outputCol="genero_idx", handleInvalid="keep")
    pais_indexer = StringIndexer(inputCol="pais", outputCol="pais_idx", handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["genero_idx", "pais_idx", "edad", "ingresos"],
        outputCol="features",
    )
    classifier = LogisticRegression(featuresCol="features", labelCol="churn", maxIter=30)
    return Pipeline(stages=[genero_indexer, pais_indexer, assembler, classifier])


def main() -> None:
    args = build_parser().parse_args()
    spark = SparkSession.builder.appName("CustomerChurnPySparkPipeline").getOrCreate()

    df = build_sample_data(spark)
    train, test = df.randomSplit([0.75, 0.25], seed=42)

    model = build_pipeline().fit(train)
    predictions = model.transform(test)

    evaluator = BinaryClassificationEvaluator(labelCol="churn", rawPredictionCol="rawPrediction")
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")
    predictions.select("id_cliente", "churn", "prediction", "probability").show(truncate=False)

    if args.predictions_output:
        predictions.select("id_cliente", "churn", "prediction", "probability").write.mode(
            "overwrite"
        ).csv(args.predictions_output, header=True)

    spark.stop()


if __name__ == "__main__":
    main()
