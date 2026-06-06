# Esquema ML PySpark Databricks

Plantilla de proyecto de machine learning con PySpark para clasificación binaria de churn. El repositorio muestra una estructura mínima pero completa: creación de datos, preprocesamiento, pipeline de MLlib, entrenamiento, evaluación y exportación opcional de predicciones.

## Objetivo

Predecir si un cliente abandonará un servicio a partir de variables demográficas y de comportamiento básicas:

- Género
- Edad
- Ingresos
- País

El dataset incluido en el ejemplo es sintético y sirve para documentar la estructura de trabajo. En un caso real, la misma arquitectura se puede adaptar a datos en Delta, Parquet, CSV o tablas gestionadas en Databricks.

## Estructura

```text
.
├── Esquema ML PySpark Databricks.ipynb
├── src/
│   └── churn_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Pipeline

1. Creación o carga del DataFrame.
2. Inspección del esquema.
3. Indexado de variables categóricas con `StringIndexer`.
4. Ensamblado de variables con `VectorAssembler`.
5. Separación train/test.
6. Entrenamiento con `LogisticRegression`.
7. Evaluación con AUC-ROC.
8. Exportación opcional de predicciones.

## Uso

En local con Spark instalado:

```bash
pip install -r requirements.txt
spark-submit src/churn_pipeline.py
```

En Databricks:

1. Importa el notebook en el workspace.
2. Ejecuta las celdas en orden.
3. Sustituye el dataset sintético por una tabla o ruta real cuando sea necesario.

Para guardar predicciones:

```bash
spark-submit src/churn_pipeline.py --predictions-output outputs/churn_predictions
```

## Adaptación a datos reales

Para usar este esquema con una fuente real:

1. Sustituye la función de creación de datos por `spark.read`.
2. Ajusta la columna objetivo.
3. Revisa las variables categóricas y numéricas.
4. Añade validación de calidad de datos antes del entrenamiento.
5. Guarda el modelo con `model.write().overwrite().save(...)` si se necesita despliegue posterior.

