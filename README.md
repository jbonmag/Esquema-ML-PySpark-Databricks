# 🔍 Customer Churn Prediction with PySpark

This project demonstrates a complete Machine Learning pipeline using **PySpark** on **Databricks** to predict customer churn based on demographic and behavioral data.

---

## 🧠 Objective

To predict whether a customer will churn (leave the service) based on features such as gender, age, income, and country.

---

## ⚙️ Technologies Used

- [x] Apache Spark (PySpark)
- [x] Databricks Notebooks
- [x] Python 3
- [x] Spark MLlib
- [x] Logistic Regression

---

## 📊 Project Pipeline

1. **Dataset loading** from simulated customer data.
2. **Data exploration and schema inspection**.
3. **Categorical variable indexing** using `StringIndexer`.
4. **Feature assembly** using `VectorAssembler`.
5. **Train/test split** of the dataset.
6. **Model training** using logistic regression.
7. **Model evaluation** using AUC-ROC.
8. **(Optional)** Save predictions to CSV.

---

## 📁 Sample Input Data

```text
| id_cliente | gender | age | income |  country  | churn |
|------------|--------|-----|--------|-----------|--------|
|     1      | female |  25 | 32000  |  Spain    |   0    |
|     2      |  male  |  45 | 57000  |  Mexico   |   1    |
|     3      | female |  31 | 43000  |  Spain    |   0    |
...
```

---

## 📈 Sample Output

```text
| churn | prediction | probability                          |
|-------|------------|--------------------------------------|
|   1   |    0.0     | [0.99999, 0.00001]                   |
|   0   |    0.0     | [0.91881, 0.08118]                   |
...
AUC (Area Under ROC Curve): 0.5
```

---

## 💾 Output (Optional)

The model predictions can be saved as a CSV file:

```bash
/tmp/predicciones_clientes.csv
```

---

## 🚀 How to Run

This notebook is designed to run on **Databricks**. Simply import the notebook into your Databricks workspace and execute each cell sequentially.

---

## 📝 License

This project is licensed under the MIT License.
