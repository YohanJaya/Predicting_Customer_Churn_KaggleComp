Here’s a polished **README / Project Guide** for your CatBoost + Optuna + Feature Engineering pipeline, formatted for GitHub or project documentation:

---

# Churn Prediction Pipeline – CatBoost + Optuna + Feature Engineering

This project demonstrates a **full tabular classification workflow** for churn prediction, combining **data preprocessing, feature engineering with gplearn, feature selection, and hyperparameter tuning with Optuna** for CatBoost.

---

## 1. Project Overview

The workflow follows these steps:

1. **Data Loading & Exploration**

   * Load `train.csv` and `test.csv`
   * Explore shapes, head/tail, data info, descriptive statistics
   * Visualize feature distributions:

     * Numerical features → Histograms + KDE
     * Categorical features → Count plots

2. **Data Preprocessing**

   * Convert `SeniorCitizen` to categorical (`object`)
   * Map binary categorical features (`Yes` → 1, `No` → 0)
   * One-hot encode remaining categorical variables

3. **Feature Engineering (gplearn)**

   * Use `SymbolicTransformer` to create **new features** based on mathematical combinations
   * Transform train and test sets
   * Append new features to original datasets

4. **Feature Selection**

   * Use `mutual_info_classif` to select **top 20 features**
   * Combine with important domain features (payment method, device protection, tech support)

5. **Class Imbalance Handling**

   * Compute `scale_pos_weight` = #negative / #positive examples
   * Use in models supporting imbalance handling (CatBoost, LightGBM, XGBoost, Random Forest, SVM)

---

## 2. Exploratory Visualizations

**Numerical features:**

```python
sns.histplot(dfTrain[num_col], kde=True)
```

**Categorical features:**

```python
sns.countplot(x=dfTrain[cat_col])
```

---

## 3. Modeling & Hyperparameter Tuning

### 3.1 CatBoost Classifier

**Why CatBoost?**

* Handles categorical features natively → avoids one-hot encoding
* Built-in regularization + ordered boosting → reduces overfitting
* Strong defaults → often outperforms LightGBM/XGBoost
* Supports GPU acceleration

**Key Hyperparameters & Practices**

| Parameter             | Role / Notes                                         |
| --------------------- | ---------------------------------------------------- |
| iterations            | 400–2500, use with early stopping                    |
| depth                 | 4–8, deeper trees → risk of overfitting              |
| learning_rate         | 0.005–0.12, log scale for stability                  |
| l2_leaf_reg           | 0.5–12, regularization to fight overfitting          |
| border_count          | 32–128, number of bins for numeric features          |
| random_strength       | 0.5–8, injects noise for generalization              |
| bagging_temperature   | 0–1, randomness in bagging                           |
| early_stopping_rounds | 50–100, prevents overfitting                         |
| scale_pos_weight      | pos/neg ratio, handles class imbalance               |
| cat_features          | List of categorical columns, crucial for CatBoost    |
| use_best_model        | True, keeps only best iteration after early stopping |

**Best Practices**

* Always provide `cat_features`
* Use early stopping with hold-out validation
* Moderate tuning → defaults often perform well
* Ensemble with LightGBM/XGBoost for leaderboard improvement

---

### 3.2 Optuna for Hyperparameter Optimization

* **Pruning:** `MedianPruner`

  * Stops underperforming trials early → saves compute
  * Parameters: `n_startup_trials=5`, `n_warmup_steps=5`

**Outer cross-validation (5-fold Stratified):**

* Split train data
* Inner Optuna study per fold → tune CatBoost hyperparameters
* Fit best model per fold → evaluate AUC on validation fold
* Aggregate results → select final best parameters

---

## 4. Model Training & Predictions

**Final Model Training:**

```python
final_model = CatBoostClassifier(**best_params, early_stopping_rounds=80, verbose=100, eval_metric='AUC')
final_model.fit(xTrain, yTrain)
```

**Predictions on Test Set:**

```python
test_probs = final_model.predict_proba(xTest)[:,1]

submission = pd.DataFrame({
    "id": dfTest["id"],
    "Churn": test_probs
})

submission.to_csv("output/submission4.csv", index=False)
```

* Saves predictions to `output/submission4.csv`

---

## 5. Other Models (Optional)

The pipeline supports benchmarking other classifiers with class imbalance handling:

* **XGBoost** → `scale_pos_weight`
* **Random Forest** → `class_weight`
* **SVM** → `class_weight`
* **LightGBM** → `scale_pos_weight`
* **Naive Bayes** → `sample_weight`

Cross-validation metrics tracked: **Precision, Recall, F1, ROC-AUC**

---

## 6. Project Notes

* **Feature Engineering:** gplearn features often improve generalization
* **Feature Selection:** top features selected via mutual information
* **Hyperparameter Tuning:** Optuna with MedianPruner balances exploration + efficiency
* **CatBoost Strength:** Handles categorical variables natively, less preprocessing required

---


