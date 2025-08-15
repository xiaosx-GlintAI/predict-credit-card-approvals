# Predict Credit Card Approvals

This project builds a machine learning model to predict **credit card application approvals** using Logistic Regression with hyperparameter tuning.  
It demonstrates the **full end-to-end ML workflow** — from data acquisition to preprocessing, modeling, and evaluation.

---

## Project Overview
- **Goal:** Predict if a credit card application will be approved based on applicant data.
- **Dataset:** UCI Credit Card Approvals Dataset (mixed categorical & numerical features, missing values present).
- **Model:** Logistic Regression with `GridSearchCV` hyperparameter tuning.
- **Result:** Achieved **~82.9% accuracy** on the test set.

## End-to-end Workflow:

- **Data Acquisition & Exploration:**

   Loaded dataset from a remote source

   Performed exploratory data analysis (EDA) to understand numerical & categorical feature distributions

- **Data Preprocessing:**

  Replaced missing values (?) with NaN

  Type conversion for numerical fields

  Imputation of missing values (mean for numerical, mode for categorical)

  One-hot encoding for categorical variables

  Feature scaling using StandardScaler

- **Modeling:**

  Logistic Regression model as baseline

  Hyperparameter tuning using GridSearchCV with 5-fold cross-validation

  Best parameters found: max_iter=100, tol=0.0001

  Achieved ~82.9% accuracy on the test set

## Skills Demonstrated:

  Data cleaning & preprocessing

  Feature engineering & encoding

  Model selection & evaluation

  Hyperparameter tuning with cross-validation

  Performance reporting with confusion matrix & accuracy score

## Tech Stack:
Python, Pandas, NumPy, Scikit-learn




