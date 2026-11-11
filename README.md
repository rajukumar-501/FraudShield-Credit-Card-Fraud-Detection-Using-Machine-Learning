# Credit Card Fraud Detection

A beginner-friendly machine-learning project that compares Logistic Regression and Random Forest for identifying potentially fraudulent credit-card transactions.

> **Important:** This repository is a learning project, not a production fraud-detection system. The dataset contains anonymized features and should not be used to make real financial decisions.

## Problem

Fraudulent transactions are rare compared with legitimate transactions. This creates a **class-imbalance** problem: a model can achieve high accuracy by predicting nearly every transaction as legitimate while detecting very little fraud.

The project therefore evaluates models using precision, recall, F1-score, ROC-AUC and a confusion matrix—not accuracy alone.

## What the notebook does

1. Loads and inspects transaction data.
2. Checks missing values and class distribution.
3. Explores transaction amounts and fraud prevalence.
4. Splits the data into training and test sets using stratification.
5. Scales `Time` and `Amount` using statistics learned from the training set.
6. Trains Logistic Regression and Random Forest models.
7. Compares precision, recall, F1-score and ROC-AUC.
8. Visualizes the confusion matrix and Random Forest feature importance.

## Technology

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository structure

```text
credit-card-fraud-detection/
├── notebooks/
│   └── credit-card-fraud-detection.ipynb
├── data/                       # Create this folder locally; dataset not included
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the anonymized `creditcard.csv` dataset from the source you are permitted to use (for example, the well-known Kaggle Credit Card Fraud Detection dataset).
4. Create a `data/` folder in the project root and place `creditcard.csv` inside it.
5. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `notebooks/credit-card-fraud-detection.ipynb` and run the cells from top to bottom.

If your dataset is stored elsewhere, change `DATA_PATH` in the dataset-loading cell.

## Evaluation: what to understand

- **Precision:** Of the transactions flagged as fraud, how many were actually fraud?
- **Recall:** Of all actual fraud transactions, how many did the model detect?
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** How well the model ranks fraud cases above legitimate cases across thresholds.
- **Confusion matrix:** Counts true positives, false positives, true negatives and false negatives.

In fraud screening, low recall can mean missed fraud, while low precision can mean too many legitimate transactions are incorrectly flagged. The appropriate trade-off depends on operational costs.

## Before presenting this project

- Run the notebook yourself and confirm every output.
- Record the actual test-set metrics produced by your run; do not copy unverified accuracy or performance claims.
- Understand the train/test split, class imbalance, scaling and evaluation metrics.
- Make and document at least one genuine improvement of your own.
- If you publish an adapted version, preserve attribution and follow the original author's licence. The ZIP supplied for this working copy did not include a licence file; check the original repository and obtain permission if needed before redistributing its code.

## Possible next improvement

Experiment with a decision threshold other than 0.50 and compare the resulting precision and recall. Choose a threshold based on a clearly explained trade-off, not on test-set tuning.
