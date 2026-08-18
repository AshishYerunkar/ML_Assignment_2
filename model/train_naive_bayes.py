from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

DATA_FILE = BASE_DIR / "spambase.data"
NAMES_FILE = BASE_DIR / "spambase.names"

RANDOM_STATE = 5083
TEST_SIZE = 0.20


# =========================================================
# LOAD FEATURE NAMES
# =========================================================

def load_feature_names(names_file):
    feature_names = []

    with open(names_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("|"):
                continue

            if ":" in line:
                feature_name = line.split(":", 1)[0].strip()
                feature_names.append(feature_name)

    return feature_names


feature_names = load_feature_names(NAMES_FILE)

if len(feature_names) != 57:
    raise ValueError(
        f"Expected 57 input features, but found {len(feature_names)}."
    )


# =========================================================
# LOAD DATASET
# =========================================================

column_names = feature_names + ["spam"]

dataset = pd.read_csv(
    DATA_FILE,
    header=None,
    names=column_names
)


# =========================================================
# DATASET VALIDATION
# =========================================================

print("\n===== DATASET VALIDATION =====")
print(f"Dataset shape           : {dataset.shape}")
print(f"Number of instances     : {dataset.shape[0]}")
print(f"Number of input features: {dataset.shape[1] - 1}")
print(f"Missing values          : {dataset.isnull().sum().sum()}")
print(f"Duplicate column names  : {dataset.columns.duplicated().sum()}")


# =========================================================
# PREDICTORS AND TARGET
# =========================================================

X = dataset.drop(columns=["spam"])
y = dataset["spam"].astype(int)


# =========================================================
# SAME STRATIFIED TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n===== TRAIN / TEST SPLIT =====")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape : {y_test.shape}")

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())


# =========================================================
# GAUSSIAN NAIVE BAYES
# =========================================================

print("\n===== GAUSSIAN NAIVE BAYES CLASSIFIER =====")

naive_bayes_model = GaussianNB()

naive_bayes_model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred_nb = naive_bayes_model.predict(X_test)

y_prob_nb = naive_bayes_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# EVALUATION METRICS
# =========================================================

naive_bayes_metrics = {
    "ML Model Name": "Naive Bayes",
    "Accuracy": accuracy_score(
        y_test,
        y_pred_nb
    ),
    "AUC": roc_auc_score(
        y_test,
        y_prob_nb
    ),
    "Precision": precision_score(
        y_test,
        y_pred_nb,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        y_pred_nb,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        y_pred_nb,
        zero_division=0
    ),
    "MCC": matthews_corrcoef(
        y_test,
        y_pred_nb
    )
}

naive_bayes_results = pd.DataFrame(
    [naive_bayes_metrics]
)

print("\n===== NAIVE BAYES METRICS =====")

print(
    naive_bayes_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# =========================================================
# SAVE TRAINED MODEL
# =========================================================

naive_bayes_model_path = (
    MODEL_DIR / "naive_bayes.pkl"
)

joblib.dump(
    naive_bayes_model,
    naive_bayes_model_path
)

print(
    f"\nNaive Bayes model saved to: "
    f"{naive_bayes_model_path}"
)

print("\nStep 7 completed successfully.")
