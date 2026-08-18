from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
# K-NEAREST NEIGHBORS
# =========================================================

print("\n===== K-NEAREST NEIGHBORS CLASSIFIER =====")

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)

knn_model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTIONS
# =========================================================

y_pred_knn = knn_model.predict(X_test)

y_prob_knn = knn_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# EVALUATION METRICS
# =========================================================

knn_metrics = {
    "ML Model Name": "K-Nearest Neighbors",
    "Accuracy": accuracy_score(
        y_test,
        y_pred_knn
    ),
    "AUC": roc_auc_score(
        y_test,
        y_prob_knn
    ),
    "Precision": precision_score(
        y_test,
        y_pred_knn,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        y_pred_knn,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        y_pred_knn,
        zero_division=0
    ),
    "MCC": matthews_corrcoef(
        y_test,
        y_pred_knn
    )
}

knn_results = pd.DataFrame(
    [knn_metrics]
)

print("\n===== KNN METRICS =====")

print(
    knn_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# =========================================================
# SAVE TRAINED MODEL
# =========================================================

knn_model_path = MODEL_DIR / "knn.pkl"

joblib.dump(
    knn_model,
    knn_model_path
)

print(
    f"\nKNN model saved to: "
    f"{knn_model_path}"
)

print("\nStep 6 completed successfully.")
