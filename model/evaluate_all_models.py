from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
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

OUTPUT_FILE = BASE_DIR / "model_comparison.csv"

RANDOM_STATE = 5083
TEST_SIZE = 0.20


# =========================================================
# LOAD FEATURE NAMES
# =========================================================

def load_feature_names(names_file):
    feature_names = []

    with open(
        names_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:
            line = line.strip()

            if not line or line.startswith("|"):
                continue

            if ":" in line:
                feature_name = line.split(
                    ":",
                    1
                )[0].strip()

                feature_names.append(
                    feature_name
                )

    return feature_names


feature_names = load_feature_names(
    NAMES_FILE
)

if len(feature_names) != 57:
    raise ValueError(
        f"Expected 57 features, "
        f"but found {len(feature_names)}."
    )


# =========================================================
# LOAD DATASET
# =========================================================

column_names = (
    feature_names
    + ["spam"]
)

dataset = pd.read_csv(
    DATA_FILE,
    header=None,
    names=column_names
)

X = dataset.drop(
    columns=["spam"]
)

y = dataset["spam"].astype(int)


# =========================================================
# RECREATE EXACT SAME TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    "\n===== FINAL TEST DATA VALIDATION ====="
)

print(
    f"Test observations : {len(X_test)}"
)

print(
    f"Input features    : {X_test.shape[1]}"
)

print(
    "\nTest target distribution:"
)

print(
    y_test.value_counts().sort_index()
)


# =========================================================
# SAVED MODEL FILES
# =========================================================

model_files = {
    "Logistic Regression":
        MODEL_DIR / "logistic_regression.pkl",

    "Decision Tree":
        MODEL_DIR / "decision_tree.pkl",

    "K-Nearest Neighbors":
        MODEL_DIR / "knn.pkl",

    "Naive Bayes":
        MODEL_DIR / "naive_bayes.pkl",

    "Random Forest":
        MODEL_DIR / "random_forest.pkl"
}


# =========================================================
# VERIFY ALL REQUIRED FILES EXIST
# =========================================================

for model_name, model_path in model_files.items():

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing saved model for "
            f"{model_name}: {model_path}"
        )


# =========================================================
# EVALUATE ALL MODELS
# =========================================================

results = []

for model_name, model_path in model_files.items():

    print(
        f"\nEvaluating: {model_name}"
    )

    model = joblib.load(
        model_path
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    model_metrics = {
        "ML Model Name":
            model_name,

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "AUC":
            roc_auc_score(
                y_test,
                y_prob
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_test,
                y_pred
            )
    }

    results.append(
        model_metrics
    )


# =========================================================
# FINAL COMPARISON TABLE
# =========================================================

comparison_df = pd.DataFrame(
    results
)

print(
    "\n===== FINAL MODEL COMPARISON ====="
)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}"
    )
)


# =========================================================
# SAVE RESULTS TO CSV
# =========================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nComparison table saved to: "
    f"{OUTPUT_FILE}"
)


# =========================================================
# IDENTIFY BEST MODELS
# =========================================================

print(
    "\n===== BEST MODEL BY METRIC ====="
)

for metric in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]:
    best_index = (
        comparison_df[metric].idxmax()
    )

    best_model = comparison_df.loc[
        best_index,
        "ML Model Name"
    ]

    best_value = comparison_df.loc[
        best_index,
        metric
    ]

    print(
        f"{metric:<10}: "
        f"{best_model} "
        f"({best_value:.4f})"
    )


print(
    "\nStep 9 completed successfully."
)
