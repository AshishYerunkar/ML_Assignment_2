from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REFERENCE_TEST_FILE = BASE_DIR / "test_data.csv"

TARGET_COLUMN = "spam"

MODEL_FILES = {
    "Logistic Regression":
        MODEL_DIR / "logistic_regression.pkl",

    "Decision Tree":
        MODEL_DIR / "decision_tree.pkl",

    "K-Nearest Neighbors":
        MODEL_DIR / "knn.pkl",

    "Naive Bayes":
        MODEL_DIR / "naive_bayes.pkl",

    "Random Forest":
        MODEL_DIR / "random_forest.pkl",
}


st.set_page_config(
    page_title="Spam Classification Model Evaluator",
    page_icon="📧",
    layout="wide",
)


# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_saved_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_expected_features(reference_file):
    columns = pd.read_csv(
        reference_file,
        nrows=0
    ).columns.tolist()

    if TARGET_COLUMN not in columns:
        raise ValueError(
            f"Reference file does not contain "
            f"the target column '{TARGET_COLUMN}'."
        )

    return [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_uploaded_data(
    uploaded_data,
    expected_features
):
    errors = []

    if uploaded_data.empty:
        errors.append(
            "The uploaded CSV contains no rows."
        )
        return None, None, errors

    if uploaded_data.columns.duplicated().any():
        errors.append(
            "The uploaded CSV contains duplicate column names."
        )

    if TARGET_COLUMN not in uploaded_data.columns:
        errors.append(
            f"The uploaded CSV must contain "
            f"the target column '{TARGET_COLUMN}'."
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in uploaded_data.columns
    ]

    if missing_features:
        errors.append(
            "Missing required feature columns: "
            + ", ".join(missing_features)
        )

    allowed_columns = (
        expected_features
        + [TARGET_COLUMN]
    )

    unexpected_columns = [
        column
        for column in uploaded_data.columns
        if column not in allowed_columns
    ]

    if unexpected_columns:
        errors.append(
            "Unexpected columns found: "
            + ", ".join(unexpected_columns)
        )

    if errors:
        return None, None, errors

    if uploaded_data.isnull().sum().sum() > 0:
        errors.append(
            "The uploaded CSV contains missing values."
        )
        return None, None, errors

    try:
        X = uploaded_data[
            expected_features
        ].apply(
            pd.to_numeric,
            errors="raise"
        )

        y = pd.to_numeric(
            uploaded_data[TARGET_COLUMN],
            errors="raise"
        )

    except Exception:
        errors.append(
            "All predictor and target values "
            "must be numeric."
        )
        return None, None, errors

    unique_targets = set(
        y.unique()
    )

    if not unique_targets.issubset({0, 1}):
        errors.append(
            "The target column must contain "
            "only 0 and 1."
        )
        return None, None, errors

    return X, y.astype(int), errors


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(
    model,
    X,
    y_true
):
    y_pred = model.predict(X)

    y_probability = model.predict_proba(
        X
    )[:, 1]

    if y_true.nunique() == 2:
        auc_value = roc_auc_score(
            y_true,
            y_probability
        )
    else:
        auc_value = float("nan")

    metrics = {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "AUC":
            auc_value,

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                y_pred
            ),
    }

    return metrics, y_pred


def format_metric(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.4f}"


# =========================================================
# CHECK REQUIRED PROJECT FILES
# =========================================================

if not REFERENCE_TEST_FILE.exists():
    st.error(
        "Reference test_data.csv could not be found."
    )
    st.stop()


missing_model_files = [
    model_name
    for model_name, model_path
    in MODEL_FILES.items()
    if not model_path.exists()
]

if missing_model_files:
    st.error(
        "Missing saved model files for: "
        + ", ".join(missing_model_files)
    )
    st.stop()


expected_features = load_expected_features(
    REFERENCE_TEST_FILE
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title(
    "📧 Spam Classification Model Evaluator"
)

st.write(
    "Evaluate multiple machine-learning classifiers "
    "for spam detection using labelled test data."
)

st.caption(
    "Target convention: 0 = Non-spam, 1 = Spam"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "Evaluation Controls"
    )

    selected_model_name = st.selectbox(
        "Select classification model",
        options=list(
            MODEL_FILES.keys()
        )
    )

    st.markdown(
        """
        **Test file requirements**

        Upload a CSV containing:

        - 57 Spambase predictor columns
        - `spam` target column
        - target values 0 or 1
        - no missing values
        """
    )


# =========================================================
# CSV UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload labelled test CSV",
    type=["csv"],
    help=(
        "Use test_data.csv supplied with "
        "this project for the standard evaluation."
    )
)


if uploaded_file is None:

    st.info(
        "Upload test_data.csv to begin model evaluation."
    )

    st.stop()


# =========================================================
# READ UPLOADED FILE
# =========================================================

try:
    uploaded_data = pd.read_csv(
        uploaded_file
    )

except Exception as error:
    st.error(
        f"Unable to read the CSV file: {error}"
    )
    st.stop()


st.subheader(
    "Uploaded Test Data"
)

summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)

summary_col1.metric(
    "Rows",
    uploaded_data.shape[0]
)

summary_col2.metric(
    "Columns",
    uploaded_data.shape[1]
)

summary_col3.metric(
    "Expected Predictors",
    len(expected_features)
)


with st.expander(
    "Preview uploaded data"
):

    st.dataframe(
        uploaded_data.head(10),
        width="stretch"
    )


# =========================================================
# VALIDATE UPLOADED DATA
# =========================================================

X_test, y_test, validation_errors = (
    validate_uploaded_data(
        uploaded_data,
        expected_features
    )
)


if validation_errors:

    for validation_error in validation_errors:
        st.error(
            validation_error
        )

    st.stop()


st.success(
    "CSV validation passed successfully."
)


# =========================================================
# SELECTED MODEL EVALUATION
# =========================================================

selected_model_path = MODEL_FILES[
    selected_model_name
]

selected_model = load_saved_model(
    selected_model_path
)

selected_metrics, selected_predictions = (
    evaluate_model(
        selected_model,
        X_test,
        y_test
    )
)


st.divider()

st.header(
    f"Selected Model: {selected_model_name}"
)

st.subheader(
    "Evaluation Metrics"
)


metric_row_1 = st.columns(3)

metric_row_1[0].metric(
    "Accuracy",
    format_metric(
        selected_metrics["Accuracy"]
    )
)

metric_row_1[1].metric(
    "AUC",
    format_metric(
        selected_metrics["AUC"]
    )
)

metric_row_1[2].metric(
    "Precision",
    format_metric(
        selected_metrics["Precision"]
    )
)


metric_row_2 = st.columns(3)

metric_row_2[0].metric(
    "Recall",
    format_metric(
        selected_metrics["Recall"]
    )
)

metric_row_2[1].metric(
    "F1 Score",
    format_metric(
        selected_metrics["F1"]
    )
)

metric_row_2[2].metric(
    "MCC",
    format_metric(
        selected_metrics["MCC"]
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader(
    "Confusion Matrix"
)

confusion_values = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
)

figure, axis = plt.subplots(
    figsize=(6, 5)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_values,
    display_labels=[
        "Non-spam (0)",
        "Spam (1)"
    ]
)

display.plot(
    ax=axis,
    values_format="d"
)

axis.set_title(
    f"{selected_model_name} Confusion Matrix"
)

st.pyplot(
    figure
)

plt.close(
    figure
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=[
        "Non-spam",
        "Spam"
    ],
    output_dict=True,
    zero_division=0
)

report_dataframe = (
    pd.DataFrame(report)
    .transpose()
    .round(4)
)

st.dataframe(
    report_dataframe,
    width="stretch"
)


# =========================================================
# COMPARE ALL FIVE MODELS
# =========================================================

st.divider()

st.header(
    "Comparison of All Models"
)

comparison_results = []

for model_name, model_path in MODEL_FILES.items():

    current_model = load_saved_model(
        model_path
    )

    current_metrics, _ = evaluate_model(
        current_model,
        X_test,
        y_test
    )

    comparison_results.append(
        {
            "ML Model Name":
                model_name,

            "Accuracy":
                current_metrics["Accuracy"],

            "AUC":
                current_metrics["AUC"],

            "Precision":
                current_metrics["Precision"],

            "Recall":
                current_metrics["Recall"],

            "F1":
                current_metrics["F1"],

            "MCC":
                current_metrics["MCC"],
        }
    )


comparison_dataframe = pd.DataFrame(
    comparison_results
)

display_comparison = (
    comparison_dataframe.copy()
)

numeric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]

display_comparison[numeric_columns] = (
    display_comparison[
        numeric_columns
    ].round(4)
)

st.dataframe(
    display_comparison,
    width="stretch",
    hide_index=True
)


best_mcc_index = (
    comparison_dataframe["MCC"]
    .idxmax()
)

best_overall_model = (
    comparison_dataframe.loc[
        best_mcc_index,
        "ML Model Name"
    ]
)

best_mcc_value = (
    comparison_dataframe.loc[
        best_mcc_index,
        "MCC"
    ]
)


st.success(
    f"Highest MCC on the uploaded test data: "
    f"{best_overall_model} "
    f"({best_mcc_value:.4f})"
)


st.caption(
    "All displayed metrics are calculated "
    "from the CSV uploaded in the current session."
)
