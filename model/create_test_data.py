from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "spambase.data"
NAMES_FILE = BASE_DIR / "spambase.names"
OUTPUT_FILE = BASE_DIR / "test_data.csv"

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
# LOAD ORIGINAL DATASET
# =========================================================

column_names = feature_names + ["spam"]

dataset = pd.read_csv(
    DATA_FILE,
    header=None,
    names=column_names
)


# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

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


# =========================================================
# CREATE TEST DATASET
# =========================================================

test_data = X_test.copy()

test_data["spam"] = y_test


# =========================================================
# VALIDATE TEST DATA
# =========================================================

print("\n===== TEST DATA VALIDATION =====")

print(
    f"Test data shape       : {test_data.shape}"
)

print(
    f"Input features        : {test_data.shape[1] - 1}"
)

print(
    f"Target column         : {test_data.columns[-1]}"
)

print(
    f"Missing values        : "
    f"{test_data.isnull().sum().sum()}"
)

print(
    f"Duplicate column names: "
    f"{test_data.columns.duplicated().sum()}"
)

print(
    "\nTarget distribution:"
)

print(
    test_data["spam"]
    .value_counts()
    .sort_index()
)


# =========================================================
# SAVE CSV
# =========================================================

test_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nTest data saved to: {OUTPUT_FILE}"
)

print(
    "\nStep 10 completed successfully."
)
