"""
train.py - Train and compare loan-approval classifiers.

DEPENDENCIES
    pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib

HOW TO RUN
    1. Put 'loan_data.csv' next to this script. Expected columns (the classic
       "Loan Prediction" dataset):
         Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
         ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
         Credit_History, Property_Area, Loan_Status (Y/N)
    2. python train.py
    3. Output: 'loan_model.pkl' (a joblib bundle, see load example at bottom).

DESIGN NOTES (why it is written this way)
    * The train/test split happens BEFORE imputation, encoding and SMOTE, and
      every preprocessing step is fitted on the training data only. This
      prevents data leakage, which would otherwise inflate your test metrics.
    * SMOTE is applied to the training split only. The test set keeps its
      natural class distribution so the metrics reflect real-world use.
    * Tree-based models don't need feature scaling, so no scaler is used.
"""

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_PATH = "loan_data.csv"
MODEL_PATH = "loan_model.pkl"
TARGET = "Loan_Status"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Apply SMOTE when the minority class is below this share of the training data.
IMBALANCE_THRESHOLD = 0.40

# Gender/Marital status are protected attributes in lending in many
# jurisdictions. Dropping them is the safer default for a consumer-facing app.
# Set to False only if you have a legal/compliance basis for using them.
DROP_SENSITIVE = True
SENSITIVE_COLUMNS = ["Gender", "Married"]


# --------------------------------------------------------------------------- #
# 1. Data preparation helpers
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features. IMPORTANT: your Streamlit app must apply this
    exact same function to user input before calling the model (copy it into
    your app, or move it into a shared module).

    Assumes LoanAmount is in thousands (as in the classic dataset) and
    Loan_Amount_Term is in months.
    """
    df = df.copy()

    # Total household income (applicant + co-applicant).
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Approximate monthly instalment (ignores interest - a simple proxy).
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000) / term

    # Debt-to-income style ratio: monthly instalment relative to total income.
    income = df["TotalIncome"].replace(0, np.nan)
    df["DTI"] = df["EMI"] / income

    # Log transforms tame the heavy right skew in income / loan amount.
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    return df


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the CSV and perform light, leakage-safe cleaning."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: '{path}' not found. Place your dataset next to train.py.")

    # Drop identifier column - it carries no predictive signal.
    df = df.drop(columns=[c for c in ["Loan_ID"] if c in df.columns])

    # Drop rows with no target; they cannot be used for supervised learning.
    df = df.dropna(subset=[TARGET])

    # Normalise Dependents ("3+" -> "3") and keep it categorical-friendly.
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
        df["Dependents"] = df["Dependents"].replace("nan", np.nan)

    # Encode target: Y -> 1 (approved), N -> 0 (rejected).
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Credit_History is binary (0/1); treat it as categorical so the mode is
    # used for imputation and it is one-hot encoded.
    if "Credit_History" in df.columns:
        df["Credit_History"] = df["Credit_History"].astype("float")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Numeric columns     -> median imputation
    Categorical columns -> most-frequent (mode) imputation + one-hot encoding
    """
    categorical_cols = [
        c for c in ["Gender", "Married", "Dependents", "Education",
                    "Self_Employed", "Property_Area", "Credit_History"]
        if c in X.columns
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" keeps the web app from crashing on new values.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


# --------------------------------------------------------------------------- #
# 2. Evaluation helper
# --------------------------------------------------------------------------- #
def evaluate(name: str, model, X_test, y_test) -> float:
    """Print a full report and return ROC-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    # Report includes Precision, Recall and F1 for each class.
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))
    print("Confusion matrix [rows=true, cols=pred]:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    return auc


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    # ---- Load, clean, engineer features --------------------------------- #
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    if DROP_SENSITIVE:
        df = df.drop(columns=[c for c in SENSITIVE_COLUMNS if c in df.columns])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
    print("Missing values per column (before imputation):")
    print(X.isna().sum()[X.isna().sum() > 0].to_string() or "  none")

    # ---- Split FIRST (stratified keeps class ratio in both splits) ------ #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # ---- Fit preprocessing on TRAIN only, apply to both ----------------- #
    preprocessor = build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # ---- Class imbalance check + SMOTE (training split only) ------------ #
    class_share = y_train.value_counts(normalize=True)
    print("\nTraining class distribution:")
    print(class_share.rename({0: "Rejected (0)", 1: "Approved (1)"}).round(3).to_string())

    if class_share.min() < IMBALANCE_THRESHOLD:
        print(f"Minority share < {IMBALANCE_THRESHOLD:.0%} -> applying SMOTE to training data.")
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)
        print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
    else:
        print("Classes reasonably balanced -> skipping SMOTE.")
        X_train_res, y_train_res = X_train_p, y_train

    # ---- Define candidate models ---------------------------------------- #
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,            # limit depth to curb overfitting on small data
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # ---- Train, evaluate, compare --------------------------------------- #
    results = {}
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        results[name] = evaluate(name, model, X_test_p, y_test)

    print("\nROC-AUC summary:")
    for name, auc in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<15} {auc:.4f}")

    # ---- Select best model by ROC-AUC and export ------------------------ #
    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]:.4f})")

    # Everything the web app needs, in a single file.
    bundle = {
        "model": models[best_name],
        "preprocessor": preprocessor,            # fitted imputers + encoder
        "model_name": best_name,
        "roc_auc": results[best_name],
        # Columns (after feature engineering) the preprocessor expects:
        "input_columns": list(X.columns),
        "dropped_sensitive": SENSITIVE_COLUMNS if DROP_SENSITIVE else [],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved artifacts to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# HOW TO USE THE SAVED MODEL IN STREAMLIT (app.py)
# --------------------------------------------------------------------------- #
# import joblib, pandas as pd
# from train import engineer_features       # or copy the function into your app
#
# bundle = joblib.load("loan_model.pkl")
#
# # Build a one-row DataFrame from the form inputs with the RAW columns
# # (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
# #  Credit_History, Dependents, Education, Self_Employed, Property_Area):
# row = pd.DataFrame([{...}])
# row = engineer_features(row)
# row = row[bundle["input_columns"]]        # same column order as training
#
# X = bundle["preprocessor"].transform(row)
# prob = bundle["model"].predict_proba(X)[0, 1]
# st.write("Approved" if prob >= 0.5 else "Rejected", f"({prob:.0%})")
