"""
app.py - Loan Eligibility Self-Check (Streamlit)

DEPENDENCIES
    pip install streamlit pandas numpy scikit-learn xgboost joblib

HOW TO RUN
    1. Put 'loan_model.pkl' (created by train.py) in the same folder as this file.
    2. streamlit run app.py

NOTES
    * The saved bundle holds: model, preprocessor, input_columns (and metadata).
      The feature-engineering function is NOT inside the pickle, so an exact
      copy of train.py's engineer_features() lives below. Keep the two in sync.
    * Install the same scikit-learn / xgboost versions used for training,
      otherwise unpickling can fail or behave differently.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration - adjust to match the units of your training data
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(__file__).parent / "loan_model.pkl"
CURRENCY = "$"          # symbol shown in the UI (display only)
LOAN_UNIT = 1000        # training data stores LoanAmount in thousands
APPROVAL_THRESHOLD = 0.50   # probability at/above which we call it "likely approved"
STRONG_THRESHOLD = 0.70     # probability at/above which we call it "strong"
LOAN_TERMS = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]  # months


# --------------------------------------------------------------------------- #
# Feature engineering - MUST match train.py exactly
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the same derived features the model was trained on."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    term = df["Loan_Amount_Term"].replace(0, np.nan)
    df["EMI"] = (df["LoanAmount"] * 1000) / term
    income = df["TotalIncome"].replace(0, np.nan)
    df["DTI"] = df["EMI"] / income
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    return df


# --------------------------------------------------------------------------- #
# Model loading and prediction
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_bundle(path: Path):
    """Load the saved bundle once and reuse it across reruns."""
    return joblib.load(path)


def predict_probability(bundle: dict, inputs: dict) -> float:
    """Raw user inputs -> probability (0-1) that the loan is approved."""
    row = pd.DataFrame([inputs])
    row = engineer_features(row)
    # Put columns in the exact order used at training time. Any column the
    # model expects but we didn't collect becomes NaN, which the fitted
    # imputers in the preprocessor handle.
    row = row.reindex(columns=bundle["input_columns"])
    features = bundle["preprocessor"].transform(row)
    return float(bundle["model"].predict_proba(features)[0, 1])


def monthly_repayment_ratio(inputs: dict) -> float:
    """Rough repayment-to-income ratio (principal only, no interest)."""
    income = inputs["ApplicantIncome"] + inputs["CoapplicantIncome"]
    if income <= 0 or inputs["Loan_Amount_Term"] <= 0:
        return float("nan")
    return (inputs["LoanAmount"] * LOAN_UNIT / inputs["Loan_Amount_Term"]) / income


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


def readiness_gauge(prob: float, label: str = "Loan Readiness") -> None:
    """Colour-coded horizontal gauge (pure HTML/CSS, no extra dependencies)."""
    pct = prob * 100
    if prob >= STRONG_THRESHOLD:
        color = "#2e9e5b"      # green
    elif prob >= APPROVAL_THRESHOLD:
        color = "#e0a800"      # amber
    else:
        color = "#d64545"      # red
    st.markdown(
        f"""
        <div style="margin:0.25rem 0 0.75rem 0;">
          <div style="display:flex;justify-content:space-between;font-weight:600;">
            <span>{label}</span><span style="color:{color};">{pct:.0f}%</span>
          </div>
          <div style="background:#e6e6e6;border-radius:12px;height:26px;overflow:hidden;">
            <div style="width:{pct:.0f}%;background:{color};height:100%;
                        border-radius:12px;transition:width .4s;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_tips(inputs: dict, prob: float) -> list[str]:
    """Plain-English, rule-based suggestions tailored to the user's inputs."""
    tips = []
    ratio = monthly_repayment_ratio(inputs)

    if inputs["Credit_History"] == 0.0:
        tips.append(
            "**Credit history matters most.** Lenders weigh past repayments heavily. "
            "Paying bills on time, clearing overdue accounts and avoiding new "
            "defaults will steadily improve this."
        )
    elif pd.isna(inputs["Credit_History"]):
        tips.append(
            "**Build a credit track record.** With no history, lenders have little "
            "to go on. A small credit product repaid on time can help."
        )
    if pd.notna(ratio) and ratio > 0.30:
        tips.append(
            "**Your repayments look heavy compared with your income.** Consider "
            "lowering the requested amount or clearing existing debts to improve "
            "your debt-to-income ratio."
        )
    if inputs["Loan_Amount_Term"] < 180 and prob < STRONG_THRESHOLD:
        tips.append(
            "**Consider a longer term.** Spreading repayments over more months "
            "lowers the monthly burden (though total interest paid will rise)."
        )
    if inputs["CoapplicantIncome"] == 0:
        tips.append(
            "**Add a co-applicant.** A working spouse, parent or relative applying "
            "with you raises combined household income."
        )
    if not tips:
        tips.append(
            "**Small improvements add up.** Try the What-If Simulator below to see "
            "which change helps you most."
        )
    return tips


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Loan Eligibility Check", page_icon="🏦", layout="centered")

st.title("🏦 Check Your Loan Readiness")
st.markdown(
    """
    Welcome! This is a **free, risk-free, educational self-assessment tool**.
    Answer a few simple questions and get an estimate of how ready you are to
    apply for a loan, plus tips to improve your chances.

    🔒 Nothing is submitted to any lender and **this does not affect your credit
    score**. Results are only an estimate and **not a lending decision**.
    """
)

# Load the model with friendly error messages instead of a raw traceback.
try:
    bundle = load_bundle(MODEL_PATH)
    for key in ("model", "preprocessor", "input_columns"):
        if key not in bundle:
            raise KeyError(key)
except FileNotFoundError:
    st.error("The model file `loan_model.pkl` was not found. Run `train.py` first and "
             "place the file next to `app.py`.")
    st.stop()
except Exception as exc:  # corrupted file, version mismatch, missing keys...
    st.error("The model file could not be loaded. Make sure the scikit-learn and "
             "xgboost versions match the ones used for training.")
    st.caption(f"Technical detail: {type(exc).__name__}: {exc}")
    st.stop()

# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
st.header("1. Tell us about yourself")

with st.form("eligibility_form"):
    st.subheader("Income")
    c1, c2 = st.columns(2)
    with c1:
        applicant_income = st.number_input(
            f"Your Monthly Income ({CURRENCY})", min_value=0, max_value=1_000_000,
            value=5000, step=500, help="Your monthly income before tax.")
    with c2:
        coapplicant_income = st.number_input(
            f"Co-applicant's Monthly Income ({CURRENCY})", min_value=0, max_value=1_000_000,
            value=0, step=500,
            help="Spouse, parent or relative applying with you. Leave 0 if applying alone.")

    st.subheader("The loan you want")
    loan_amount = st.number_input(
        f"Requested Loan Amount ({CURRENCY})", min_value=1_000, max_value=2_000_000,
        value=150_000, step=5_000)
    loan_term = st.select_slider(
        "Loan Term (months)", options=LOAN_TERMS, value=360,
        help="How many months you'd take to repay. 360 months = 30 years.")

    st.subheader("About you")
    c3, c4 = st.columns(2)
    with c3:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        employment = st.selectbox(
            "Employment Status", ["Salaried / Employed", "Self-Employed"])
    with c4:
        dependents = st.selectbox(
            "Number of Dependents", ["0", "1", "2", "3+"],
            help="People who rely on your income (children, parents, etc.).")
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"],
            help="Where the property you want to buy or use as security is located.")

    credit_choice = st.radio(
        "Credit History",
        ["Good: I repay on time and meet lender guidelines",
         "Poor: I have missed payments or defaults",
         "Not sure / no credit history yet"],
        help="Your track record of repaying past loans and credit cards.")

    # If the model was trained WITH gender / marital status, ask for them too.
    extra = {}
    if "Gender" in bundle["input_columns"]:
        extra["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    if "Married" in bundle["input_columns"]:
        extra["Married"] = st.selectbox("Married", ["Yes", "No"])

    submitted = st.form_submit_button("Check My Eligibility", type="primary")

# Convert friendly form values into the raw values the model was trained on.
if submitted:
    if applicant_income + coapplicant_income <= 0:
        st.error("Please enter an income greater than zero to continue.")
    else:
        credit_value = {"Good": 1.0, "Poor": 0.0}.get(credit_choice.split(":")[0], np.nan)
        inputs = {
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": loan_amount / LOAN_UNIT,       # model expects thousands
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": credit_value,
            "Dependents": dependents.replace("+", ""),   # "3+" -> "3" as in training
            "Education": education,
            "Self_Employed": "Yes" if employment == "Self-Employed" else "No",
            "Property_Area": property_area,
            **extra,
        }
        # Persist across reruns so the What-If sliders don't wipe the results.
        st.session_state["inputs"] = inputs
        st.session_state["prob"] = predict_probability(bundle, inputs)

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if "inputs" in st.session_state:
    inputs = st.session_state["inputs"]
    prob = st.session_state["prob"]

    st.header("2. Your Loan Readiness")
    readiness_gauge(prob)

    ratio = monthly_repayment_ratio(inputs)
    m1, m2, m3 = st.columns(3)
    m1.metric("Readiness score", f"{prob * 100:.0f}%")
    m2.metric("Rough monthly repayment",
              money(inputs["LoanAmount"] * LOAN_UNIT / inputs["Loan_Amount_Term"]),
              help="Loan amount divided by months. Excludes interest, so real "
                   "repayments will be higher.")
    m3.metric("Repayment vs. income",
              f"{ratio * 100:.0f}%" if pd.notna(ratio) else "n/a",
              help="Share of your household income going to repayments "
                   "(before interest). Lower is better.")

    if prob >= STRONG_THRESHOLD:
        st.success("🎉 **You are in a strong position to approach a lender!**")
        st.markdown(
            "**Suggested next steps**\n"
            "- Gather proof of income (payslips, bank statements, tax returns).\n"
            "- Collect ID and address documents.\n"
            "- Compare interest rates and fees from several lenders.\n"
            "- Check your credit report for errors before applying."
        )
    elif prob >= APPROVAL_THRESHOLD:
        st.info("👍 **You're on the right track,** but your application is borderline. "
                "A few improvements could make it much stronger.")
        for tip in build_tips(inputs, prob):
            st.markdown(f"- {tip}")
    else:
        st.warning("🌱 **Your approval odds look low right now,** but there are "
                   "clear steps that can improve them.")
        for tip in build_tips(inputs, prob):
            st.markdown(f"- {tip}")

    # ----------------------------------------------------------------------- #
    # What-If Simulator
    # ----------------------------------------------------------------------- #
    st.header("3. What-If Simulator")
    st.caption("Move the sliders to see how changes would affect your score instantly.")

    s1, s2 = st.columns(2)
    with s1:
        reduction = st.slider("Lower the loan amount by", 0, 60, 0, step=5, format="%d%%")
        extra_income = st.slider(
            f"Add monthly household income ({CURRENCY})", 0, 5000, 0, step=100,
            help="For example a co-applicant, a raise or a side income.")
    with s2:
        # Only default to your current term if it's one of the selectable options.
        what_if_term = st.select_slider(
            "Loan term (months)", options=LOAN_TERMS,
            value=int(inputs["Loan_Amount_Term"])
            if int(inputs["Loan_Amount_Term"]) in LOAN_TERMS else 360,
            key="whatif_term")
        improve_credit = False
        if inputs["Credit_History"] != 1.0:
            improve_credit = st.toggle("Imagine I had a good credit history", value=False)

    # Copy the user's inputs and apply the hypothetical changes.
    scenario = dict(inputs)
    scenario["LoanAmount"] = inputs["LoanAmount"] * (1 - reduction / 100)
    scenario["CoapplicantIncome"] = inputs["CoapplicantIncome"] + extra_income
    scenario["Loan_Amount_Term"] = float(what_if_term)
    if improve_credit:
        scenario["Credit_History"] = 1.0

    new_prob = predict_probability(bundle, scenario)
    delta = (new_prob - prob) * 100

    readiness_gauge(new_prob, label="Readiness with your changes")
    w1, w2 = st.columns(2)
    w1.metric("Original score", f"{prob * 100:.0f}%")
    w2.metric("New score", f"{new_prob * 100:.0f}%", delta=f"{delta:+.0f} pts")
    st.caption(
        f"Scenario loan amount: **{money(scenario['LoanAmount'] * LOAN_UNIT)}** "
        f"over **{what_if_term} months**.")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "⚠️ This tool is for education only. It is an estimate from a statistical "
    "model, not financial advice and not a guarantee of approval. Real lenders "
    "use their own criteria and will check your documents and credit report."
)
