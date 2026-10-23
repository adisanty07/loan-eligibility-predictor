# 🏦 Loan Readiness & Eligibility Predictor

An interactive, educational machine learning web application built with **Streamlit** and **XGBoost** that helps users self-assess their loan application readiness, calculate debt-to-income (DTI) metrics, and explore interactive "What-If" financial scenarios.

> ⚠️ **Disclaimer:** This tool is strictly for educational and self-assessment purposes. It is an estimate from a statistical model and does not constitute financial advice or an official lending decision.

---

## 🚀 Key Features
- **Instant Readiness Score:** Calculates a probability score and gauges your eligibility likelihood with a custom visual gauge.
- **Smart Financial Metrics:** Automatically computes estimated monthly installments (EMI) and Debt-to-Income (DTI) ratios.
- **What-If Simulator:** Interactive sliders allowing users to test how adjusting loan amounts, household income, or terms impacts their score in real time.
- **Actionable Feedback:** Generates tailored suggestions to improve loan approval odds.
- **Privacy First:** Runs locally without submitting sensitive personal data to any external lenders.

---

## 🛠️ Tech Stack
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/), [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)

---

## 📂 Project Structure
```text
LoanPredictor/
├── app.py              # Main Streamlit web application interface
├── train.py            # Model training script with feature engineering & SMOTE
├── loan_data.csv       # Dataset used for training classifiers
├── loan_model.pkl      # Serialized machine learning pipeline model artifact
└── requirements.txt    # Python package dependencies