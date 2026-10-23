# 🏦 Loan Readiness & Eligibility Predictor

An interactive machine learning web application that helps users explore loan application readiness through predicted approval probabilities, financial metrics, and real-time What-If simulations.

Built with **Python, Streamlit, Scikit-learn, and XGBoost**.

> ⚠️ **Disclaimer:** This project is for educational and self-assessment purposes only. Predictions are estimates from a machine learning model, not financial advice or an official lending decision.

## ✨ Features

- **Loan Readiness Score:** View the model's estimated loan approval probability.
- **Financial Metrics:** Calculate approximate monthly repayments and a repayment-to-income ratio.
- **What-If Simulator:** Experiment with loan amount, household income, repayment term, and credit history.
- **Personalized Suggestions:** Get general suggestions based on the entered financial details.
- **Machine Learning Pipeline:** Compare Random Forest and XGBoost classifiers.
- **Data Preprocessing:** Handle missing values and encode categorical features.
- **Class Imbalance Handling:** Apply SMOTE to the training data when the specified imbalance condition is met.
- **Local Execution:** Run the application on your own computer.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| Pandas & NumPy | Data processing and feature engineering |
| Scikit-learn | Preprocessing, model training and evaluation |
| XGBoost | Gradient-boosted classification |
| Imbalanced-learn | SMOTE oversampling |
| Joblib | Saving and loading the trained model |

## 📁 Project Structure

```text
loan-eligibility-predictor/
├── app.py                  # Streamlit application
├── train.py                # Model training and evaluation
├── loan_data.csv           # Training dataset
├── loan_model.pkl          # Saved model bundle
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

Follow these steps to run the project locally.

### Prerequisites

- Python 3.10 or newer recommended
- Git
- Visual Studio Code (optional)
- An internet connection for the initial package installation

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/adisanty07/loan-eligibility-predictor.git
cd loan-eligibility-predictor
```

### Step 2: Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

If you use PowerShell and activation is blocked, use Command Prompt or follow Python's virtual environment activation guidance.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

After renaming `requirements.txt.txt` to `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run the application

The repository includes a saved model file, `loan_model.pkl`. If it loads successfully with the installed library versions, start the application using:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your browser.

### Step 5: Use the application

1. Enter applicant and co-applicant income.
2. Select the requested loan amount and repayment term.
3. Provide education, employment, dependents, property area, and credit-history information.
4. Click **Check My Eligibility**.
5. Review the readiness score and estimated repayment metrics.
6. Experiment with the What-If Simulator to compare hypothetical scenarios.

## 🧠 Machine Learning Workflow

The training pipeline in `train.py` follows these steps:

1. Load and clean the loan dataset.
2. Remove the loan identifier and rows without a valid target.
3. Engineer features such as total income, approximate EMI, DTI, and log-transformed income and loan amount.
4. Split the data into stratified training and testing sets.
5. Fit the preprocessing pipeline using training data only.
6. Apply SMOTE to the training split when the configured imbalance threshold is met.
7. Train and evaluate Random Forest and XGBoost classifiers.
8. Compare ROC-AUC scores and save the best model, fitted preprocessor, and required feature metadata to `loan_model.pkl`.

### Retrain the model (optional)

If you want to train a fresh model using the dataset:

```bash
python train.py
```

This regenerates `loan_model.pkl`. The training script expects `loan_data.csv` in the project root.

Keep the preprocessing logic and dependency versions compatible with the application when retraining or loading saved model artifacts.

## 🧮 Key Concepts

- **Classification:** Estimates the likelihood of a historical loan-approval outcome.
- **Feature Engineering:** Creates additional financial indicators from the original inputs.
- **Data Leakage Prevention:** Performs the train/test split before fitting preprocessing steps and applying oversampling.
- **Model Evaluation:** Uses classification metrics and ROC-AUC to compare candidate models.
- **Interactive Simulation:** Recalculates model predictions for hypothetical changes in applicant information.

## 🐛 Troubleshooting

**1. `requirements.txt` not found**

Check that the file has been renamed from `requirements.txt.txt` to `requirements.txt`.

**2. `loan_model.pkl` not found**

Confirm that the model file exists in the same directory as `app.py`. If it is missing, verify the dataset and dependencies, then run:

```bash
python train.py
```

**3. Model loading or pickle errors**

The installed versions of Scikit-learn, XGBoost, and other relevant libraries may differ from those used to create the saved model. Install compatible versions and retrain the model if necessary.

**4. `streamlit` command not found**

Activate the virtual environment and install the requirements again. You can also try:

```bash
python -m streamlit run app.py
```

**5. Dataset not found during training**

Make sure `loan_data.csv` is present in the repository root, alongside `train.py`.

## 🔒 Privacy & Responsible Use

- Do not enter real banking credentials, account numbers, or other unnecessary sensitive information.
- Treat predictions as estimates, not guarantees of loan approval.
- Actual lending decisions depend on lender-specific criteria and additional checks.
- The model may inherit limitations or biases from its training dataset.

## 🔮 Future Improvements

- Add explainability for individual predictions.
- Include more comprehensive model evaluation and fairness analysis.
- Add automated tests and dependency version pinning.
- Deploy a public demo with appropriate privacy safeguards.
- Improve currency labels and financial calculations for the intended user population.

## 👨‍💻 Author

**Aditya Rai**

GitHub: [@adisanty07](https://github.com/adisanty07)

Repository: [Loan Readiness & Eligibility Predictor](https://github.com/adisanty07/loan-eligibility-predictor)

---

⭐ If you find this project useful, consider giving the repository a star.

**Note:** This project is an educational machine learning demonstration and is not a substitute for professional financial guidance.
