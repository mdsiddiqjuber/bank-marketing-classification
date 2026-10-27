# Bank Marketing: Classification Model Comparison

A Fundamentals of Data Science case study that compares Logistic Regression, Decision Tree, K-Nearest Neighbours (KNN), and Random Forest to predict whether a client subscribes to a term deposit.

## Included files

- `bank_marketing_classification.ipynb` — executed notebook with code, results, tables, and visualizations.
- `bank-full.csv` — supplied UCI Bank Marketing full dataset, semicolon-delimited.
- `Bank_Marketing_Classification_Case_Study.pdf` — formatted six-page report for submission.
- `Bank_Marketing_Classification_Case_Study.docx` — editable report source.
- `analyze.py` — standalone script to rerun EDA, train/evaluate the four models, save plots/tables, and serialize the selected model.
- `app.py` — Streamlit form-based demo for customer-level predictions.
- `models/best_bank_marketing_model.joblib` — fitted selected pipeline.
- `figures/` — required EDA and model evaluation figures, including target distribution, the metric comparison, four confusion matrices, and ROC curves.
- `outputs/` — dataset audits, model comparison table, classification reports, and model-selection notes.

## Important modeling decision

The `duration` feature is excluded from the primary prediction models because the call length is unknown before the call happens. This follows the supplied assignment's explicit instruction and avoids an unrealistic pre-contact prediction scenario.

## Setup (Windows PowerShell)

From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the analysis

```powershell
python analyze.py
```

This creates/replaces the figures and outputs and saves `models/best_bank_marketing_model.joblib`.

## Run the notebook

Open `bank_marketing_classification.ipynb` in VS Code, select the same Python 3 kernel where requirements were installed, and use **Run All**. The notebook includes actual executed outputs in the supplied version.

## Run the Streamlit app

First run `python analyze.py`, then:

```powershell
streamlit run app.py
```

## Dataset and target

The full dataset has 45,211 records and 17 columns including the target `y`. `y=yes` means the customer subscribed; `y=no` means the customer did not. There are 39,922 `no` cases (88.30%) and 5,289 `yes` cases (11.70%). The data file uses a semicolon (`;`) as delimiter.

## Current test-set comparison

Stratified 80:20 split (`random_state=42`), 36,168 train rows and 9,043 test rows. Model selection in this coursework run uses the highest F1-score, with ROC-AUC as a tie-breaker. Decision Tree has the highest F1-score (0.345); Random Forest has the highest ROC-AUC (0.803) and precision (0.668). Accuracy is not used alone because the target is imbalanced. Results may vary with software versions or settings.

## Data and evaluation caveats

The literal `unknown` values are treated as categories; there are no blank missing cells or exact duplicate rows in this dataset. The chosen Decision Tree misses a substantial share of subscribers at the default threshold, so it should be seen as a prioritization aid rather than an automatic decision system. Threshold choice, validation on later campaigns, contact costs, and customer experience should be considered before practical use.

## Academic reference

Moro, S., Laureano, R., & Cortez, P. (2011). *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. Proceedings of the European Simulation and Modelling Conference (ESM'2011), pp. 117–121.

## Student fields to confirm before submission

- Name: Md. Siddiq Juber
- Roll No.: 160124733363
- Section: replace `[Enter Section]` with the correct section.
- Faculty: replace `[Enter Faculty Name]` with the assigned faculty name.
