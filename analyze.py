"""Bank Marketing classification case study analysis.

Run from the project root with: python analyze.py
The primary models intentionally exclude `duration` because it is unknown before a call.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings('ignore', category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'bank-full.csv'
FIG_DIR = ROOT / 'figures'
OUT_DIR = ROOT / 'outputs'
MODEL_DIR = ROOT / 'models'
for d in (FIG_DIR, OUT_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Task 1: dataset load and inspection. UCI Bank Marketing CSV uses semicolons.
df = pd.read_csv(DATA_PATH, sep=';')
if 'y' not in df.columns:
    raise ValueError("Expected target column 'y' in bank-full.csv")
# Convert target to a binary label. `duration` is excluded for realistic pre-contact prediction.
y = (df['y'].astype(str).str.lower() == 'yes').astype(int)
DROP_COLS = ['y', 'duration']
X = df.drop(columns=DROP_COLS)
NUMERIC_FEATURES = X.select_dtypes(include=np.number).columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude=np.number).columns.tolist()

# Stratified 80:20 split ensures the imbalanced target keeps similar proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Preprocessing is fitted on the training partition only, preventing data leakage.
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipe, NUMERIC_FEATURES),
    ('categorical', categorical_pipe, CATEGORICAL_FEATURES),
], remainder='drop')

models = {
    'Logistic Regression': LogisticRegression(max_iter=1500, solver='liblinear', random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, min_samples_leaf=5),
    'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=160, min_samples_leaf=2, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1),
}

# EDA outputs
sns.set_theme(style='whitegrid', context='notebook')
counts = df['y'].value_counts().reindex(['no', 'yes'])
fig, ax = plt.subplots(figsize=(7.5, 4.6))
colors = ['#527da8', '#e38b48']
bars = ax.bar(['Did not subscribe', 'Subscribed'], counts.values, color=colors)
ax.set_title('Bank Marketing: Target Class Distribution')
ax.set_xlabel('Term-deposit subscription outcome')
ax.set_ylabel('Number of customers')
ax.bar_label(bars, labels=[f'{v:,}\n({v / len(df):.1%})' for v in counts.values], padding=3)
fig.tight_layout(); fig.savefig(FIG_DIR / 'target_distribution.png', dpi=170); plt.close(fig)

job_rates = (df.assign(subscribed=(df['y'] == 'yes').astype(int)).groupby('job', observed=False)['subscribed'].mean().sort_values()) * 100
fig, ax = plt.subplots(figsize=(9, 5.2))
job_rates.plot(kind='barh', ax=ax, color='#5b8e7d')
ax.set_title('Term-Deposit Subscription Rate by Job')
ax.set_xlabel('Customers who subscribed (%)'); ax.set_ylabel('Job category')
for p in ax.patches: ax.annotate(f'{p.get_width():.1f}%', (p.get_width(), p.get_y()+p.get_height()/2), xytext=(4,0), textcoords='offset points', va='center', fontsize=8)
ax.set_xlim(0, max(job_rates)*1.18)
fig.tight_layout(); fig.savefig(FIG_DIR / 'subscription_by_job.png', dpi=170); plt.close(fig)

month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
month_rates = (df.assign(subscribed=(df['y'] == 'yes').astype(int)).groupby('month', observed=False)['subscribed'].mean().reindex(month_order)) * 100
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(month_rates.index, month_rates.values, color='#527da8')
ax.set_title('Term-Deposit Subscription Rate by Contact Month')
ax.set_xlabel('Month'); ax.set_ylabel('Customers who subscribed (%)')
for i, v in enumerate(month_rates.values): ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, float(month_rates.max())*1.18)
fig.tight_layout(); fig.savefig(FIG_DIR / 'subscription_by_month.png', dpi=170); plt.close(fig)

# Numerical feature relationships: use quantile bins for highly skewed balance and attempts.
plot_df = df[['age','balance','campaign','y']].copy()
plot_df['subscription'] = (plot_df['y'] == 'yes').astype(int)
plot_df['age_group'] = pd.cut(plot_df['age'], bins=[17,25,35,45,55,65,100], right=True, include_lowest=True,
                              labels=['18-25','26-35','36-45','46-55','56-65','66+'])
age_rates = plot_df.groupby('age_group', observed=False)['subscription'].mean()*100
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(age_rates.index.astype(str), age_rates.values, color='#d28b52')
ax.set_title('Term-Deposit Subscription Rate by Age Group')
ax.set_xlabel('Age group (years)'); ax.set_ylabel('Customers who subscribed (%)')
for i,v in enumerate(age_rates.values): ax.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_ylim(0, max(age_rates.values)*1.18)
fig.tight_layout(); fig.savefig(FIG_DIR / 'subscription_by_age_group.png', dpi=170); plt.close(fig)

campaign_bins = pd.cut(df['campaign'], bins=[0,1,2,3,5,10,100], include_lowest=True,
                       labels=['1','2','3','4-5','6-10','11+'])
campaign_rates = pd.DataFrame({'campaign_group': campaign_bins, 'subscribed': (df['y']=='yes').astype(int)}).groupby('campaign_group', observed=False)['subscribed'].agg(['mean','size'])
campaign_rates['rate_pct'] = campaign_rates['mean']*100
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(campaign_rates.index.astype(str), campaign_rates['rate_pct'], color='#6c8fa7')
ax.set_title('Subscription Rate by Number of Contacts in Current Campaign')
ax.set_xlabel('Number of contacts during current campaign'); ax.set_ylabel('Customers who subscribed (%)')
for i, v in enumerate(campaign_rates['rate_pct'].values): ax.text(i, v+0.35, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_ylim(0, max(campaign_rates['rate_pct'].values)*1.20)
fig.tight_layout(); fig.savefig(FIG_DIR / 'subscription_by_campaign_count.png', dpi=170); plt.close(fig)

# Missing / unknown / duplicate audit
missing = df.isna().sum().sort_values(ascending=False)
unknown_counts = {c: int((df[c].astype(str).str.lower() == 'unknown').sum()) for c in df.columns if df[c].dtype == 'object'}
summary = {
    'rows': int(df.shape[0]), 'columns_including_target': int(df.shape[1]),
    'duplicate_rows': int(df.duplicated().sum()), 'missing_cells': int(df.isna().sum().sum()),
    'target_counts': {str(k): int(v) for k, v in df['y'].value_counts().items()},
    'target_percentages': {str(k): float(v) for k, v in (df['y'].value_counts(normalize=True)*100).items()},
    'unknown_category_counts': unknown_counts,
    'excluded_features': ['duration'],
    'numeric_features_used': NUMERIC_FEATURES,
    'categorical_features_used': CATEGORICAL_FEATURES,
    'train_rows': int(len(X_train)), 'test_rows': int(len(X_test)),
}
(OUT_DIR / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
missing.rename('missing_count').to_csv(OUT_DIR / 'missing_value_audit.csv')
pd.DataFrame({'unknown_count': pd.Series(unknown_counts)}).to_csv(OUT_DIR / 'unknown_category_audit.csv')

# Model training and evaluation using the same split, outcome and preprocessing pipeline.
results = []
all_estimators = {}
all_predictions = {}
for name, estimator in models.items():
    pipe = Pipeline([('preprocess', preprocessor), ('model', estimator)])
    print(f'Training {name}...', flush=True)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    row = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'True Negatives': int(confusion_matrix(y_test, y_pred, labels=[0,1])[0,0]),
        'False Positives': int(confusion_matrix(y_test, y_pred, labels=[0,1])[0,1]),
        'False Negatives': int(confusion_matrix(y_test, y_pred, labels=[0,1])[1,0]),
        'True Positives': int(confusion_matrix(y_test, y_pred, labels=[0,1])[1,1]),
    }
    results.append(row)
    all_estimators[name] = pipe
    all_predictions[name] = (y_pred, y_prob)
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=['Predicted no','Predicted yes'], yticklabels=['Actual no','Actual yes'], ax=ax)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted class'); ax.set_ylabel('Actual class')
    fig.tight_layout(); safe = name.lower().replace(' ','_').replace('-','')
    fig.savefig(FIG_DIR / f'confusion_matrix_{safe}.png', dpi=170); plt.close(fig)

metrics = pd.DataFrame(results).sort_values(['F1-score','ROC-AUC'], ascending=False).reset_index(drop=True)
metrics.to_csv(OUT_DIR / 'model_comparison.csv', index=False, float_format='%.6f')

# Common grouped comparison plot (accuracy, precision, recall, F1).
plot_metrics = metrics.set_index('Model')[['Accuracy','Precision','Recall','F1-score']]
ax = plot_metrics.plot(kind='bar', figsize=(10, 5.5), width=0.82)
handles, labels = ax.get_legend_handles_labels(); ax.legend_.remove()
fig.legend(handles, labels, title='Metric', loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=8, title_fontsize=8)
ax.set_title('Classification Model Comparison on the Held-out Test Set', pad=8)
ax.set_xlabel('Classification algorithm'); ax.set_ylabel('Score (0 to 1)')
ax.set_ylim(0, 1.02); plt.xticks(rotation=15, ha='right'); plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.savefig(FIG_DIR / 'model_metrics_comparison.png', dpi=170); plt.close()

# ROC comparison
fig, ax = plt.subplots(figsize=(7.6, 6))
for name, (yp, prob) in all_predictions.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_value = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_value:.3f})')
ax.plot([0,1],[0,1], linestyle='--', color='gray', label='Random baseline')
ax.set_title('ROC Curves for Bank Marketing Classifiers')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate (Recall)')
ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(loc='lower right', fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / 'roc_curves.png', dpi=170); plt.close(fig)

# Define a transparent primary selection criterion: maximize F1, then ROC-AUC.
best_name = str(metrics.iloc[0]['Model'])
best_model = all_estimators[best_name]
joblib.dump(best_model, MODEL_DIR / 'best_bank_marketing_model.joblib')
(OUT_DIR / 'best_model.json').write_text(json.dumps({
    'best_model_by_f1_then_roc_auc': best_name,
    'selection_criterion': 'Highest test F1-score; ROC-AUC used as tie-breaker. Accuracy is not used alone because y is imbalanced.',
    'warning': 'The test comparison is for this coursework assignment. For operational deployment, select/tune using validation or cross-validation and keep a final untouched test set.'
}, indent=2), encoding='utf-8')

# Detailed classification reports per model.
for name, (yp, prob) in all_predictions.items():
    safe = name.lower().replace(' ','_').replace('-','')
    (OUT_DIR / f'classification_report_{safe}.txt').write_text(
        f'{name}\n\n' + classification_report(y_test, yp, target_names=['No subscription','Subscription'], zero_division=0), encoding='utf-8')

print('\nDATA SUMMARY')
print(json.dumps(summary, indent=2))
print('\nMODEL COMPARISON')
print(metrics[['Model','Accuracy','Precision','Recall','F1-score','ROC-AUC']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print(f'\nBest model by F1 then ROC-AUC: {best_name}')
