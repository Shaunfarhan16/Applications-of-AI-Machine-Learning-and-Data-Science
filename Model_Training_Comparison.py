import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

# 1. Load dataset
df = pd.read_csv("C:/Applicarion of AI,ML,DS/Telco-Customer-Churn.csv")

# 2. Convert TotalCharges to numeric, impute missing with median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 3. Ensure correct numeric types
df = df.astype({
    'SeniorCitizen': 'int64',
    'tenure': 'int64',
    'MonthlyCharges': 'float64'
})

# 4. Encode binary categorical columns using map (no replace warnings)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].str.strip().str.capitalize().map({'Yes': 1, 'No': 0}).astype(int)

# 5. Encode gender to binary
df['gender'] = df['gender'].str.strip().map({'Male': 1, 'Female': 0}).astype(int)

# 6. Clean multi-class categorical features
multi_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
for col in multi_cols:
    df[col] = df[col].str.strip()

# 7. Drop customerID and encode
df_clean = df.drop(columns=['customerID'])
df_encoded = pd.get_dummies(df_clean, columns=multi_cols, drop_first=True)

# 8. Train/test split
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 9. Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(
        scale_pos_weight=(y_train==0).sum() / (y_train==1).sum(),
        eval_metric='logloss',
        random_state=42
    )
}

# 10. Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    })

# 11. Display results
results_df = pd.DataFrame(results)
print(results_df)

import shap

# Fit XGBoost again for SHAP
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42,
                    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum())
xgb.fit(X_train, y_train)

# SHAP analysis
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test)

# SHAP summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns)

from sklearn.metrics import RocCurveDisplay

best_model = xgb  # or whichever performed best
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title('ROC Curve – XGBoost')
plt.show()


