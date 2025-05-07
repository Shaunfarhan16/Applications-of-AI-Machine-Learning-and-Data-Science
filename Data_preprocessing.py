import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("C:/Applicarion of AI,ML,DS/Telco-Customer-Churn.csv")

# 2. Handle TotalCharges: convert to numeric & median impute
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # invalid → NaN
median_tc = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_tc)               # assign back, no chained inplace

# 3. Ensure correct dtypes
df = df.astype({
    'SeniorCitizen':  'int64',
    'MonthlyCharges': 'float64',
    'tenure':         'int64'
})

# 4. Standardize & encode binary categorical columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
# strip spaces and capitalize
for col in binary_cols:
    df[col] = df[col].str.strip().str.capitalize()
# map Yes/No to 1/0 using Series.map (avoids downcasting warning)
yes_no_map = {'Yes': 1, 'No': 0}
for col in binary_cols:
    df[col] = df[col].map(yes_no_map)

# 5. Clean multi-level categorical columns
multi_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
for col in multi_cols:
    df[col] = df[col].str.strip()

# 6. Outlier detection (IQR counts only)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
outlier_summary = {}
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outlier_summary[col] = df[(df[col] < lb) | (df[col] > ub)].shape[0]
# (We log counts but do not remove any rows)

# 7. Drop identifier
df_clean = df.drop(columns=['customerID'])

# 8. One-hot encode multi-level categoricals (drop_first=True)
df_encoded = pd.get_dummies(df_clean, columns=multi_cols, drop_first=True)

# 9. Train/test split (80/20 stratified on churn)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 10. Save cleaned and encoded datasets (optional)
df_clean.to_csv('Telco_Customer_Churn_Cleaned.csv', index=False)
df_encoded.to_csv('Telco_Customer_Churn_Encoded.csv', index=False)

# 11. Summary printout
print("Cleaned data shape:", df_clean.shape)
print("Encoded data shape:", df_encoded.shape)
print("Median TotalCharges imputed:", median_tc)
print("Outlier counts (IQR):", outlier_summary)
print("Train/test shapes:", X_train.shape, X_test.shape)
print("Churn distribution:\n", df_clean['Churn'].value_counts(normalize=True))

import matplotlib.pyplot as plt
import seaborn as sns

# Churn distribution
sns.countplot(x='Churn', data=df_clean)
plt.title('Churn Class Distribution')
plt.show()

# Churn rate by contract type
sns.barplot(x='Contract', y='Churn', data=df_clean)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Proportion of Churn')
plt.show()

# Tenure histogram split by churn
sns.histplot(data=df_clean, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn')
plt.show()

# Monthly charges boxplot
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_clean)
plt.title('Monthly Charges vs. Churn')
plt.show()






