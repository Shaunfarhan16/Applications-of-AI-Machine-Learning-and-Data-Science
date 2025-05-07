import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load encoded dataset
df = pd.read_csv(r"C:\Applicarion of AI,ML,DS\Telco_Customer_Churn_Cleaned.csv")

# Reload original for raw features if needed
df_original = pd.read_csv(r"C:\Applicarion of AI,ML,DS\Telco-Customer-Churn.csv")
df['Contract'] = df_original['Contract']
df['tenure'] = df_original['tenure']
df['MonthlyCharges'] = df_original['MonthlyCharges']
df['Churn'] = df_original['Churn'].map({'Yes': 1, 'No': 0})

sns.set(style="whitegrid")

# Figure 1: Churn rate by contract type
plt.figure(figsize=(6, 4))
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values()
sns.barplot(x=contract_churn.index, y=contract_churn.values, color='steelblue')
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.xlabel('Contract Type')
plt.tight_layout()
plt.show()

# Figure 2: Tenure histogram by churn status
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title('Customer Tenure Distribution by Churn Status')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.show()

# Figure 3: Box plot of MonthlyCharges vs Churn
plt.figure(figsize=(6, 4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs. Churn')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.ylabel('Monthly Charges (£)')
plt.tight_layout()
plt.show()

# Figure 4: Correlation heatmap (numeric only)
df_numeric = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
plt.figure(figsize=(6, 5))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Figure 5: Churn class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='pastel')
plt.title('Churn Class Distribution')
plt.xlabel('Churn (0=No, 1=Yes)')
plt.ylabel('Customer Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.tight_layout()
plt.show()
