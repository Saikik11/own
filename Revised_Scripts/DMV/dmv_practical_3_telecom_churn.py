# DMV Practical 3: Telecom Customer Churn Data Cleaning & Modeling
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset
data = pd.read_csv("Telcom_Customer_Churn.csv")

# 2. Clean TotalCharges (fix whitespace issue) and drop duplicates
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', np.nan), errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
data.drop_duplicates(inplace=True)

# 3. Encode Categorical Columns
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# 4. Scale Numeric Columns
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])

# 5. Train & Evaluate Models
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred))

# 6. Visualizations
sns.countplot(x='Churn', data=data)
plt.title("Churn Distribution")
plt.show()

sns.boxplot(x='Churn', y='tenure', data=data)
plt.title("Tenure vs Churn")
plt.show()
