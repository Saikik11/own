# ML Practical 5: Random Forest Car Evaluation Classifier
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree

# 1. Load Dataset
df = pd.read_csv("car_evaluation.csv", header=None)
df.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] if df.shape[1] == 7 else ['buying', 'maint', 'doors', 'lug_boot', 'safety', 'class']

# 2. Encode Categorical Variables
for col in df.columns:
    df[col], _ = pd.factorize(df[col])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 4. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# 5. Predict and Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Feature Importance Bar Plot
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# 7. Native Decision Tree Plot
plt.figure(figsize=(12, 6))
plot_tree(clf.estimators_[0], max_depth=2, feature_names=X.columns, filled=True)
plt.title("Decision Tree from Forest (Depth=2)")
plt.show()
