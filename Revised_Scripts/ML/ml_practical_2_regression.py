# ML Practical 2: Uber Fare Prediction (Linear, Ridge, Lasso Regression)
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load and Clean
df = pd.read_csv("uber.csv", nrows=50000)
df.drop(['Unnamed: 0', 'key'], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

# 2. Haversine Distance (km)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine(df['pickup_longitude'], df['pickup_latitude'],
                               df['dropoff_longitude'], df['dropoff_latitude'])

# 3. Remove Outliers
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 500)]
df = df[(df['distance_km'] >= 1) & (df['distance_km'] <= 130)]

# 4. Features and Target
X = df[['distance_km', 'passenger_count']]
y = df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train and Compare Models
for name, model in [("Linear", LinearRegression()),
                     ("Ridge", Ridge(alpha=1.0)),
                     ("Lasso", Lasso(alpha=0.1))]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name}: R²={r2_score(y_test, pred):.4f}, RMSE={np.sqrt(mean_squared_error(y_test, pred)):.4f}")

# 6. Actual vs Predicted Plot
lr = LinearRegression().fit(X_train, y_train)
plt.scatter(y_test, lr.predict(X_test), alpha=0.4)
plt.plot([0, 100], [0, 100], 'r--')
plt.xlabel("Actual Fare ($)")
plt.ylabel("Predicted Fare ($)")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()
