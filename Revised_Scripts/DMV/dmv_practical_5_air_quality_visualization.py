# DMV Practical 5: Air Quality Index (AQI) Trends
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# 1. Load Dataset
data = pd.read_csv("data.csv", encoding="cp1252", low_memory=False)

# 2. Harmonize State names
data['state'] = data['state'].replace({'Uttaranchal': 'Uttarakhand'})

# 3. Impute Missing Values with Mean
val_cols = ['so2', 'no2', 'rspm', 'spm']
val_cols = [c for c in val_cols if c in data.columns]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data[val_cols] = imputer.fit_transform(data[val_cols])

# 4. Visualizations: Top 5 and Bottom 5 States by NO2
state_rank = data.groupby('state')['no2'].median().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
state_rank.head(5).plot(kind='bar', ax=axes[0], color='crimson')
axes[0].set_title("Top 5 States - Highest NO2")
axes[0].set_ylabel("NO2 (ug/m3)")

state_rank.tail(5).plot(kind='bar', ax=axes[1], color='forestgreen')
axes[1].set_title("Bottom 5 States - Lowest NO2")
axes[1].set_ylabel("NO2 (ug/m3)")
plt.tight_layout()
plt.show()

# 5. Correlation Heatmap
sns.heatmap(data[val_cols].corr(), annot=True, cmap="YlOrRd")
plt.title("Pollutant Correlation Heatmap")
plt.show()
