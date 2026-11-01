# DMV Practical 4: Real Estate Market Data Wrangling
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("Bengaluru_House_Data.csv")
df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

# 2. Extract BHK
df['bhk'] = df['size'].apply(lambda x: int(str(x).split()[0]))

# 3. Clean total_sqft range strings
def convert_sqft(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        try: return (float(tokens[0]) + float(tokens[1])) / 2.0
        except ValueError: return None
    try: return float(x)
    except ValueError: return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df.dropna(subset=['total_sqft'], inplace=True)

# 4. Feature Engineering: Price per SqFt
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# 5. Reduce Location Cardinality
location_stats = df['location'].value_counts()
locations_less_than_10 = location_stats[location_stats <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

# 6. Remove Outliers (< 300 sqft/BHK and IQR on price_per_sqft)
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price_per_sqft'] >= Q1 - 1.5*IQR) & (df['price_per_sqft'] <= Q3 + 1.5*IQR)]
print("Cleaned Shape:", df.shape)

# 7. Scatter Plot: SqFt vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='total_sqft', y='price', hue='bhk', data=df, palette='viridis')
plt.title("Total SqFt vs Price")
plt.show()
