# DMV Practical 6: Retail Sales Performance by Region & Category
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = pd.read_csv("customer_shopping_data.csv")

# 2. Clean Data & Total Amount
df.drop_duplicates(inplace=True)
df['total_amount'] = df['price'] * df['quantity'] if 'quantity' in df.columns else df['price']

# 3. Groupby Aggregations
region_sales = df.groupby("shopping_mall")["total_amount"].sum().sort_values(ascending=False)
category_sales_sorted = df.groupby("category")["total_amount"].sum().sort_values(ascending=True) # Fix: correctly defined
combined_sales = df.groupby(["shopping_mall", "category"])["total_amount"].sum().unstack().fillna(0)

# 4. Visualizations
plt.figure(figsize=(6, 6))
region_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales by Region (Shopping Mall)")
plt.ylabel("")
plt.show()

plt.figure(figsize=(9, 5))
category_sales_sorted.plot(kind='barh', color='teal')
plt.title("Sales by Product Category")
plt.xlabel("Total Sales")
plt.show()

plt.figure(figsize=(9, 5))
sns.heatmap(combined_sales, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Sales Heatmap by Region and Category")
plt.show()
