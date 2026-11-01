# DMV Practical 1: Sales Data from Multiple File Formats
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# Ensure script runs from its own directory so data files are always found
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load File 1: CSV
csv = pd.read_csv("sales_data_sample.csv", encoding="cp1252")
print("CSV Loaded. Shape:", csv.shape)

# 2. Load File 2: Excel (.xlsx)
ed = pd.read_excel("Sample-Sales-Data.xlsx")
print("Excel Loaded. Shape:", ed.shape)

# 3. Load File 3: JSON
with open("customers.json", "r") as f:
    cust_df = pd.DataFrame(json.load(f))
print("JSON Loaded. Shape:", cust_df.shape)

# 4. Clean Data
csv['ORDERDATE'] = pd.to_datetime(csv['ORDERDATE'])
csv['TERRITORY'] = csv['TERRITORY'].fillna('Unknown')
csv['YEAR_MONTH'] = csv['ORDERDATE'].dt.to_period('M')

# 5. Key Metrics
total_revenue = csv['SALES'].sum()
print(f"Total Revenue: ${total_revenue:,.2f}")

product_sales = csv.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
monthly_sales = csv.groupby('YEAR_MONTH')['SALES'].sum()
monthly_sales.index = monthly_sales.index.astype(str)

# 6. Visualizations
plt.figure(figsize=(10, 5))
product_sales.plot(kind='bar', color='teal')
plt.title("Product-wise Sales")
plt.ylabel("Sales ($)")
plt.show()

plt.figure(figsize=(10, 5))
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trends")
plt.ylabel("Sales ($)")
plt.show()

csv.groupby('TERRITORY')['SALES'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title("Sales by Territory")
plt.ylabel("")
plt.show()
