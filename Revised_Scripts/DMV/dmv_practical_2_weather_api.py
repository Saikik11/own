# DMV Practical 2: Weather Data via OpenWeatherMap API
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fetch Data from API
api_key = "fb365aa6104829b44455572365ff3b4e"
lat, lon = 18.5204, 73.8567  # Pune
url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"

response = requests.get(url)
data = response.json()

# 2. Parse JSON into DataFrame
records = []
for entry in data.get('list', []):
    dt = datetime.datetime.fromtimestamp(entry['dt'])
    temp = entry['main']['temp']
    hum = entry['main']['humidity']
    press = entry['main']['pressure']
    speed = entry['wind']['speed']
    desc = entry['weather'][0]['description']
    records.append([dt, temp, hum, press, speed, desc])

df = pd.DataFrame(records, columns=["DateTime", "Temperature", "Humidity", "Pressure", "Wind_Speed", "Description"])
print("Weather data loaded successfully. Shape:", df.shape)
print(df.head(3))

# 3. Visualizations
plt.figure(figsize=(10, 4))
sns.lineplot(x="DateTime", y="Temperature", data=df, marker='o', color='orange')
plt.title("Temperature Trend Over Time (Celsius)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df[["Temperature", "Humidity", "Pressure", "Wind_Speed"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
