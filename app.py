import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tkinter as tk
from tkinter import messagebox

# === Load Dataset ===
df = pd.read_csv("air_quality_dataset.csv")
df.dropna(inplace=True)

# === Select features and target ===
features = ['Temperature', 'Humidity', 'Wind Speed', 'Pressure', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
target = 'AQI'

X = df[features]
y = df[target]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === GUI Code ===
def predict():
    try:
        values = [float(entries[f].get()) for f in features]
        prediction = model.predict([values])[0]
        messagebox.showinfo("Predicted AQI", f"Predicted Air Quality Index (AQI): {round(prediction, 2)}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for all fields.")

# Create GUI window
window = tk.Tk()
window.title("Air Quality Index Predictor")

entries = {}
for i, f in enumerate(features):
    tk.Label(window, text=f).grid(row=i, column=0, padx=10, pady=5)
    ent = tk.Entry(window)
    ent.grid(row=i, column=1, padx=10, pady=5)
    entries[f] = ent

tk.Button(window, text="Predict AQI", command=predict).grid(row=len(features), column=0, columnspan=2, pady=10)

window.mainloop()
