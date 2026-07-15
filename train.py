# train.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("data/house_price.csv")

# Select input (X) and output (y)
X = data[["Size"]]
y = data["Price"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully!")
print("Model saved as model.pkl")