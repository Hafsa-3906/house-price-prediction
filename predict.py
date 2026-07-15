import joblib

# Load the trained model
model = joblib.load("model.pkl")

print("🏠 House Price Prediction")
print("-" * 30)

# Get house size from user
size = float(input("Enter house size (square feet): "))

# Predict price
predicted_price = model.predict([[size]])

print(f"\nEstimated House Price: ${predicted_price[0]:,.2f}")