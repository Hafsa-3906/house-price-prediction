from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    size = float(request.form["size"])
    prediction = model.predict([[size]])

    return render_template(
        "index.html",
        prediction_text=f"Estimated House Price: ${prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)