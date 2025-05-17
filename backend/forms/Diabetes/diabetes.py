from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]

        # Scale and predict
        scaled_data = scaler.transform([features])
        prediction = model.predict(scaled_data)[0]

        # Show result on the same page
        result_message = "You have disease" if prediction == 1 else "You don't have disease"
        return render_template("index.html", result=result_message)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=5001, debug=True)
