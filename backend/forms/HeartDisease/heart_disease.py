import pickle
import numpy as np
from flask import Flask, request, render_template

# Load Model and Scaler
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        features = [
            float(request.form["Age"]),
            float(request.form["Sex"]),
            float(request.form["Chest_pain_type"]),
            float(request.form["BP"]),
            float(request.form["Cholesterol"]),
            float(request.form["FBS_over_120"]),
            float(request.form["EKG_results"]),
            float(request.form["Max_HR"]),
            float(request.form["Exercise_angina"]),
            float(request.form["ST_depression"]),
            float(request.form["Slope_of_ST"]),
            float(request.form["Number_of_vessels_fluro"]),
            float(request.form["Thallium"])
        ]

        # Scale and predict
        scaled_data = scaler.transform([features])
        prediction = model.predict(scaled_data)[0]

        # Show result on the same page
        result_message = "You have heart disease" if prediction == 1 else "You do not have heart disease"
        return render_template("index.html", result=result_message)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=5002, debug=True)
    