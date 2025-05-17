import pickle
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import CORS

# Load the trained model and scaler
with open("stroke_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        data = request.form

        # Convert form data to feature list
        features = [
            float(data["age"]),
            int(data["hypertension"]),
            int(data["heart_disease"]),
            float(data["avg_glucose_level"]),
            float(data["bmi"]),
            int(data.get("gender_Male", 0)),
            int(data.get("ever_married_Yes", 0)),
            int(data.get("work_type_Private", 0)),
            int(data.get("work_type_Self-employed", 0)),
            int(data.get("work_type_children", 0)),
            int(data.get("Residence_type_Urban", 0)),
            int(data.get("smoking_status_formerly smoked", 0)),
            int(data.get("smoking_status_never smoked", 0)),
            int(data.get("smoking_status_smokes", 0)),
        ]

        # Scale the input
        scaled_data = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_data)[0]

        # Set message and color
        if prediction == 1:
            result = "⚠️ Stroke Detected! High Risk"
            color = "red"
        else:
            result = "✅ No Stroke Risk"
            color = "green"

        # Render result in HTML
        return render_template("index.html", result=result, color=color)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}", color="black")

if __name__ == "__main__":
    app.run(port=5004, debug=True)
