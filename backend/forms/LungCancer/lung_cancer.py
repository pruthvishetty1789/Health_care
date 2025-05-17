import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Load Model and Scaler
with open("lung_cancer_model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        features = [
            int(data["Age"]),
            int(data["Gender"]),
            int(data["Smoking"]),
            int(data["Yellow_Fingers"]),
            int(data["Anxiety"]),
            int(data["Peer_Pressure"]),
            int(data["Chronic_Disease"]),
            int(data["Fatigue"]),
            int(data["Allergy"]),
            int(data["Wheezing"]),
            int(data["Alcohol_Consuming"]),
            int(data["Coughing"])
        ]

        scaled_data = scaler.transform([features])
        prediction = model.predict(scaled_data)[0]
        result = "You may have lung cancer" if prediction == 1 else "You do not have lung cancer"

        if request.is_json:
            return jsonify({"prediction": result})
        else:
            return render_template("index.html", result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5003, debug=True)
