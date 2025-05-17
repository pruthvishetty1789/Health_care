from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

DISEASE_MODELS = {
    "Diabetes": {"port": 5001, "file": "diabetes.py"},
    "LungCancer": {"port": 5003, "file": "lung_cancer.py"},
    "Stroke": {"port": 5004, "file": "stroke.py"},
    "HeartDisease": {"port": 5002, "file": "heart_disease.py"},
}

@app.route('/run', methods=['POST'])
def run_disease():
    data = request.get_json()
    disease = data.get("disease")

    if disease not in DISEASE_MODELS:
        return jsonify({"error": "Invalid disease selected"}), 400

    model_info = DISEASE_MODELS[disease]
    port = model_info["port"]
    script_path = model_info["file"]

    try:
        # Check if port is already in use (server already running)
        if os.system(f"netstat -an | find \"{port}\"") == 0:
            return jsonify({"message": f"{disease} server is already running on port {port}"}), 200

        # Run model script in a subprocess
        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({"message": f"{disease} server started", "port": port}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
