from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the model
model = joblib.load("cycle_predictor_model.pkl")

# Create the app
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Cycle Predictor API is live on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([data["features"]])
        prediction = model.predict(features)
        return jsonify({
            "predicted_cycle_length": round(float(prediction[0]), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
