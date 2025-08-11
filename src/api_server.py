# backend/api_server.py
from flask import Flask, request, jsonify
import numpy as np
from model_inference import StressModel

app = Flask(__name__)
model = StressModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expect JSON with "features": [[...], [...], ...]
    input_features = np.array(data['features'])
    labels, probs = model.predict_stress(input_features)
    return jsonify({
        'predictions': labels.tolist(),
        'probabilities': probs.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
