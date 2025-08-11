import numpy as np
from keras.models import load_model
import joblib

class StressModel:
    def __init__(self, model_path="model.h5", scaler_path="scaler.pkl", encoder_path="encoder.pkl"):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

    def predict_stress(self, input_data):
        """
        input_data: numpy array shape (n_samples, n_features)
        """
        # Normalize input data
        input_scaled = self.scaler.transform(input_data).reshape(-1, 1, input_data.shape[1])
        pred_probs = self.model.predict(input_scaled)
        pred_class_indices = np.argmax(pred_probs, axis=1)
        pred_labels = self.encoder.categories_[0][pred_class_indices]
        return pred_labels, pred_probs
