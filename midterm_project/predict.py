#!/usr/bin/env python
"""
predict.py — Flask Prediction Microservice
Smart Grid "Dunkelflaute" Shortfall Predictor (Germany)

Accepts JSON weather parameters via POST /predict and returns the
predicted 24-hour-ahead residual load along with a Dunkelflaute risk alert.
"""

import pickle
from flask import Flask, request, jsonify


# Load the serialized LightGBM model at startup
with open('dunkelflaute_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('dunkelflaute-prediction-service')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict 24-hour-ahead residual load from weather features.

    Expected JSON payload:
    {
        "temperature": 2.5,
        "wind_speed_100m": 18.3,
        "direct_radiation": 0.0,
        "diffuse_radiation": 0.0,
        "hour": 3,
        "month": 1,
        "is_winter": 1,
        "wind_speed_10m": 10.1,
        "total_radiation": 0.0,
        "wind_chill_index": 2.525
    }
    """
    grid_features = request.get_json()

    # Structure feature array matching the model's training column order
    input_data = [[
        grid_features['temperature'],
        grid_features['wind_speed_100m'],
        grid_features['direct_radiation'],
        grid_features['diffuse_radiation'],
        grid_features['hour'],
        grid_features['month'],
        grid_features.get('is_winter', int(grid_features['month'] in [11, 12, 1, 2, 3])),
        grid_features.get('wind_speed_10m', grid_features['wind_speed_100m'] * 0.5),
        grid_features.get('total_radiation',
                          grid_features['direct_radiation'] + grid_features['diffuse_radiation']),
        grid_features.get('wind_chill_index',
                          grid_features['temperature'] * grid_features.get('wind_speed_10m',
                          grid_features['wind_speed_100m'] * 0.5) / 10),
    ]]

    prediction = model.predict(input_data)[0]

    # Business logic: high residual load indicates severe Dunkelflaute stress
    # Threshold ~45 GW represents critical shortfall zone for the German grid
    shortfall_alert = bool(prediction > 45000)

    return jsonify({
        'predicted_residual_load_mw': round(float(prediction), 2),
        'dunkelflaute_risk_alert': shortfall_alert,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

