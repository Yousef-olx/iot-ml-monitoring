"""
Flask API for ML Model Predictions
Serves the trained XGBoost model via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js to call this API

# Global variables for model and scaler
model = None
scaler = None

# Load model immediately when module is imported (for gunicorn)
def init_app():
    """Initialize the application - called on module import"""
    global model, scaler
    if model is None:
        print("Initializing Flask app and loading model...")
        load_model()

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    try:
        model_path = 'rf_model.pkl'
        scaler_path = 'scaler.pkl'
        
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please run 'python predict_fast.py' first to train the model")
            return False
            
        if not os.path.exists(scaler_path):
            print(f"ERROR: Scaler file not found at {scaler_path}")
            print("Please run 'python predict_fast.py' first to train the model")
            return False
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        return False

def create_features(df):
    """
    Create all engineered features from base sensor readings
    Must match the EXACT feature names used during training
    """
    # Base feature columns
    feature_cols = ['temperature', 'vibration', 'pressure', 'humidity']
    
    # Interaction features (use SHORT names like training script)
    df['temp_vib_interaction'] = df['temperature'] * df['vibration']
    df['temp_pressure_interaction'] = df['temperature'] * df['pressure']
    df['vib_pressure_interaction'] = df['vibration'] * df['pressure']
    
    # Humidity interaction features
    df['temperature_humidity_interaction'] = df['temperature'] * df['humidity']
    df['vibration_humidity_interaction'] = df['vibration'] * df['humidity']
    df['pressure_humidity_interaction'] = df['pressure'] * df['humidity']
    
    # Statistical features across all base features
    df['feature_mean'] = df[feature_cols].mean(axis=1)
    df['feature_std'] = df[feature_cols].std(axis=1)
    df['feature_range'] = df[feature_cols].max(axis=1) - df[feature_cols].min(axis=1)
    
    # Squared features
    df['temperature_squared'] = df['temperature'] ** 2
    df['vibration_squared'] = df['vibration'] ** 2
    df['pressure_squared'] = df['pressure'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    
    # Critical condition flag
    df['critical_condition'] = (
        (df['temperature'] > 100) | 
        (df['vibration'] > 5) | 
        (df['pressure'] > 150)
    ).astype(int)
    
    return df

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None or scaler is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please train the model first.',
            'model_loaded': False
        }), 503
    
    return jsonify({
        'status': 'ok',
        'message': 'ML API is running',
        'model_loaded': True
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction for a single sensor reading
    
    Expected JSON body:
    {
        "temperature": 75.5,
        "vibration": 3.2,
        "pressure": 85.0,
        "humidity": 50.0
    }
    """
    try:
        if model is None or scaler is None:
            print("❌ ERROR: Model not loaded")
            return jsonify({
                'error': 'Model not loaded. Please run predict_fast.py first to train the model.'
            }), 503
        
        # Get JSON data
        data = request.get_json()
        print(f"📥 Received data: {data}")
        
        if not data:
            print("❌ ERROR: No data provided")
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['temperature', 'vibration', 'pressure', 'humidity']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ ERROR: Missing fields: {missing_fields}")
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        print(f"✓ All required fields present")
        
        # Create DataFrame with base features
        input_df = pd.DataFrame([{
            'temperature': float(data['temperature']),
            'vibration': float(data['vibration']),
            'pressure': float(data['pressure']),
            'humidity': float(data['humidity'])
        }])
        
        # Create engineered features
        input_df = create_features(input_df)
        
        # CRITICAL: Ensure columns are in the EXACT order the scaler expects
        # This order comes from df.select_dtypes() which sorts alphabetically!
        expected_columns = [
            'temperature', 'pressure', 'vibration', 'humidity',  # Note: alphabetical, not logical order!
            'temp_vib_interaction', 'temp_pressure_interaction', 'vib_pressure_interaction',
            'temperature_humidity_interaction', 'vibration_humidity_interaction', 'pressure_humidity_interaction',
            'feature_mean', 'feature_std', 'feature_range',
            'temperature_squared', 'vibration_squared', 'pressure_squared', 'humidity_squared',
            'critical_condition'
        ]
        input_df = input_df[expected_columns]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Calculate risk score (probability of failure class)
        failure_risk = float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0])
        failure_risk = failure_risk * 100  # Convert to percentage
        
        # Determine risk level
        if failure_risk < 30:
            risk_level = 'Low'
            recommended_action = 'Continue normal operation'
        elif failure_risk < 60:
            risk_level = 'Medium'
            recommended_action = 'Schedule inspection within 7 days'
        elif failure_risk < 80:
            risk_level = 'High'
            recommended_action = 'Schedule maintenance within 48 hours'
        else:
            risk_level = 'Critical'
            recommended_action = 'Immediate maintenance required'
        
        # Prepare response
        response = {
            'prediction': str(prediction),
            'risk_score': round(failure_risk, 2),
            'risk_level': risk_level,
            'recommended_action': recommended_action,
            'input_values': {
                'temperature': float(data['temperature']),
                'vibration': float(data['vibration']),
                'pressure': float(data['pressure']),
                'humidity': float(data['humidity'])
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as ve:
        print(f"❌ ValueError: {str(ve)}")
        return jsonify({'error': f'Invalid data type: {str(ve)}'}), 400
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple sensor readings
    
    Expected JSON body:
    {
        "readings": [
            {"temperature": 75.5, "vibration": 3.2, "pressure": 85.0, "humidity": 50.0},
            {"temperature": 80.0, "vibration": 4.0, "pressure": 90.0, "humidity": 55.0}
        ]
    }
    """
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please run predict_fast.py first to train the model.'
            }), 503
        
        data = request.get_json()
        
        if not data or 'readings' not in data:
            return jsonify({'error': 'No readings provided'}), 400
        
        readings = data['readings']
        
        if not isinstance(readings, list) or len(readings) == 0:
            return jsonify({'error': 'Readings must be a non-empty array'}), 400
        
        results = []
        
        for idx, reading in enumerate(readings):
            try:
                # Create DataFrame
                input_df = pd.DataFrame([{
                    'temperature': float(reading['temperature']),
                    'vibration': float(reading['vibration']),
                    'pressure': float(reading['pressure']),
                    'humidity': float(reading['humidity'])
                }])
                
                # Create features
                input_df = create_features(input_df)
                
                # Scale and predict
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                failure_risk = float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0])
                failure_risk = failure_risk * 100
                
                if failure_risk < 30:
                    risk_level = 'Low'
                elif failure_risk < 60:
                    risk_level = 'Medium'
                elif failure_risk < 80:
                    risk_level = 'High'
                else:
                    risk_level = 'Critical'
                
                results.append({
                    'index': idx,
                    'prediction': str(prediction),
                    'risk_score': round(failure_risk, 2),
                    'risk_level': risk_level
                })
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'error': str(e)
                })
        
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("="*60)
    print("Starting Flask ML API Server")
    print("="*60)
    
    # Load model on startup
    if load_model():
        print("\nServer starting on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  GET  /health          - Check if API is running")
        print("  POST /predict         - Single prediction")
        print("  POST /batch-predict   - Multiple predictions")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\nFailed to load model. Please run 'python predict_fast.py' first.")
        print("="*60)

# Initialize app when imported by gunicorn
init_app()