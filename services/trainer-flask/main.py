import traceback
import os
import sys
import logging
import base64
import json
import joblib
import requests
import tempfile
import time
from datetime import datetime
from flask import Flask, request, jsonify
from lrnstak.training_module import ModelTrainer


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Track startup time for health endpoint
app.start_time = time.time()

@app.route('/health')
def health_check():
    """
    Health check endpoint for Docker health monitoring
    Returns service status and basic system info
    """
    try:
        # Basic service health information
        health_data = {
            'status': 'healthy',
            'service': 'lrnstak-trainer',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.start_time
        }
        
        return health_data, 200
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500

MODEL_REGISTRY_URL = 'http://registry:5000/models'
#MODEL_REGISTRY_URL = 'http://10.6.88.8:5201/models'

@app.route('/train/<string:model_name>', methods=['POST'])
def train_and_save_model(model_name):
    tmp = tempfile.NamedTemporaryFile().name
    try:
        version = request.json['version']
        parameters = request.json.get('parameters', None)
        training_data = request.json.get('training_data', None)
        testing_data = request.json.get('testing_data', None)
        validation_data = request.json.get('validation_data', None)

        if parameters is None:
            return jsonify({'message': f'parameters key is required.'}), 400

        if training_data is None:
            return jsonify({'message': f'training_data key is required.'}), 400

        app.logger.info(f"Training {model_name}/{version} {json.dumps(parameters, indent=2)}")

        model_trainer = ModelTrainer(parameters)

        # app.logger.info(f"Training Data {json.dumps(training_data, indent=2)} ==")
        trained_model, results = model_trainer.train(training_data, testing_data)

        joblib.dump(trained_model, tmp)

        with open(tmp, 'rb') as file:
            model_bytes = file.read()

        # POST the serialized model to the model registry
        response = requests.post(f'{MODEL_REGISTRY_URL}/json', json={
            'model': base64.b64encode(model_bytes).decode('utf-8'),
            'name': model_name,
            'version': version,
            'parameters': parameters,
            'training_results': results,
            'training_data': training_data
            })

        if response.status_code == 200:
            app.logger.info(f'Model {model_name}_{version} trained and saved.')
            return jsonify({'message': f'Model {model_name}_{version} trained and saved.'})
        else:
            app.logger.error(response.text)
            app.logger.error(f'Failed to save the model. Status code: {response.status_code}')
            return jsonify({'error': f'Failed to save the model. Status code: {response.status_code}'}), response.status_code

    except Exception as e:
        traceback.print_exc()
        app.logger.error(str(e))
        app.logger.error(f'Failed to save the model.')
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp)
        except:
            ## do nothing
            pass


if __name__ == '__main__':
    print("start")
    app.run(debug=True,host='0.0.0.0',port=5000)

