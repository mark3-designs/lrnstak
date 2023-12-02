import traceback
import os
import sys
import logging
import base64
import json
import joblib
import requests
import tempfile
from flask import Flask, request, jsonify
from lrnstak.training_module import ModelTrainer


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

MODEL_REGISTRY_URL = 'http://registry:5000/models'

@app.route('/train/<string:model_name>', methods=['POST'])
def train_and_save_model(model_name):
    tmp = tempfile.NamedTemporaryFile().name
    try:
        data = request.json['data']
        version = request.json['version']
        parameters = request.json.get('parameters', None)
        training_data = request.json.get('training_data', None)
        testing_data = request.json.get('testing_data', None)
        validation_data = request.json.get('validation_data', None)

        if parameters is None:
            return jsonify({'message': f'Training parameters are required.'}), 400

        app.logger.info(f"Training {model_name}/{version} {json.dumps(parameters)}")

        model_trainer = ModelTrainer(parameters)

        trained_model, results = model_trainer.train(data, training_data)

        joblib.dump(trained_model, tmp)

        with open(tmp, 'rb') as file:
            model_bytes = file.read()

        trained_with = []
        trained_with.append(data)
        if training_data is not None:
            trained_with.append(training_data)

        # POST the serialized model to the model registry
        response = requests.post(f'{MODEL_REGISTRY_URL}/json', json={
            'model': base64.b64encode(model_bytes).decode('utf-8'),
            'name': model_name,
            'version': version,
            'parameters': parameters,
            'training_results': results,
            'training_data': trained_with
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

