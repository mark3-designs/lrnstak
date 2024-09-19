from flask import Flask, request, jsonify
import sys
import logging
import base64
import joblib
from file_storage import Utils, Storage

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)

# In-memory storage
models = {}
utils = Utils()
storage = Storage()

@app.route('/models/<string:model_name>/<string:version>', methods=['GET'])
def get_model(model_name, version):
    model_key = f"{model_name}_{version}"
    try:
        if model_key in models:
            found = models[model_key]
        else:
            model_bytes, parameters = storage.read(model_name, version)
            serialized_model = base64.b64encode(model_bytes).decode('utf-8')
            models[model_key] = {
                "model": serialized_model,
                "parameters": parameters
                }

        return jsonify(models[model_key])

    except FileNotFoundError as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404

@app.route('/models/<string:model_name>/<string:version>/info', methods=['GET'])
def get_model_info(model_name, version):
    try:
        parameters, training_scores = storage.info(model_name, version)
        return jsonify({'parameters': parameters, 'training_scores': training_scores})
    except FileNotFoundError as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404

@app.route('/models/<string:model_name>/<string:version>/training_data', methods=['GET'])
def get_model_training_data(model_name, version):
    try:
        parameters, training_data = storage.get_training_data(model_name, version)
        return jsonify({'parameters': parameters, 'training_data': training_data})
    except FileNotFoundError as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404

@app.route('/models/json', methods=['POST'])
def upload_model_json():
    try:
        data = request.json
        version = data.get('version', 'latest')
        model_name = data['name']
        parameters = data.get('parameters', {})
        results = data.get('training_results', {})
        training_data = data.get('training_data', [])
        model_bytes = base64.b64decode(data['model'])
        model = utils.read_bytes(model_bytes)
        storage.save_model(model, model_name, version, parameters, results)
        storage.save_training_data(model_name, version, training_data)

        model_key = f"{model_name}_{version}"

        models[model_key] = {
            "model": base64.b64encode(model_bytes).decode('utf-8'),
            "parameters": parameters
            }

        return jsonify({"message": "Model uploaded successfully"})
    except Exception as e:
#         raise e
        print(f'ERROR {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.route('/models/file', methods=['POST'])
def upload_model_file():
    try:
        # Assume the model is sent in the request payload
        model_file = request.files['model']
        model = joblib.load(model_file)

        model_name = request.form['name']
        version = request.form.get('version', 'latest')
        parameters = request.form.get('parameters', {})

        storage.save_model(model, model_name, version, parameters)
        model_bytes, _ = storage.read(model_name, version)

        model_key = f"{model_name}_{version}"
        models[model_key] = {
            "model": base64.b64encode(model_bytes).decode('utf-8'),
            "parameters": parameters
            }

        return jsonify({"message": "Model uploaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify(storage.list())

@app.route('/models/<string:model_name>/best', methods=['GET'])
def best_model(model_name):
    return jsonify(storage.get_best_model(model_name))


if __name__ == '__main__':
    print("starting registry service...")
    app.run(debug=True,host='0.0.0.0',port=5000)

