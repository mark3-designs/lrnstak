import sys
import logging
from flask import Flask, request, jsonify
from lrnstak.predictions_module import Model
from lrnstak.registry_client import CachingRegistry

app = Flask(__name__)

# Configure Flask app logging
app.logger.setLevel(logging.DEBUG)

registry = CachingRegistry()

@app.route('/predict/model/<string:model_name>', methods=['POST'])
def predict_model(model_name):
    try:
        data = request.json['data']
        version = request.json.get('version', 'latest')

        model, parameters, status = registry.get_and_cache(model_name, version)

        if status == 200:
            print("Predicting...")
            app.logger.info(f"Prediction Running for {model_name}/{version}")
            prediction = Model(app.logger).evaluate(model, data, parameters)

            #prediction['symbol'] = data[0]['symbol']
            #prediction['symbol_exchange'] = data[0]['symbol_exchange']
            #prediction['timestamp'] = data[-1]['last_timestamp']
            return jsonify(prediction)
        else:
            return jsonify({'error': 'registry failure'}), status

    except Exception as e:
        app.logger.exception("ERROR", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/predict/<string:target_label>', methods=['POST'])
def predict(target_label):
    try:
        data = request.json

        prediction = Model(app.logger).predict(data, target_label)

        prediction['symbol'] = data[0]['symbol']
        prediction['symbol_exchange'] = data[0]['symbol_exchange']
        prediction['timestamp'] = data[-1]['last_timestamp']
        return jsonify(prediction)

    except Exception as e:
        app.logger.exception("ERROR", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("starting predictions service...")
    app.run(debug=True,host='0.0.0.0',port=5000)
