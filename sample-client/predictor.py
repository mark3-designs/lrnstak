import sys
import requests
import json
from data_source import QuotesSource

lrnstak_server = 'localhost:5000'
quotes = QuotesSource(f'http://{lrnstak_server}/yfinance')

def get_data(symbol, days, until = None):
    return quotes.get_data(symbol, days, until)

def predict_next_value(symbol, target, model_version, data):
    predict_url = f"http://{lrnstak_server}/predict/model/{symbol}_{target}"
    payload = {'version': f"{model_version}", 'data': data}
    response = requests.post(predict_url, json=payload)
    if response.status_code == 200:
        predictions = response.json()
        if isinstance(predictions, dict):
            raise Exception(f'{predictions}')
        return predictions
    raise Exception(f'ERROR {response.status_code} {response.text}')

def run_predictions(symbol, model_version, quotes, target_features = ['close']):

    def predict_target(target_feature):
        return predict_next_value(symbol, target_feature, f'{model_version}', quotes)

    # Predict the next values for all target features
    predictions = {feature: predict_target(feature) for feature in target_features}

    next_data = []

    for real_data, *prediction_data in zip(quotes, *predictions.values()):
        next_entry = {**real_data, **{f'next_{feature}': data['prediction'] for feature, data in zip(target_features, prediction_data)}}
        next_data.append(next_entry)

    return next_data



if __name__ == "__main__":
    symbols = ['DELL', 'AAPL', 'QQQ', 'LOW', 'TGT', 'WMT', 'INTC', 'AMD', 'MSFT', 'SPWR', 'HMC', 'STLA']
    symbols.extend(['JNPR', 'TME', 'AAL', 'F'])
    symbols.extend(['COKE', 'KO', 'OEC', 'ERO', 'GSM', 'HBM', 'BAK', 'SAND', 'AG', 'BAC', 'DHT'])
    symbols.extend(['PLUG', 'RUN', 'PYPL'])

    symbol = sys.argv[1] if len(sys.argv) > 1 else 'MSFT'
    model_version = sys.argv[2] if len(sys.argv) > 2 else 'default'

    features = ['close', 'high', 'low']

    # until = datetime.strptime('2023-08-12T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    until = None # will default to now

    training_days = 30
    quotes = get_data(symbol, training_days, until)
    next_predictions = run_predictions(symbol, model_version, quotes, features)
    print(json.dumps(next_predictions))




