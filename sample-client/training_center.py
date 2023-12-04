import requests
from datetime import datetime, timedelta
import concurrent.futures
from data_source import QuotesSource

lrnstak_server = 'localhost:5000'
quotes = QuotesSource(f'http://{lrnstak_server}/yfinance')

def train_model(symbol, model_name, model_version, parameters, days):
    url = f"http://{lrnstak_server}/train/{model_name}"
    until = datetime.utcnow() - timedelta(days=30)

    meta = parameters.get('metadata', {})
    meta['training_until'] = until.strftime('%Y-%m-%dT%H:%M:%SZ')
    meta['training_days'] = days
    parameters['metadata'] = meta

    payload = {
        'version': model_version,
        'training_data': quotes.get_data(symbol, min(120, days * 2), until),
        'testing_data': quotes.get_data(symbol, days),
        'parameters': parameters
    }

    # print(f"POST {payload}")
    headers = {'Content-Type': 'Application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    raise Exception(f'Error training model {response.status_code} {response.text}')

def make_models(symbol, engine, model_version, target, features, rules = {}, days=30):
    try:
        train_features = sorted(features.copy())
        if target in train_features:
            train_features.remove(target)
        parameters = {
            'engine': engine,
            'target_label': target,
            'feature_labels': train_features,
            'rules': rules,
            'metadata': {},
        }
        train_model(symbol, f"{symbol}_{target}", f"{model_version}", parameters, days)
        print(f"Model Trained: {symbol}_{target}_{model_version}")
    except Exception as e:
        print(f'Error: {str(e)}')

def predict_next_value(symbol, target, model_version, data):
    predict_url = f"http://{lrnstak_server}/predict/model/{symbol}_{target}"
    print(f'Running predictions for {symbol}_{target} version {model_version}')
    response = requests.post(predict_url, json={'version': f"{model_version}", 'data': data})
    if response.status_code == 200:
        predictions = response.json()
        return predictions
    raise Exception(f'ERROR {response.status_code} POST {predict_url} [data]')

if __name__ == "__main__":

    symbols = ['DELL', 'AAPL', 'QQQ', 'LOW', 'TGT', 'WMT', 'INTC', 'AMD', 'MSFT', 'HMC', 'STLA']

    # symbols = ['AAPL']
    # symbols = ['MSFT']

    training_days = 60

    model_definitions = []

    model_definitions.append({
        'engine': 'default',
        'version': 'default',
        'target': 'close',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'default',
        'version': 'default',
        'target': 'high',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'default',
        'version': 'default',
        'target': 'low',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })

    model_definitions.append({
        'engine': 'tensorflow',
        'version': 'v1',
        'target': 'close',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'tensorflow',
        'version': 'v1',
        'target': 'high',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'tensorflow',
        'version': 'v1',
        'target': 'low',
        'features': ['open', 'high', 'low', 'volume'],
        'rules': { }
    })

    model_definitions.append({
        'engine': 'default',
        'version': 'v2',
        'target': 'close',
        'features': ['high', 'low'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'default',
        'version': 'v2',
        'target': 'high',
        'features': ['high', 'low'],
        'rules': { }
    })
    model_definitions.append({
        'engine': 'default',
        'version': 'v2',
        'target': 'low',
        'features': ['high', 'low'],
        'rules': { }
    })

    def process(model_definition, symbol):
        print(symbol)
        engine = model_definition.get('engine', 'default')
        version = model_definition.get('version', 'default')
        make_models(symbol, engine, f'{version}', model_definition['target'], model_definition['features'], model_definition['rules'], training_days)


    concurrent_mode = True

    if not concurrent_mode:
        for model in model_definitions:
            for symbol in symbols:
                process(model, symbol)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process, model, symbol) for model in model_definitions for symbol in symbols]
            concurrent.futures.wait(futures)