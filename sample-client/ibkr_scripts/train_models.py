import requests
import json
from datetime import datetime, timedelta
from ib4j_client import Ib4jClient

ib4j = Ib4jClient()

def get_data(symbol, interval, days, since = None):
    return ib4j.get_data(symbol, interval, days, since)

def train_model(symbol, interval, model_name, model_version, parameters, days):
    url = f"http://ibconnect.cyberdyne:5000/train/{model_name}"
    since = datetime.utcnow() - timedelta(days=days)
    payload = {
        'version': model_version,
        'data': get_data(symbol, interval, days, since),
        'training_data': get_data(symbol, interval, days),
        'parameters': parameters
    }
    headers = {'Content-Type': 'Application/json'}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    raise Exception(f'Error training model {response.status_code} {response.text}')

def make_models(symbol, model_version, target, features, rules = {}, days=30):
    make_model(symbol, '1_hour', model_version, target, features, rules, 'hour', 14)
    make_model(symbol, '1_day', model_version, target, features, rules, 'day', days)
    make_model(symbol, '1_week', model_version, target, features, rules, 'week', 300)

def make_model(symbol, interval, model_version, target, features, rules = {}, period = 'day', days = 30):
    train_features = sorted(features.copy())
    if target in train_features:
        train_features.remove(target)
    parameters = {
        'target_label': target,
        'feature_labels': train_features,
        'rules': rules,
    }
    train_model(symbol, interval, f"{symbol}_{target}", f"{model_version}_{period}", parameters, days)
    print(f"Model Version: {symbol}_{target}_{model_version}_{period}")


if __name__ == "__main__":
    symbols = ['DELL', 'AAPL', 'QQQ', 'LOW', 'TGT', 'WMT', 'INTC', 'AMD', 'MSFT', 'HMC', 'STLA']
    model_version = 'v9'
    training_days = 90
    v4_features = ['last_close', 'last_open', 'last_high', 'price_max', 'last_low', 'price_min', 'price_sentiment', 'percentile_close', 'percentile_high', 'percentile_low']
    v5_features = ['last_close', 'last_open', 'price_max', 'price_min', 'price_sentiment', 'percentile_close', 'percentile_high', 'percentile_low']
    features = ['last_close', 'last_open', 'percentile_high', 'percentile_close', 'price_sentiment']
    rules = {
#         'extract': {
#             'last_timestamp': {
               # the following would produce new features in the dataset for the extracted elements
               # names of the new features will be the concatenation of the source feature name and the element name
#                 'type': 'datetime',
#                 'elements': ['month', 'year', 'day_of_week']
#             }
#         },
        'flatten': {
            'history_change': 'sum',
            'history_high': 'max',
            'history_low': 'min',
            'history_volume': 'avg',
            'history_trades': 'avg',
        },
#         'categorize': {
#             'last_change': 'onehot'
#         }
    }

    for symbol in symbols:
#         for feature in features:
        make_models(symbol, model_version, 'last_close', features, rules, training_days)
        make_models(symbol, model_version, 'last_open', features, rules, training_days)

        make_models(symbol, model_version, 'price_avg', features, rules, training_days)
        make_models(symbol, model_version, 'price_min', features, rules, training_days)
        make_models(symbol, model_version, 'price_max', features, rules, training_days)
        make_models(symbol, model_version, 'last_high', features, rules, training_days)
        make_models(symbol, model_version, 'last_low', features, rules, training_days)
        make_models(symbol, model_version, 'last_volume', ['price_avg', 'price_max', 'price_min', 'last_high', 'last_low'], rules, training_days)
        make_models(symbol, model_version, 'last_trades', ['price_avg', 'price_max', 'price_min', 'last_high', 'last_low'], rules, training_days)

        make_models(symbol, model_version, 'price_sentiment', ['last_open', 'last_close'], rules, training_days)
        make_models(symbol, model_version, 'percentile_low', ['price_avg', 'price_max', 'price_min', 'last_high', 'last_low'], rules, training_days)
        make_models(symbol, model_version, 'percentile_high', ['price_avg', 'price_max', 'price_min', 'last_high', 'last_low'], rules, training_days)
        make_models(symbol, model_version, 'percentile_close', ['price_avg', 'price_max', 'price_min', 'last_high', 'last_low'], rules, training_days)
