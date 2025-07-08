import requests
import json
from datetime import datetime, timedelta
from cache import NamespacedCache

from ib4j_client import Ib4jClient

ib4j = Ib4jClient()

def get_data(symbol, interval, days, since = None):
    return ib4j.get_data(symbol, interval, days, since)

#    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
#     now='2023-11-01T00:00:00Z'
#    url = f"http://ibconnect.cyberdyne:8080/f/{interval}/{symbol}?days={days}&nocache=false&date={now}"
#    response = requests.get(url)
#    if response.status_code == 200:
#        return response.json()
#    raise Exception(f'error {response.status_code} from data source.')

def predict_next_value(symbol, target, model_version, data):
    predict_url = f"http://ibconnect.cyberdyne:5000/predict/model/{symbol}_{target}"
    print(f'Running predictions for {symbol}_{target} version {model_version}')
    payload = {'version': f"{model_version}", 'data': data}
    response = requests.post(predict_url, json=payload)
    if response.status_code == 200:
        predictions = response.json()
        if isinstance(predictions, dict):
            raise Exception(f'{predictions}')
        return predictions
#     print(f'ERROR POST {predict_url} version={model_version} {response} {len(payload["data"])} {json.dumps(payload["data"][-1], indent=2)}')
    raise Exception(f'ERROR {response.status_code} {response.text}')

def next_timestamp(timestamp, period):
    timestamp_format = "%Y-%m-%dT%H:%M:%SZ"
    dt = datetime.strptime(timestamp, timestamp_format)

    time_difference = None
    if period == 'hour':
        time_difference = timedelta(hours=1)
    if period == 'day':
        time_difference = timedelta(days=1)
    if period == 'week':
        time_difference = timedelta(weeks=1)
    if period == 'month':
        time_difference = timedelta(days=31)

    new_timestamp = dt + time_difference
    new_timestamp_str = new_timestamp.strftime(timestamp_format)

    return new_timestamp.timestamp(), new_timestamp_str



def run_predictions(symbol, model_version, period, quotes, target_features = ['last_close']):

    def predict_target(target_feature):
        return predict_next_value(symbol, target_feature, f'{model_version}_{period}', quotes)

    # Predict the next values for all target features
    predictions = {feature: predict_target(feature) for feature in target_features}
    # print(json.dumps(predictions, indent=2))

    # Iterate through real data and add predicted values
    previous = None
    next_data = []

    for real_data, *prediction_data in zip(quotes, *predictions.values()):
#         next_entry = {**real_data, **{f'{feature}': data['prediction'] for feature, data in zip(target_features, prediction_data)}}
        #print(json.dumps(real_data, indent=2))
        #print(json.dumps(prediction_data, indent=2))
        next_entry = {**real_data, **{f'next_{feature.replace("last_", "")}': data['prediction'] for feature, data in zip(target_features, prediction_data)}}
        next_data.append(next_entry)
        previous = next_entry

        #uxtime, ts = next_timestamp(next_entry['last_timestamp'], period)
        #next_entry['last_uxtime'] = uxtime
        #next_entry['last_timestamp'] = ts

    return next_data


if __name__ == "__main__":
    symbols = ['DELL', 'AAPL', 'QQQ', 'LOW', 'TGT', 'WMT', 'INTC', 'AMD', 'MSFT', 'HMC', 'STLA']
    model_version = 'v9'

    period = 'day'
    training_days = 30
    v4_features = ['last_close', 'last_open', 'last_high', 'price_max', 'last_low', 'price_min', 'price_sentiment', 'percentile_close', 'percentile_high', 'percentile_low']
    v5_features = ['last_close', 'last_open', 'price_max', 'price_min', 'price_sentiment', 'percentile_close', 'percentile_high', 'percentile_low']
    features = ['last_close', 'last_open', 'price_min', 'price_sentiment']

    since = datetime.strptime('2023-08-12T00:00:00Z', '%Y-%m-%dT%H:%M:%SZ')
    since = None # will default to now
    for symbol in symbols:

        quotes = get_data(symbol, f'1_{period}', training_days, since)

        next = run_predictions(symbol, model_version, period, quotes, features)
        quotes.append(next[-1])
#         print(next[-1])
        NamespacedCache('predicted').put(f'{symbol}', { symbol: next })

        for _ in range(30):
            next = run_predictions(symbol, model_version, period, next, features)
            quotes.append(next[-1])

        #  4. predict the next set of values, append the prediction and run prediction model again
        #make_model(symbol, model_version, 'last_close', ['last_open', 'last_volume', 'last_trades', 'percentile_close', 'percentile_high', 'percentile_low'])

        # print(json.dumps(next_data, indent=2))

        # pred_next_last_close = predict_next_value(symbol, 'last_close', f'{model_version}_week', next_data)
        # print(json.dumps(pred_next_last_close, indent=2))



