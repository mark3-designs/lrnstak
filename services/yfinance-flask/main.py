from flask import Flask, request, jsonify
import yfinance as yf
import sys
import logging
import base64
from datetime import datetime, timedelta
from storage import NamespacedCache

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)

storage = NamespacedCache('yfinance', base_url='http://cache:5000/cache')

@app.route('/yfinance/<string:symbol>/until/<string:until>', methods=['GET'])
def get_quotes(symbol, until):
    days = int(request.args.get('days', 30))
    until_date = datetime.strptime(until, '%Y-%m-%dT%H:%M:%SZ')
    start_date = until_date - timedelta(days=days)
    timekey = until_date.strftime("%Y%m%d%H")

    key = f'{symbol}_{timekey}.{days}'

    found = storage.get(key)
    if found:
        return jsonify(found)

    try:
        data = yf.download(symbol, start=start_date, end=until_date)
        # Convert column names to lowercase with underscores
        data.columns = [col.lower().replace(' ', '_') for col in data.columns]
        records = data.to_dict(orient='records')

        now = datetime.utcnow()
        is_in_current_day = now.year == until_date.year and now.month == until_date.month and now.day == until_date.day
        if is_in_current_day:
            storage.put(key, records, ttl=900)
        else:
            storage.put(key, records, ttl=24*60*60)

        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("starting service...")
    app.run(debug=True,host='0.0.0.0',port=5000)

