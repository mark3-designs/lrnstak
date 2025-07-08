from flask import Flask, request, jsonify
import yfinance as yf
import sys
import logging
import base64
import threading
import time
import requests
from datetime import datetime, timedelta
from storage import NamespacedCache

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
            'service': 'lrnstak-yfinance',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.start_time
        }
        
        # Check cache connectivity
        try:
            # Test cache connectivity via storage
            response = requests.get('http://cache:5000/health', timeout=2)
            health_data['cache_status'] = 'connected' if response.status_code == 200 else 'disconnected'
        except:
            health_data['cache_status'] = 'disconnected'
            health_data['status'] = 'degraded'
        
        return health_data, 200
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500

storage = NamespacedCache('yfinance', base_url='http://cache:5000/cache')

yfinance_lock = threading.Lock()

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
        with yfinance_lock:
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

