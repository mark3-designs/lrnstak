from flask import Flask, request, jsonify
import json
import base64
import time
from datetime import datetime
from backend import RedisBackend

app = Flask(__name__)

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
            'service': 'lrnstak-cache',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime': time.time() - app.start_time
        }
        
        # Check Redis connectivity
        try:
            # Test Redis connectivity via store
            store.get('test_key')  # This will test the connection
            health_data['redis_status'] = 'connected'
        except:
            health_data['redis_status'] = 'disconnected'
            health_data['status'] = 'degraded'
        
        return health_data, 200
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500

store = RedisBackend('10.6.88.8', 4415, 0)

@app.route('/cache/<string:key>', methods=['PUT', 'POST'])
def put_kv(key):
    data = request.get_json()
    value = data.get('value')
    ttl = data.get('ttl', 0)

    if key and value:
        store.put(key, json.dumps(value), ttl)
        return jsonify({"message": "OK"})
    else:
        return jsonify({"error": "Key and Value are required"}), 400

@app.route('/cache/<string:key>', methods=['GET'])
def get_kv(key):
    value = store.get(key)
    if value:
        return jsonify(json.loads(value.decode('utf-8')))
    else:
        return jsonify({"message": "not found"}), 404

@app.route('/cache/push/<string:key>', methods=['PUT'])
def push(key):
    data = request.get_json()
    value = data.get('value')
    ttl = data.get('ttl', 0)

    if key and value:
        store.push_left(key, json.dumps(value), ttl)
        return jsonify({"message": "OK"})
    else:
        return jsonify({"error": "Key and Value are required"}), 400

@app.route('/cache/peek/<string:key>', methods=['GET'])
def peek(key):
    pos = int(request.args.get('pos', default=0))
    try:
        return jsonify(store.peek_list(key, pos))
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/cache/pop/<string:key>', methods=['GET'])
def pop(key):
    if key:
        value, found = store.pop_first(key)
        if found:
            return jsonify(value)
        return jsonify({"error": "not found"}), 404
    else:
        return jsonify({"error": "Key is required"}), 400

@app.route('/cache/ttl/<string:key>', methods=['PUT'])
def update_ttl(key):
    data = request.get_json()
    ttl = data.get('ttl', 0)

    if key:
        if store.exists(key):
            store.set_ttl(key, ttl)
            return jsonify({"message": "OK"})
        else:
            return jsonify({"error": f"Key {key} not found"}), 404
    else:
        return jsonify({"error": "Key is required"}), 400

@app.route('/cache/<string:key>', methods=['DELETE'])
def delete_data(key):

    if key is None:
        return jsonify({"error": "Key required"}), 400

    if store.exists(key):
        deleted = store.delete(key)
        return jsonify({"message": "OK", "value": deleted})

    return jsonify({"message": "OK", "value": None})

if __name__ == '__main__':
    print("starting cache api service...")
    app.run(debug=True,host='0.0.0.0',port=5000)

