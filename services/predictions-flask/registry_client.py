import requests
import base64
import json
import os
import tempfile
import joblib
import time

class Registry:
    def __init__(self, base_url = "http://registry:5000/models"):
        self.base_url = base_url

    def __read_bytes(self, bytes):
        tmp = tempfile.NamedTemporaryFile().name
        with open(tmp, 'wb') as file:
            file.write(bytes)
        model = joblib.load(tmp)
        os.remove(tmp)
        return model

    def get(self, model_name, version):
        registry_url = f'{self.base_url}/{model_name}/{version}'

        response = requests.get(registry_url)
        if response.status_code == 200:
            data = response.json()

            model_bytes = base64.b64decode(data['model'])
            parameters = data.get('parameters', {})

            return self.__read_bytes(model_bytes), parameters, 200
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None, None, response.status_code

            from cache import Cache  # Assuming you have a Cache class

class CachingRegistry(Registry):
    def __init__(self, base_url="http://registry:5000/models"):
        super().__init__(base_url)
        self.cache = {}

    def __is_expired(self, cached_entry):
        expires = cached_entry.get('expires', 0)
        if expires == 0:
            return False
        return time.time() > expires

    def get_and_cache(self, model_name, version):
        # Check if the response is already in the cache
        found = self.cache.get(f"{model_name}_{version}")
        if found and not self.__is_expired(found):
            return found['data']

        # If not in the cache, retrieve from the parent class (Registry)
        model, parameters, status_code = self.get(model_name, version)

        # Cache the response for future use
        if status_code == 200:
            ttl = 600
            self.cache[f'{model_name}_{version}'] = {
                'data': (model, parameters, status_code),
                'expires': time.time() + ttl
                }
            return model, parameters, status_code

        raise Exception(f"Error: {status_code}")
