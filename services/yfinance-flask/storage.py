import requests

class Cache:
    def __init__(self, base_url = "http://cache:5000/cache"):
        self.base_url = base_url

    def set_key(self, key, value, ttl=0):
        url = f"{self.base_url}/{key}"
        data = {"value": value, "ttl": ttl}
        print(f"PUT {key} in cache ttl={ttl}")
        response = requests.put(url, json=data)
        if response.status_code == 200:
            return response.json()
        raise Exception(f'{url} {response.status_code} {response.text}')

    def delete(self, key):
        url = f"{self.base_url}/{key}"
        response = requests.delete(url)
        if response.status_code == 200:
            return response.json()
        raise Exception(f'{url} {response.status_code} {response.text}')

    def put(self, key, value, ttl=0):
        return self.set_key(key, value, ttl)

    def get(self, key):
        url = f"{self.base_url}/{key}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return None
        raise Exception(f'Error GET {response.status_code} {response.text}')

    def compute_if_absent(self, key, ttl, provider):
        url = f"{self.base_url}/{key}"
        if ttl == -1:
            requests.delete(url)
            data = provider(key)['value']
            self.set_key(key, data, 900)
            return data
        else:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                data = provider(key)['value']
                self.set_key(key, data, ttl)
                return data

        raise Exception(f'{response.status_code} {response.text}')

    def ttl(self, key, ttl):
        url = f"{self.base_url}/ttl/{key}"
        data = {"ttl": ttl}
        response = requests.put(url, json=data)
        if response.status_code == 200:
            return response.json()
        raise Exception(f'{response.status_code} {response.text}')


class NamespacedCache(Cache):
    def __init__(self, namespace, separator = '-', base_url = "http://cache:5000/cache"):
        super().__init__(base_url)
        self.namespace = namespace
        self.separator = separator

    def key_for(self, key):
        return f'{self.namespace}{self.separator}{key}'

    def put(self, key, value, ttl=0):
        ns_key = self.key_for(key)
        return super().set_key(ns_key, value, ttl)

    def get(self, key):
        ns_key = self.key_for(key)
        return super().get(ns_key)

    def ttl(self, key, ttl):
        ns_key = self.key_for(key)
        return super().ttl(ns_key, ttl)

    def compute_if_absent(self, key, ttl, provider, invalidate = False):
        ns_key = self.key_for(key)
        if invalidate:
            super().delete(ns_key)
        return super().compute_if_absent(ns_key, ttl, provider)
