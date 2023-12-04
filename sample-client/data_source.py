import requests
from datetime import datetime, timedelta

class QuotesSource:

    def __init__(self, base_url = "http://localhost:5000/yfinance"):
        self.base_url = base_url

    def get_data(self, symbol, days, until = None):
        if until is None:
            until_isodate = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            until_isodate = until.strftime('%Y-%m-%dT%H:%M:%SZ')

        url = f'{self.base_url}/{symbol}/until/{until_isodate}?days={days}'
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        raise Exception(f'{response.status_code} returned for GET {url}')


