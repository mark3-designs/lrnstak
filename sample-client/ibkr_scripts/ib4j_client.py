import requests
from datetime import datetime, timedelta
from cache import NamespacedCache

class Ib4jClient:

    # Define a constant for the default TTL (10 days)
    DEFAULT_TTL = 10 * 24 * 60 * 60

    def __init__(self, base_url = "http://ibconnect.cyberdyne:8080"):
        self.base_url = base_url
        self.cache = NamespacedCache('quotes')

    def calculate_ttl(self, interval, since = None):
        # Calculate TTL based on interval and since
        if since is not None:
            return self.DEFAULT_TTL  # Use default TTL if there is a 'since' date

        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day, 7, 0, 0)
        today_end = datetime(now.year, now.month, now.day, 18, 0, 0)

        if now.weekday() < 5 and today_start <= now <= today_end:
            # Weekday between 7:00 AM and 6:00 PM
            return 600  # 10 mins
        else:
            # Not within the specified times, set TTL until tomorrow morning
            tomorrow_start = today_start + timedelta(days=1)
            time_until_tomorrow = tomorrow_start - now
            return int(time_until_tomorrow.total_seconds())

    def get_data(self, symbol, interval, days, since = None):
        clear_cache = False
        cache_key = f'{interval}_{symbol}_{days}days'
        since_isodate = None
        ttl = self.calculate_ttl(interval, since)
        if since is None:
            since_isodate = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            since_isodate = since.strftime('%Y-%m-%dT%H:%M:%SZ')
            cache_key = f'{cache_key}.{since.strftime("%Y-%m-%d")}'

        url = f'{self.base_url}/f/{interval}/{symbol}?days={days}&nocache=true&date={since_isodate}'
        fetch_data = lambda key: {"value": requests.get(url).json()}
        return self.cache.compute_if_absent(cache_key, ttl, fetch_data, clear_cache)

