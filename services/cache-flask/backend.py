import redis
import json
import base64

class RedisBackend:

    #def __init__(self, host='redis', port=6379, db=0):
    def __init__(self, host='10.6.88.8', port=4415, db=0):
        self.redis_client = redis.StrictRedis(host, port, db)

    def put(self, key, value, ttl=0):
        self.redis_client.set(key, value)
        if (ttl > 0):
            self.redis_client.expire(key, ttl)

    def set_ttl(self, key, ttl):
        self.redis_client.expire(key, ttl)

    def get(self, key):
        return self.redis_client.get(key)

    def exists(self, key):
        return self.redis_client.exists(key)

    def delete(self, key):
        return self.redis_client.delete(key)

    def push_left(self, key, value):
        return self.redis_client.lpush(key, value)

    def pop_first(self, key, n=1):
        val = self.redis_client.lpop(key)
        if val is not None:
            return val, True
        return None, False

    def pop_last(self, key, n=1):
        val = self.redis_client.rpop(key)
        if val is not None:
            return val, True
        return None, False

    def peek_list(self, key, pos=0):
        val = self.redis_client.lindex(key, pos)
        if val is not None:
            return val, True
        return None, False
