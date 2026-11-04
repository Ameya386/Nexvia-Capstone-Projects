"""Step 6 - Redis cache for recommendation results. Falls back to an in-process dict if Redis is not running,
so the demo never breaks. `cache.backend` tells you which one is active."""
import json, time
from src import config as C


class Cache:
    def __init__(self):
        self._mem = {}
        self.backend = "memory"
        try:
            import redis
            self.r = redis.Redis(host=C.REDIS_HOST, port=C.REDIS_PORT, socket_connect_timeout=0.3,
                                 socket_timeout=0.3)
            self.r.ping()
            self.backend = "redis"
        except Exception:
            self.r = None

    def get(self, key):
        if self.r is not None:
            v = self.r.get(key)
            return json.loads(v) if v else None
        v = self._mem.get(key)
        if v and v[0] > time.time():
            return v[1]
        return None

    def set(self, key, value, ttl=C.CACHE_TTL):
        if self.r is not None:
            self.r.setex(key, ttl, json.dumps(value))
        else:
            self._mem[key] = (time.time() + ttl, value)
