"""Session service."""
from storage.cache import Cache


def clear_session(cache: Cache, key: str) -> None:
    """Clear a user session by invalidating its cache key."""
    cache.invalidate(key)
