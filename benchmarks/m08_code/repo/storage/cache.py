"""Small cache abstraction."""


class Cache:
    """In-memory cache used by sessions."""

    def get(self, key: str) -> str | None:
        """Read one cache entry."""
        return None

    def invalidate(self, key: str) -> None:
        """Invalidate one cached session entry."""
        return None
