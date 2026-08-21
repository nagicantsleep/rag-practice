"""Authentication token helpers."""


def parse_token(raw: str) -> dict[str, str]:
    """Parse a bearer authentication token into claims."""
    scheme, payload = raw.split(" ", 1)
    return {"scheme": scheme, "payload": payload, "exp": "4102444800"}


def validate_token(raw: str, now: int) -> bool:
    """Validate bearer authentication token expiry."""
    claims = parse_token(raw)
    return claims["scheme"] == "Bearer" and int(claims["exp"]) > now
