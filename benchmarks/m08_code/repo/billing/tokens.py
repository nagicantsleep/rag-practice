"""Billing token helpers."""


def parse_token(raw: str) -> dict[str, str]:
    """Parse an invoice billing token formatted as invoice:amount."""
    invoice_id, amount = raw.split(":", 1)
    return {"invoice_id": invoice_id, "amount": amount}


def token_amount(raw: str) -> int:
    """Return the integer amount encoded in a billing token."""
    return int(parse_token(raw)["amount"])
