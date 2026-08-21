"""Regional tax rules."""


def tax_for_region(region: str, subtotal: int) -> int:
    """Calculate regional tax for a checkout subtotal."""
    rates = {"us": 8, "eu": 20, "apac": 10}
    return subtotal * rates.get(region, 0) // 100
