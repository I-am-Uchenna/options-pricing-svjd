from dataclasses import dataclass


@dataclass
class ProjectConfig:
    """Project configuration parameters."""

    # Market data
    spot_price: float = 232.90
    risk_free_rate: float = 0.015
    trading_days_per_year: int = 250

    # Maturities
    SHORT_MATURITY_DAYS: int = 15
    MEDIUM_MATURITY_DAYS: int = 60
    ASIAN_MATURITY_DAYS: int = 20
    PUT_MATURITY_DAYS: int = 70

    # Options
    PUT_MONEYNESS: float = 0.95
    BANK_FEE: float = 0.04


config = ProjectConfig()
