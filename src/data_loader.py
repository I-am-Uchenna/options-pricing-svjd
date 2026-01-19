import numpy as np
import pandas as pd
from scipy.stats import norm


def create_market_data(spot: float = 232.90, r: float = 0.015) -> pd.DataFrame:
    """
    Create realistic option market data using Black-Scholes with smile.

    This simulates what we would get from real market data.
    """
    np.random.seed(42)

    data = []

    maturities_days = [15, 30, 60, 90]

    for mat_days in maturities_days:
        T = mat_days / 250

        moneyness = np.linspace(0.85, 1.15, 13)
        strikes = spot * moneyness

        atm_vol = 0.25 + 0.02 * np.sqrt(T)
        skew = -0.10 - 0.05 * T
        smile = 0.08

        for K in strikes:
            m = K / spot

            impl_vol = atm_vol + skew * (m - 1) + smile * (m - 1) ** 2
            impl_vol = max(impl_vol, 0.10)

            d1 = (np.log(spot / K) + (r + 0.5 * impl_vol**2) * T) / (
                impl_vol * np.sqrt(T)
            )
            d2 = d1 - impl_vol * np.sqrt(T)

            call_price = spot * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

            call_price *= 1 + np.random.uniform(-0.005, 0.005)
            put_price *= 1 + np.random.uniform(-0.005, 0.005)

            data.append(
                {
                    "strike": K,
                    "maturity_days": mat_days,
                    "maturity_years": T,
                    "price": max(call_price, 0.01),
                    "type": "call",
                    "moneyness": m,
                    "impl_vol": impl_vol,
                }
            )

            data.append(
                {
                    "strike": K,
                    "maturity_days": mat_days,
                    "maturity_years": T,
                    "price": max(put_price, 0.01),
                    "type": "put",
                    "moneyness": m,
                    "impl_vol": impl_vol,
                }
            )

    return pd.DataFrame(data)
