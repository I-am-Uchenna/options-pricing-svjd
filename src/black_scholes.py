"""Black-Scholes-Merton analytics for European options.

The functions in this module are intentionally dependency-light and deterministic.
They provide transparent benchmark pricing, Greeks, and implied-volatility
inversion for research workflows and sanity checks.
"""

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class Greeks:
    """First- and second-order sensitivities under Black-Scholes-Merton."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""

    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _validate_inputs(
    spot: float, strike: float, maturity: float, volatility: float
) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity < 0:
        raise ValueError("maturity cannot be negative")
    if volatility < 0:
        raise ValueError("volatility cannot be negative")


def d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return the Black-Scholes d1 and d2 terms."""

    _validate_inputs(spot, strike, maturity, volatility)
    if maturity == 0 or volatility == 0:
        forward = spot * exp((rate - dividend_yield) * maturity)
        d1 = float("inf") if forward >= strike else float("-inf")
        return d1, d1

    sigma_sqrt_t = volatility * sqrt(maturity)
    d1 = (
        log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def black_scholes_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a European option under Black-Scholes-Merton."""

    _validate_inputs(spot, strike, maturity, volatility)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if maturity == 0:
        intrinsic = spot - strike if option_type == "call" else strike - spot
        return max(intrinsic, 0.0)

    if volatility == 0:
        forward = spot * exp((rate - dividend_yield) * maturity)
        discounted = exp(-rate * maturity)
        intrinsic = forward - strike if option_type == "call" else strike - forward
        return discounted * max(intrinsic, 0.0)

    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    discounted_spot = spot * exp(-dividend_yield * maturity)
    discounted_strike = strike * exp(-rate * maturity)

    if option_type == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)

    return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)


def black_scholes_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Return annualized Black-Scholes-Merton Greeks."""

    _validate_inputs(spot, strike, maturity, volatility)
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    if maturity == 0 or volatility == 0:
        if option_type == "call":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

    d1, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    pdf_d1 = normal_pdf(d1)
    exp_qt = exp(-dividend_yield * maturity)
    exp_rt = exp(-rate * maturity)

    gamma = exp_qt * pdf_d1 / (spot * volatility * sqrt(maturity))
    vega = spot * exp_qt * pdf_d1 * sqrt(maturity) / 100.0

    if option_type == "call":
        delta = exp_qt * normal_cdf(d1)
        theta = (
            -spot * exp_qt * pdf_d1 * volatility / (2.0 * sqrt(maturity))
            - rate * strike * exp_rt * normal_cdf(d2)
            + dividend_yield * spot * exp_qt * normal_cdf(d1)
        ) / 365.0
        rho = strike * maturity * exp_rt * normal_cdf(d2) / 100.0
    else:
        delta = exp_qt * (normal_cdf(d1) - 1.0)
        theta = (
            -spot * exp_qt * pdf_d1 * volatility / (2.0 * sqrt(maturity))
            + rate * strike * exp_rt * normal_cdf(-d2)
            - dividend_yield * spot * exp_qt * normal_cdf(-d1)
        ) / 365.0
        rho = -strike * maturity * exp_rt * normal_cdf(-d2) / 100.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_volatility(
    option_type: OptionType,
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float = 0.0,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float:
    """Invert Black-Scholes-Merton price to implied volatility by bisection."""

    if market_price <= 0:
        raise ValueError("market_price must be positive")

    low = lower
    high = upper
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        model_price = black_scholes_price(
            option_type, spot, strike, maturity, rate, mid, dividend_yield
        )
        error = model_price - market_price
        if abs(error) < tolerance:
            return mid
        if error > 0:
            high = mid
        else:
            low = mid

    return 0.5 * (low + high)


def probability_itm(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Risk-neutral probability that the option expires in the money."""

    _, d2 = d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    if option_type == "call":
        return normal_cdf(d2)
    if option_type == "put":
        return normal_cdf(-d2)
    raise ValueError("option_type must be 'call' or 'put'")
