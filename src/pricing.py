from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .models import BatesModel, HestonModel


@dataclass
class AsianPricingResult:
    """Asian option pricing result."""

    price: float
    std_error: float
    ci_lower: float
    ci_upper: float
    num_simulations: int
    exercise_probability: float


class AsianOptionPricer:
    """Monte Carlo pricer for Asian options."""

    def __init__(self, model: Union[HestonModel, BatesModel], r: float):
        self.model = model
        self.r = r
        self.S0 = model.S0

    def price_arithmetic_call(
        self,
        K: float,
        T: float,
        num_simulations: int = 100000,
        num_steps: Optional[int] = None,
        include_S0: bool = True,
        seed: Optional[int] = None,
    ) -> AsianPricingResult:
        """
        Price arithmetic average Asian call option.

        Payoff: max(A - K, 0) where A is arithmetic average.
        """
        if num_steps is None:
            num_steps = max(int(T * 250), 5)

        paths, _ = self.model.simulate_paths(T, num_steps, num_simulations, seed=seed)

        if include_S0:
            averages = np.mean(paths, axis=1)
        else:
            averages = np.mean(paths[:, 1:], axis=1)

        payoffs = np.maximum(averages - K, 0)

        discount = np.exp(-self.r * T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)

        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error

        exercise_prob = np.mean(payoffs > 0)

        return AsianPricingResult(
            price=price,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            num_simulations=num_simulations,
            exercise_probability=exercise_prob,
        )

    def price_arithmetic_put(
        self,
        K: float,
        T: float,
        num_simulations: int = 100000,
        num_steps: Optional[int] = None,
        include_S0: bool = True,
        seed: Optional[int] = None,
    ) -> AsianPricingResult:
        """Price arithmetic average Asian put option."""
        if num_steps is None:
            num_steps = max(int(T * 250), 5)

        paths, _ = self.model.simulate_paths(T, num_steps, num_simulations, seed=seed)

        if include_S0:
            averages = np.mean(paths, axis=1)
        else:
            averages = np.mean(paths[:, 1:], axis=1)

        payoffs = np.maximum(K - averages, 0)

        discount = np.exp(-self.r * T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)

        return AsianPricingResult(
            price=price,
            std_error=std_error,
            ci_lower=price - 1.96 * std_error,
            ci_upper=price + 1.96 * std_error,
            num_simulations=num_simulations,
            exercise_probability=np.mean(payoffs > 0),
        )
