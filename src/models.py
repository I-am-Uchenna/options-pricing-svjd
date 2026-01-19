from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class HestonParameters:
    """
    Heston model parameters.

    Attributes:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of variance
        rho: Correlation between spot and variance
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HestonParameters":
        return cls(v0=arr[0], kappa=arr[1], theta=arr[2], sigma=arr[3], rho=arr[4])

    def feller_ratio(self) -> float:
        """Feller condition ratio (should be > 1)."""
        return (2 * self.kappa * self.theta) / (self.sigma ** 2 + 1e-10)

    def is_valid(self) -> bool:
        """Check if parameters satisfy constraints."""
        return (
            self.v0 > 0
            and self.kappa > 0
            and self.theta > 0
            and self.sigma > 0
            and -1 < self.rho < 1
        )


class HestonModel:
    """
    Heston (1993) stochastic volatility model.

    dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_1
    dv_t = kappa*(theta - v_t)*dt + sigma*sqrt(v_t)*dW_2
    corr(dW_1, dW_2) = rho
    """

    def __init__(self, params: HestonParameters, r: float, S0: float):
        self.params = params
        self.r = r
        self.S0 = S0

    def characteristic_function(self, u: complex, T: float) -> complex:
        """
        Compute characteristic function using the formulation from
        Albrecher et al. (2007) which is more numerically stable.
        """
        v0 = self.params.v0
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho

        i = complex(0, 1)

        sigma2 = sigma * sigma

        tmp = kappa - i * rho * sigma * u
        d = np.sqrt(tmp * tmp + sigma2 * u * (u + i))

        g1 = tmp + d
        g2 = tmp - d
        g = g2 / (g1 + 1e-10)

        exp_dT = np.exp(-d * T)

        D = (g2 / sigma2) * (1 - exp_dT) / (1 - g * exp_dT + 1e-10)
        C = kappa * theta / sigma2 * (
            g2 * T - 2 * np.log((1 - g * exp_dT) / (1 - g + 1e-10) + 1e-10)
        )

        return np.exp(C + D * v0 + i * u * np.log(self.S0) + i * u * self.r * T)

    def _integrand_call(self, u: float, K: float, T: float) -> float:
        """Integrand for call option pricing."""
        try:
            i = complex(0, 1)
            log_K = np.log(K)

            cf = self.characteristic_function(u - 0.5 * i, T)
            integrand = np.real(np.exp(-i * u * log_K) * cf / (u * u + 0.25))

            if np.isnan(integrand) or np.isinf(integrand):
                return 0.0
            return integrand
        except Exception:
            return 0.0

    def price_call_lewis(self, K: float, T: float) -> float:
        """
        Price European call using Lewis (2001) formula.

        This is more stable than the original Heston formula.
        """
        from scipy.integrate import quad

        try:
            discount = np.exp(-self.r * T)

            integral, _ = quad(
                lambda u: self._integrand_call(u, K, T),
                0,
                100,
                limit=200,
                epsabs=1e-8,
                epsrel=1e-8,
            )

            price = self.S0 - discount * np.sqrt(self.S0 * K) / np.pi * integral

            return max(price, 0.0)
        except Exception:
            return max(self.S0 - K * np.exp(-self.r * T), 0.0)

    def price_put_lewis(self, K: float, T: float) -> float:
        """Price European put using put-call parity."""
        call = self.price_call_lewis(K, T)
        put = call - self.S0 + K * np.exp(-self.r * T)
        return max(put, 0.0)

    def price_call_carr_madan(
        self, K: float, T: float, N: int = 4096, alpha: float = 1.5, eta: float = 0.25
    ) -> float:
        """
        Price European call using Carr-Madan (1999) FFT method.
        """
        try:
            lambda_val = 2 * np.pi / (N * eta)
            b = N * lambda_val / 2

            ku = -b + lambda_val * np.arange(N)
            v = eta * np.arange(N)

            discount = np.exp(-self.r * T)

            i = complex(0, 1)

            psi = np.zeros(N, dtype=complex)

            for j in range(N):
                vj = v[j]
                if vj == 0:
                    vj = 1e-10

                u_arg = vj - (alpha + 1) * i

                try:
                    cf = self.characteristic_function(u_arg, T)
                    denominator = alpha**2 + alpha - vj**2 + i * vj * (2 * alpha + 1)

                    if abs(denominator) > 1e-10:
                        psi[j] = discount * cf / denominator
                    else:
                        psi[j] = 0.0
                except Exception:
                    psi[j] = 0.0

            simpson = 3 + (-1) ** (np.arange(N) + 1)
            simpson[0] = 1
            simpson = simpson / 3

            x = np.exp(i * b * v) * psi * eta * simpson

            fft_result = np.fft.fft(x)

            call_prices = np.real(np.exp(-alpha * ku) / np.pi * fft_result)

            log_K = np.log(K)

            idx = np.searchsorted(ku, log_K)

            if idx <= 0:
                price = call_prices[0]
            elif idx >= N:
                price = call_prices[-1]
            else:
                w = (log_K - ku[idx - 1]) / (ku[idx] - ku[idx - 1])
                price = (1 - w) * call_prices[idx - 1] + w * call_prices[idx]

            return max(price, 0.0)

        except Exception:
            return self.price_call_lewis(K, T)

    def price_put_carr_madan(self, K: float, T: float) -> float:
        """Price European put using Carr-Madan and put-call parity."""
        call = self.price_call_carr_madan(K, T)
        put = call - self.S0 + K * np.exp(-self.r * T)
        return max(put, 0.0)

    def simulate_paths(
        self, T: float, num_steps: int, num_paths: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate paths using Euler-Maruyama with full truncation.
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.params.v0

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho

        for t in range(num_steps):
            Z1 = np.random.standard_normal(num_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(num_paths)

            v_plus = np.maximum(v[:, t], 0)
            sqrt_v = np.sqrt(v_plus)

            v[:, t + 1] = v[:, t] + kappa * (theta - v_plus) * dt + sigma * sqrt_v * sqrt_dt * Z2
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)

            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1
            )

        return S, v


@dataclass
class BatesParameters:
    """
    Bates (1996) model parameters (Heston + jumps).

    Attributes:
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of variance
        rho: Correlation
        lambda_j: Jump intensity (jumps per year)
        mu_j: Mean jump size (log)
        sigma_j: Jump size volatility
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    lambda_j: float
    mu_j: float
    sigma_j: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.v0,
                self.kappa,
                self.theta,
                self.sigma,
                self.rho,
                self.lambda_j,
                self.mu_j,
                self.sigma_j,
            ]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "BatesParameters":
        return cls(
            v0=arr[0],
            kappa=arr[1],
            theta=arr[2],
            sigma=arr[3],
            rho=arr[4],
            lambda_j=arr[5],
            mu_j=arr[6],
            sigma_j=arr[7],
        )

    def feller_ratio(self) -> float:
        return (2 * self.kappa * self.theta) / (self.sigma ** 2 + 1e-10)

    def jump_compensator(self) -> float:
        """E[e^J - 1] for drift adjustment."""
        return np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1


class BatesModel:
    """
    Bates (1996) stochastic volatility model with jumps.

    dS_t/S_t = (r - lambda*k)*dt + sqrt(v_t)*dW_1 + (e^J - 1)*dN_t
    dv_t = kappa*(theta - v_t)*dt + sigma*sqrt(v_t)*dW_2
    """

    def __init__(self, params: BatesParameters, r: float, S0: float):
        self.params = params
        self.r = r
        self.S0 = S0

    def characteristic_function(self, u: complex, T: float) -> complex:
        """
        Characteristic function for Bates model.
        Combines Heston CF with jump component.
        """
        v0 = self.params.v0
        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        lambda_j = self.params.lambda_j
        mu_j = self.params.mu_j
        sigma_j = self.params.sigma_j

        i = complex(0, 1)

        sigma2 = sigma * sigma
        tmp = kappa - i * rho * sigma * u
        d = np.sqrt(tmp * tmp + sigma2 * u * (u + i))

        g2 = tmp - d
        g1 = tmp + d
        g = g2 / (g1 + 1e-10)

        exp_dT = np.exp(-d * T)

        D = (g2 / sigma2) * (1 - exp_dT) / (1 - g * exp_dT + 1e-10)
        C = kappa * theta / sigma2 * (
            g2 * T - 2 * np.log((1 - g * exp_dT) / (1 - g + 1e-10) + 1e-10)
        )

        heston_cf = np.exp(C + D * v0)

        jump_cf_single = np.exp(i * u * mu_j - 0.5 * u * u * sigma_j * sigma_j)

        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        jump_contribution = np.exp(lambda_j * T * (jump_cf_single - 1 - i * u * k))

        forward_adj = np.exp(i * u * (np.log(self.S0) + self.r * T))

        return heston_cf * jump_contribution * forward_adj

    def _integrand_call(self, u: float, K: float, T: float) -> float:
        """Integrand for call option pricing."""
        try:
            i = complex(0, 1)
            log_K = np.log(K)

            cf = self.characteristic_function(u - 0.5 * i, T)
            cf = cf / np.exp((0.5 * i + i * u) * (np.log(self.S0) + self.r * T))
            cf = cf * np.exp(i * u * np.log(self.S0) + i * u * self.r * T)

            integrand = np.real(np.exp(-i * u * log_K) * cf / (u * u + 0.25))

            if np.isnan(integrand) or np.isinf(integrand):
                return 0.0
            return integrand
        except Exception:
            return 0.0

    def price_call_lewis(self, K: float, T: float) -> float:
        """Price European call using Lewis (2001) approach."""
        from scipy.integrate import quad

        try:
            i = complex(0, 1)
            discount = np.exp(-self.r * T)
            log_K = np.log(K)

            def integrand(u):
                try:
                    z = u - 0.5 * i
                    cf = self.characteristic_function(z, T)
                    cf = cf / np.exp(i * z * (np.log(self.S0) + self.r * T))
                    result = np.real(np.exp(-i * u * log_K) * cf / (u * u + 0.25))
                    if np.isnan(result) or np.isinf(result):
                        return 0.0
                    return result
                except Exception:
                    return 0.0

            integral, _ = quad(integrand, 0, 100, limit=200)

            price = self.S0 - discount * np.sqrt(self.S0 * K) / np.pi * integral

            return max(price, 0.0)
        except Exception:
            return max(self.S0 - K * np.exp(-self.r * T), 0.0)

    def price_put_lewis(self, K: float, T: float) -> float:
        """Price European put using put-call parity."""
        call = self.price_call_lewis(K, T)
        put = call - self.S0 + K * np.exp(-self.r * T)
        return max(put, 0.0)

    def price_call_carr_madan(
        self, K: float, T: float, N: int = 4096, alpha: float = 1.5, eta: float = 0.25
    ) -> float:
        """Price European call using Carr-Madan FFT."""
        try:
            lambda_val = 2 * np.pi / (N * eta)
            b = N * lambda_val / 2
            ku = -b + lambda_val * np.arange(N)
            v = eta * np.arange(N)

            discount = np.exp(-self.r * T)
            i = complex(0, 1)

            psi = np.zeros(N, dtype=complex)

            for j in range(N):
                vj = v[j]
                if vj == 0:
                    vj = 1e-10

                u_arg = vj - (alpha + 1) * i

                try:
                    cf = self.characteristic_function(u_arg, T)
                    cf = cf / np.exp(i * u_arg * (np.log(self.S0) + self.r * T))

                    denominator = alpha**2 + alpha - vj**2 + i * vj * (2 * alpha + 1)

                    if abs(denominator) > 1e-10:
                        psi[j] = discount * cf / denominator
                except Exception:
                    psi[j] = 0.0

            simpson = 3 + (-1) ** (np.arange(N) + 1)
            simpson[0] = 1
            simpson = simpson / 3

            x = np.exp(i * b * v) * psi * eta * simpson
            fft_result = np.fft.fft(x)
            call_prices = np.real(np.exp(-alpha * ku) / np.pi * fft_result)

            log_K = np.log(K)
            idx = np.searchsorted(ku, log_K)

            if idx <= 0:
                price = call_prices[0]
            elif idx >= N:
                price = call_prices[-1]
            else:
                w = (log_K - ku[idx - 1]) / (ku[idx] - ku[idx - 1])
                price = (1 - w) * call_prices[idx - 1] + w * call_prices[idx]

            return max(price, 0.0)
        except Exception:
            return self.price_call_lewis(K, T)

    def price_put_carr_madan(self, K: float, T: float) -> float:
        """Price European put using put-call parity."""
        call = self.price_call_carr_madan(K, T)
        put = call - self.S0 + K * np.exp(-self.r * T)
        return max(put, 0.0)

    def simulate_paths(
        self, T: float, num_steps: int, num_paths: int, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate paths with jumps."""
        if seed is not None:
            np.random.seed(seed)

        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)

        S = np.zeros((num_paths, num_steps + 1))
        v = np.zeros((num_paths, num_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.params.v0

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma
        rho = self.params.rho
        lambda_j = self.params.lambda_j
        mu_j = self.params.mu_j
        sigma_j = self.params.sigma_j
        k = self.params.jump_compensator()

        for t in range(num_steps):
            Z1 = np.random.standard_normal(num_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(num_paths)

            v_plus = np.maximum(v[:, t], 0)
            sqrt_v = np.sqrt(v_plus)

            v[:, t + 1] = v[:, t] + kappa * (theta - v_plus) * dt + sigma * sqrt_v * sqrt_dt * Z2
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)

            num_jumps = np.random.poisson(lambda_j * dt, num_paths)
            jump_sizes = np.zeros(num_paths)

            if np.any(num_jumps > 0):
                jump_sizes[num_jumps > 0] = np.random.normal(
                    mu_j, sigma_j, size=np.sum(num_jumps > 0)
                )

                for i in range(num_paths):
                    if num_jumps[i] > 0:
                        jump_sizes[i] = np.sum(
                            np.random.normal(mu_j, sigma_j, num_jumps[i])
                        )

            drift = (self.r - lambda_j * k - 0.5 * v_plus) * dt
            diffusion = sqrt_v * sqrt_dt * Z1
            S[:, t + 1] = S[:, t] * np.exp(drift + diffusion + jump_sizes)

        return S, v


@dataclass
class CIRParameters:
    """CIR model parameters."""

    kappa: float
    theta: float
    sigma: float
    r0: float

    def to_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.r0])

    @classmethod
    def from_array(cls, arr: np.ndarray, r0: float) -> "CIRParameters":
        return cls(kappa=arr[0], theta=arr[1], sigma=arr[2], r0=r0)

    def feller_ratio(self) -> float:
        return (2 * self.kappa * self.theta) / (self.sigma**2 + 1e-10)

    def is_valid(self) -> bool:
        return self.kappa > 0 and self.theta > 0 and self.sigma > 0 and self.r0 > 0


class CIRModel:
    """Cox-Ingersoll-Ross (1985) interest rate model."""

    def __init__(self, params: CIRParameters):
        self.params = params

    def zero_coupon_bond_price(self, T: float, r: Optional[float] = None) -> float:
        """Analytical bond price."""
        if r is None:
            r = self.params.r0

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma

        gamma = np.sqrt(kappa**2 + 2 * sigma**2)

        exp_gamma_T = np.exp(gamma * T)

        denominator = (gamma + kappa) * (exp_gamma_T - 1) + 2 * gamma

        B = 2 * (exp_gamma_T - 1) / denominator

        A_exponent = 2 * kappa * theta / (sigma**2)
        A_base = 2 * gamma * np.exp((kappa + gamma) * T / 2) / denominator
        A = A_base**A_exponent

        return A * np.exp(-B * r)

    def yield_curve(self, T: float, r: Optional[float] = None) -> float:
        """Continuously compounded yield."""
        if T <= 1e-10:
            return r if r is not None else self.params.r0
        P = self.zero_coupon_bond_price(T, r)
        return -np.log(P) / T

    def expected_rate(self, T: float) -> float:
        """E[r_T]."""
        return self.params.theta + (self.params.r0 - self.params.theta) * np.exp(
            -self.params.kappa * T
        )

    def simulate_paths(
        self, T: float, num_steps: int, num_paths: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate using exact method (non-central chi-squared)."""
        if seed is not None:
            np.random.seed(seed)

        dt = T / num_steps

        kappa = self.params.kappa
        theta = self.params.theta
        sigma = self.params.sigma

        r = np.zeros((num_paths, num_steps + 1))
        r[:, 0] = self.params.r0

        c = sigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
        d = 4 * kappa * theta / (sigma**2)

        for t in range(num_steps):
            lambda_nc = r[:, t] * np.exp(-kappa * dt) / c
            r[:, t + 1] = c * np.random.noncentral_chisquare(d, lambda_nc)

        return r
