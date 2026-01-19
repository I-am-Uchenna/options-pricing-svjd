from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .models import BatesModel, BatesParameters, CIRModel, CIRParameters, HestonModel, HestonParameters


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    parameters: object
    rmse: float
    mse: float
    success: bool
    market_prices: np.ndarray
    model_prices: np.ndarray
    strikes: np.ndarray
    option_types: np.ndarray


class HestonCalibrator:
    """Fast calibrator for Heston model."""

    def __init__(self, S0: float, r: float):
        self.S0 = S0
        self.r = r

        self.bounds = [
            (0.02, 0.3),
            (0.5, 3.0),
            (0.02, 0.3),
            (0.2, 0.8),
            (-0.9, -0.3),
        ]

    def _price_all_options_fast(self, params, strikes, maturities, option_types):
        """Price all options using vectorized FFT."""
        try:
            heston_params = HestonParameters.from_array(params)
            model = HestonModel(heston_params, self.r, self.S0)

            prices = np.zeros(len(strikes))

            unique_mats = np.unique(maturities)

            for T in unique_mats:
                mask = maturities == T
                T_strikes = strikes[mask]
                T_types = option_types[mask]

                for i, (K, opt_type) in enumerate(zip(T_strikes, T_types)):
                    idx = np.where(mask)[0][i]
                    if opt_type.lower() == "call":
                        prices[idx] = model.price_call_lewis(K, T)
                    else:
                        prices[idx] = model.price_put_lewis(K, T)

            return prices
        except Exception:
            return np.full(len(strikes), np.nan)

    def _objective(self, params, strikes, maturities, market_prices, option_types):
        """Fast MSE objective."""
        model_prices = self._price_all_options_fast(params, strikes, maturities, option_types)

        if np.any(np.isnan(model_prices)):
            return 1e10

        return np.mean((model_prices - market_prices) ** 2)

    def calibrate(self, strikes, maturities, market_prices, option_types,
                  pricing_method='lewis', verbose=True):
        """Fast calibration using L-BFGS-B with good initial guess."""

        if verbose:
            print("Calibrating Heston model...")
            print(f"  Options: {len(strikes)}")

        avg_price = np.mean(market_prices)
        atm_mask = np.abs(strikes - self.S0) < 10
        if np.any(atm_mask):
            atm_prices = market_prices[atm_mask]
            atm_T = maturities[atm_mask]
            impl_var = np.mean((atm_prices / self.S0) ** 2 / atm_T) * 4
        else:
            impl_var = 0.04

        initial_guess = np.array([
            np.clip(impl_var, 0.02, 0.3),
            1.5,
            np.clip(impl_var, 0.02, 0.3),
            0.4,
            -0.7,
        ])

        def objective(params):
            return self._objective(params, strikes, maturities, market_prices, option_types)

        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        if not result.success or result.fun > 100:
            result = minimize(
                objective,
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4},
            )

        calibrated_params = HestonParameters.from_array(result.x)
        model_prices = self._price_all_options_fast(result.x, strikes, maturities, option_types)

        errors = model_prices - market_prices
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

        if verbose:
            print(f"  Calibration complete. RMSE: {rmse:.6f}")

        return CalibrationResult(
            parameters=calibrated_params,
            rmse=rmse,
            mse=mse,
            success=result.success,
            market_prices=market_prices,
            model_prices=model_prices,
            strikes=strikes,
            option_types=option_types,
        )


class BatesCalibrator:
    """Fast calibrator for Bates model."""

    def __init__(self, S0: float, r: float):
        self.S0 = S0
        self.r = r

        self.bounds = [
            (0.02, 0.3),
            (0.5, 3.0),
            (0.02, 0.3),
            (0.2, 0.8),
            (-0.9, -0.3),
            (0.0, 1.0),
            (-0.15, 0.0),
            (0.05, 0.2),
        ]

    def _price_all_options_fast(self, params, strikes, maturities, option_types):
        """Price all options."""
        try:
            bates_params = BatesParameters.from_array(params)
            model = BatesModel(bates_params, self.r, self.S0)

            prices = np.zeros(len(strikes))

            for i, (K, T, opt_type) in enumerate(zip(strikes, maturities, option_types)):
                if opt_type.lower() == "call":
                    prices[i] = model.price_call_lewis(K, T)
                else:
                    prices[i] = model.price_put_lewis(K, T)

            return prices
        except Exception:
            return np.full(len(strikes), np.nan)

    def _objective(self, params, strikes, maturities, market_prices, option_types):
        model_prices = self._price_all_options_fast(params, strikes, maturities, option_types)

        if np.any(np.isnan(model_prices)):
            return 1e10

        return np.mean((model_prices - market_prices) ** 2)

    def calibrate(self, strikes, maturities, market_prices, option_types,
                  pricing_method='lewis', verbose=True):

        if verbose:
            print("Calibrating Bates model...")
            print(f"  Options: {len(strikes)}")

        initial_guess = np.array([0.06, 1.5, 0.06, 0.4, -0.7, 0.2, -0.05, 0.1])

        def objective(params):
            return self._objective(params, strikes, maturities, market_prices, option_types)

        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100, 'ftol': 1e-6},
        )

        if not result.success or result.fun > 100:
            result = minimize(
                objective,
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 200},
            )

        calibrated_params = BatesParameters.from_array(result.x)
        model_prices = self._price_all_options_fast(result.x, strikes, maturities, option_types)

        mse = np.mean((model_prices - market_prices) ** 2)
        rmse = np.sqrt(mse)

        if verbose:
            print(f"  Calibration complete. RMSE: {rmse:.6f}")

        return CalibrationResult(
            parameters=calibrated_params,
            rmse=rmse,
            mse=mse,
            success=result.success,
            market_prices=market_prices,
            model_prices=model_prices,
            strikes=strikes,
            option_types=option_types,
        )


class CIRCalibrator:
    """Fast CIR calibrator."""

    def __init__(self):
        self.bounds = [
            (0.1, 5.0),
            (0.005, 0.08),
            (0.02, 0.3),
        ]

    def calibrate(self, maturities, market_rates, verbose=True):
        if verbose:
            print("Calibrating CIR model...")

        r0 = market_rates[0]

        def objective(params):
            kappa, theta, sigma = params

            if 2 * kappa * theta <= sigma**2:
                return 1e10

            cir_params = CIRParameters(kappa, theta, sigma, r0)
            model = CIRModel(cir_params)

            model_rates = np.array([model.yield_curve(T) for T in maturities])
            return np.mean((market_rates - model_rates) ** 2)

        initial_guess = np.array([1.0, 0.02, 0.1])

        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100},
        )

        kappa, theta, sigma = result.x
        calibrated_params = CIRParameters(kappa, theta, sigma, r0)

        model = CIRModel(calibrated_params)
        model_rates = np.array([model.yield_curve(T) for T in maturities])

        rmse = np.sqrt(np.mean((market_rates - model_rates) ** 2))

        if verbose:
            print(f"  Calibration complete. RMSE: {rmse:.8f}")

        return calibrated_params, rmse, model_rates
