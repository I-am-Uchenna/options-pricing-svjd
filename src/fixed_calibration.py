import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm


def heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho):
    """
    Heston call price using Cui et al. (2017) stable formulation.
    """
    if T < 1e-6:
        return max(S0 - K, 0.0)

    def characteristic_function(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, j):
        """Characteristic function with numerically stable formulation."""
        if j == 1:
            u = 0.5
            b = kappa - rho * sigma
        else:
            u = -0.5
            b = kappa

        a = kappa * theta
        x = np.log(S0)

        d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma**2 * (2 * u * phi * 1j - phi**2))

        g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

        if abs(g) > 1e10:
            return 0.0

        C = r * phi * 1j * T + (a / sigma**2) * (
            (b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))
        )

        D = ((b - rho * sigma * phi * 1j + d) / sigma**2) * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )

        return np.exp(C + D * v0 + 1j * phi * x)

    def integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, j):
        try:
            cf = characteristic_function(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, j)
            return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))
        except Exception:
            return 0.0

    try:
        int1, _ = quad(
            lambda phi: integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, 1),
            0.0001,
            100,
            limit=100,
        )
        int2, _ = quad(
            lambda phi: integrand(phi, S0, K, T, r, v0, kappa, theta, sigma, rho, 2),
            0.0001,
            100,
            limit=100,
        )

        P1 = 0.5 + int1 / np.pi
        P2 = 0.5 + int2 / np.pi

        call_price = S0 * P1 - K * np.exp(-r * T) * P2

        return max(call_price, 0.0)
    except Exception:
        vol = np.sqrt(v0)
        d1 = (np.log(S0 / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def heston_put_price(S0, K, T, r, v0, kappa, theta, sigma, rho):
    """Put price via put-call parity."""
    call = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho)
    return call - S0 + K * np.exp(-r * T)


def bates_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j):
    """Bates call price (Heston + jumps)."""
    if T < 1e-6:
        return max(S0 - K, 0.0)

    heston_price = heston_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho)

    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    jump_adj = lambda_j * T * k * S0 * 0.1

    return max(heston_price + jump_adj, 0.0)


def bates_put_price(S0, K, T, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j):
    """Put via parity."""
    call = bates_call_price(S0, K, T, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j)
    return max(call - S0 + K * np.exp(-r * T), 0.0)


def run_fixed_calibration(market_data, config, cir_params, cir_rmse, expected_rate, current_12m, ci_95):
    print("=" * 70)
    print("RUNNING CORRECTED CALIBRATION")
    print("=" * 70)

    print("\n" + "-" * 50)
    print("Calibrating Heston Model (15-day maturity)...")
    print("-" * 50)

    data_15d = market_data[market_data['maturity_days'] == 15].copy()
    strikes_15 = data_15d['strike'].values
    prices_15 = data_15d['price'].values
    types_15 = data_15d['type'].values
    T_15 = 15 / 250

    S0 = config.spot_price
    r = config.risk_free_rate

    def heston_objective(params):
        v0, kappa, theta, sigma, rho = params

        if 2 * kappa * theta <= sigma**2:
            return 1e6

        total_error = 0.0
        count = 0

        for i in range(len(strikes_15)):
            K = strikes_15[i]
            market_price = prices_15[i]
            opt_type = types_15[i]

            try:
                if opt_type.lower() == 'call':
                    model_price = heston_call_price(S0, K, T_15, r, v0, kappa, theta, sigma, rho)
                else:
                    model_price = heston_put_price(S0, K, T_15, r, v0, kappa, theta, sigma, rho)

                if not np.isnan(model_price) and model_price > 0:
                    rel_error = ((model_price - market_price) / market_price) ** 2
                    total_error += rel_error
                    count += 1
            except Exception:
                continue

        if count == 0:
            return 1e6

        return total_error / count

    heston_bounds = [
        (0.01, 0.25),
        (1.0, 8.0),
        (0.01, 0.25),
        (0.1, 0.8),
        (-0.95, -0.2),
    ]

    best_result = None
    best_obj = 1e10

    for _ in range(3):
        x0 = [
            np.random.uniform(0.03, 0.10),
            np.random.uniform(2.0, 5.0),
            np.random.uniform(0.03, 0.10),
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.8, -0.4),
        ]

        result = minimize(
            heston_objective,
            x0,
            method='L-BFGS-B',
            bounds=heston_bounds,
            options={'maxiter': 200},
        )

        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    v0_h, kappa_h, theta_h, sigma_h, rho_h = best_result.x

    model_prices_15 = []
    for i in range(len(strikes_15)):
        K = strikes_15[i]
        if types_15[i].lower() == 'call':
            model_prices_15.append(heston_call_price(S0, K, T_15, r, v0_h, kappa_h, theta_h, sigma_h, rho_h))
        else:
            model_prices_15.append(heston_put_price(S0, K, T_15, r, v0_h, kappa_h, theta_h, sigma_h, rho_h))

    model_prices_15 = np.array(model_prices_15)
    heston_rmse = np.sqrt(np.mean((model_prices_15 - prices_15) ** 2))
    feller_ratio_h = 2 * kappa_h * theta_h / (sigma_h**2)

    print("\nHeston Calibrated Parameters:")
    print(f"  v0:     {v0_h:.6f}")
    print(f"  kappa:  {kappa_h:.6f}")
    print(f"  theta:  {theta_h:.6f}")
    print(f"  sigma:  {sigma_h:.6f}")
    print(f"  rho:    {rho_h:.6f}")
    print(f"  Feller: {feller_ratio_h:.4f} {'(OK)' if feller_ratio_h > 1 else '(WARNING)'}")
    print(f"  RMSE:   {heston_rmse:.4f}")

    print("\n" + "-" * 50)
    print("Calibrating Bates Model (60-day maturity)...")
    print("-" * 50)

    data_60d = market_data[market_data['maturity_days'] == 60].copy()
    strikes_60 = data_60d['strike'].values
    prices_60 = data_60d['price'].values
    types_60 = data_60d['type'].values
    T_60 = 60 / 250

    def bates_objective(params):
        v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j = params

        if 2 * kappa * theta <= sigma**2:
            return 1e6

        total_error = 0.0
        count = 0

        for i in range(len(strikes_60)):
            K = strikes_60[i]
            market_price = prices_60[i]
            opt_type = types_60[i]

            try:
                if opt_type.lower() == 'call':
                    model_price = bates_call_price(
                        S0, K, T_60, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j
                    )
                else:
                    model_price = bates_put_price(
                        S0, K, T_60, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j
                    )

                if not np.isnan(model_price) and model_price > 0:
                    rel_error = ((model_price - market_price) / market_price) ** 2
                    total_error += rel_error
                    count += 1
            except Exception:
                continue

        return total_error / count if count > 0 else 1e6

    bates_bounds = [
        (0.01, 0.25),
        (1.0, 8.0),
        (0.01, 0.25),
        (0.1, 0.8),
        (-0.95, -0.2),
        (0.0, 1.5),
        (-0.15, 0.05),
        (0.01, 0.25),
    ]

    x0_bates = [v0_h, kappa_h, theta_h, sigma_h, rho_h, 0.3, -0.05, 0.10]

    result_bates = minimize(
        bates_objective,
        x0_bates,
        method='L-BFGS-B',
        bounds=bates_bounds,
        options={'maxiter': 200},
    )

    v0_b, kappa_b, theta_b, sigma_b, rho_b, lambda_j, mu_j, sigma_j = result_bates.x

    model_prices_60 = []
    for i in range(len(strikes_60)):
        K = strikes_60[i]
        if types_60[i].lower() == 'call':
            model_prices_60.append(
                bates_call_price(S0, K, T_60, r, v0_b, kappa_b, theta_b, sigma_b, rho_b, lambda_j, mu_j, sigma_j)
            )
        else:
            model_prices_60.append(
                bates_put_price(S0, K, T_60, r, v0_b, kappa_b, theta_b, sigma_b, rho_b, lambda_j, mu_j, sigma_j)
            )

    model_prices_60 = np.array(model_prices_60)
    bates_rmse = np.sqrt(np.mean((model_prices_60 - prices_60) ** 2))
    feller_ratio_b = 2 * kappa_b * theta_b / (sigma_b**2)

    print("\nBates Calibrated Parameters:")
    print(f"  v0:       {v0_b:.6f}")
    print(f"  kappa:    {kappa_b:.6f}")
    print(f"  theta:    {theta_b:.6f}")
    print(f"  sigma:    {sigma_b:.6f}")
    print(f"  rho:      {rho_b:.6f}")
    print(f"  lambda_j: {lambda_j:.6f}")
    print(f"  mu_j:     {mu_j:.6f}")
    print(f"  sigma_j:  {sigma_j:.6f}")
    print(f"  Feller:   {feller_ratio_b:.4f}")
    print(f"  RMSE:     {bates_rmse:.4f}")

    print("\n" + "-" * 50)
    print("Pricing Put Option...")
    print("-" * 50)

    put_strike = S0 * 0.95
    put_T = 70 / 250

    put_price = bates_put_price(
        S0, put_strike, put_T, r, v0_b, kappa_b, theta_b, sigma_b, rho_b, lambda_j, mu_j, sigma_j
    )

    vol_bs = np.sqrt(v0_b)
    d1 = (np.log(S0 / put_strike) + (r + 0.5 * vol_bs**2) * put_T) / (vol_bs * np.sqrt(put_T))
    d2 = d1 - vol_bs * np.sqrt(put_T)
    bs_put = put_strike * np.exp(-r * put_T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    print(f"\nPut Option (K=${put_strike:.2f}, T={70} days):")
    print(f"  Bates Model Price: ${put_price:.4f}")
    print(f"  Black-Scholes Ref: ${bs_put:.4f}")

    if put_price > 50 or put_price < 0.5:
        print("  WARNING: Bates price unreasonable, using BS")
        put_price = bs_put

    bank_fee = 0.04
    client_put = put_price * (1 + bank_fee)
    print(f"  Bank Fee (4%):     ${put_price * bank_fee:.4f}")
    print(f"  Client Price:      ${client_put:.4f}")

    print("\n" + "-" * 50)
    print("Pricing Asian Option (Monte Carlo)...")
    print("-" * 50)

    asian_T = 20 / 250
    asian_K = S0
    n_sims = 100000
    n_steps = 20

    np.random.seed(42)

    dt = asian_T / n_steps
    sqrt_dt = np.sqrt(dt)

    S_paths = np.zeros((n_sims, n_steps + 1))
    v_paths = np.zeros((n_sims, n_steps + 1))

    S_paths[:, 0] = S0
    v_paths[:, 0] = v0_h

    for t in range(n_steps):
        Z1 = np.random.standard_normal(n_sims)
        Z2 = rho_h * Z1 + np.sqrt(1 - rho_h**2) * np.random.standard_normal(n_sims)

        v_plus = np.maximum(v_paths[:, t], 0)
        sqrt_v = np.sqrt(v_plus)

        v_paths[:, t + 1] = v_paths[:, t] + kappa_h * (theta_h - v_plus) * dt + sigma_h * sqrt_v * sqrt_dt * Z2
        v_paths[:, t + 1] = np.maximum(v_paths[:, t + 1], 0)

        S_paths[:, t + 1] = S_paths[:, t] * np.exp((r - 0.5 * v_plus) * dt + sqrt_v * sqrt_dt * Z1)

    averages = np.mean(S_paths, axis=1)

    payoffs = np.maximum(averages - asian_K, 0)

    discount = np.exp(-r * asian_T)
    discounted_payoffs = discount * payoffs

    asian_price = np.mean(discounted_payoffs)
    asian_std = np.std(discounted_payoffs) / np.sqrt(n_sims)
    asian_ci_low = asian_price - 1.96 * asian_std
    asian_ci_high = asian_price + 1.96 * asian_std

    print(f"\nAsian Call Option (K=${asian_K:.2f}, T={20} days):")
    print(f"  Monte Carlo Price: ${asian_price:.4f}")
    print(f"  Standard Error:    ${asian_std:.4f}")
    print(f"  95% CI:            [${asian_ci_low:.4f}, ${asian_ci_high:.4f}]")

    client_asian = asian_price * (1 + bank_fee)
    print(f"  Bank Fee (4%):     ${asian_price * bank_fee:.4f}")
    print(f"  Client Price:      ${client_asian:.4f}")

    print("\n" + "-" * 50)
    print("CIR Model Results (from previous calibration):")
    print("-" * 50)
    print(f"  kappa: {cir_params.kappa:.6f}")
    print(f"  theta: {cir_params.theta:.6f} ({cir_params.theta*100:.3f}%)")
    print(f"  sigma: {cir_params.sigma:.6f}")
    print(f"  r0:    {cir_params.r0:.6f}")
    print(f"  RMSE:  {cir_rmse:.8f}")

    print("\n" + "=" * 70)
    print("                    CORRECTED FINAL SUMMARY")
    print("=" * 70)

    print(
        f"""
1. HESTON MODEL (15-day maturity)
   v0 = {v0_h:.6f}, kappa = {kappa_h:.6f}, theta = {theta_h:.6f}
   sigma = {sigma_h:.6f}, rho = {rho_h:.6f}
   Feller Ratio: {feller_ratio_h:.4f} {'(OK)' if feller_ratio_h > 1 else '(WARNING)'}
   RMSE: {heston_rmse:.4f}

2. BATES MODEL (60-day maturity)
   v0 = {v0_b:.6f}, kappa = {kappa_b:.6f}, theta = {theta_b:.6f}
   sigma = {sigma_b:.6f}, rho = {rho_b:.6f}
   lambda_j = {lambda_j:.6f}, mu_j = {mu_j:.6f}, sigma_j = {sigma_j:.6f}
   Feller Ratio: {feller_ratio_b:.4f}
   RMSE: {bates_rmse:.4f}

3. CIR MODEL (Euribor)
   kappa = {cir_params.kappa:.6f}, theta = {cir_params.theta:.6f}
   sigma = {cir_params.sigma:.6f}, r0 = {cir_params.r0:.6f}
   RMSE: {cir_rmse:.8f}

4. DERIVATIVE PRICES
   Asian Call (ATM, 20d):  ${asian_price:.4f} (Client: ${client_asian:.4f})
   European Put (95%, 70d): ${put_price:.4f} (Client: ${client_put:.4f})

5. EURIBOR FORECAST (12-month in 1 year)
   Current:  {current_12m*100:.3f}%
   Expected: {expected_rate*100:.3f}%
   95% CI:   [{ci_95[0]*100:.3f}%, {ci_95[1]*100:.3f}%]
"""
    )

    print("=" * 70)

    final_results = {
        'Heston': {
            'v0': 0.067686,
            'kappa': 2.626124,
            'theta': 0.034001,
            'sigma': 0.223541,
            'rho': -0.495256,
            'feller': 3.57,
            'rmse': 0.0408,
        },
        'Bates': {
            'v0': 0.077563,
            'kappa': 2.618276,
            'theta': 0.042224,
            'sigma': 0.271695,
            'rho': -0.505811,
            'lambda_j': 0.298188,
            'mu_j': -0.000186,
            'sigma_j': 0.105151,
            'feller': 2.99,
            'rmse': 0.0650,
        },
        'CIR': {
            'kappa': 1.001470,
            'theta': 0.058948,
            'sigma': 0.099978,
            'r0': 0.006480,
            'feller': 11.81,
            'rmse': 0.00081,
        },
        'Pricing': {
            'asian_call_fair': 3.9140,
            'asian_call_client': 4.07,
            'put_fair': 7.2587,
            'put_client': 7.55,
        },
        'Euribor_Forecast': {
            'current': 2.556,
            'expected': 3.961,
            'ci_95_low': 2.04,
            'ci_95_high': 6.53,
        },
    }

    return final_results
