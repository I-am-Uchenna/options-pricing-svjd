import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from src.calibration import BatesCalibrator, CIRCalibrator, HestonCalibrator
from src.config import config
from src.data_loader import create_market_data
from src.models import BatesModel, CIRModel, HestonModel
from src.plots import (
    plot_calibration_fit,
    plot_rate_distribution,
    plot_simulation_paths,
    plot_term_structure,
)
from src.pricing import AsianOptionPricer
from src.utils import configure_environment


def run_pipeline() -> None:
    configure_environment(seed=42)

    market_data = create_market_data(config.spot_price, config.risk_free_rate)

    print("Market Data Summary:")
    print(f"  Total options: {len(market_data)}")
    print(f"  Calls: {len(market_data[market_data['type'] == 'call'])}")
    print(f"  Puts: {len(market_data[market_data['type'] == 'put'])}")
    print(f"  Maturities: {sorted(market_data['maturity_days'].unique())} days")
    print("\nSample data:")
    print(market_data.head(10).to_string(index=False))

    print("=" * 70)
    print("STEP 1: HESTON MODEL CALIBRATION - 15 DAY MATURITY")
    print("=" * 70)

    data_15d = market_data[market_data['maturity_days'] == 15].copy()

    print("\nData for 15-day maturity:")
    print(f"  Total options: {len(data_15d)}")
    print(f"  Calls: {len(data_15d[data_15d['type'] == 'call'])}")
    print(f"  Puts: {len(data_15d[data_15d['type'] == 'put'])}")

    strikes_15d = data_15d['strike'].values
    maturities_15d = data_15d['maturity_years'].values
    prices_15d = data_15d['price'].values
    types_15d = data_15d['type'].values

    heston_calibrator = HestonCalibrator(config.spot_price, config.risk_free_rate)

    print("\n" + "-" * 50)
    print("Task 1(a): Lewis (2001) Calibration")
    print("-" * 50)

    result_heston_lewis = heston_calibrator.calibrate(
        strikes_15d,
        maturities_15d,
        prices_15d,
        types_15d,
        pricing_method='lewis',
        verbose=True,
    )

    heston_lewis = result_heston_lewis.parameters
    print("\nCalibrated Parameters (Lewis):")
    print(f"  v0 (Initial Variance):    {heston_lewis.v0:.6f}")
    print(f"  kappa (Mean Reversion):   {heston_lewis.kappa:.6f}")
    print(f"  theta (Long-term Var):    {heston_lewis.theta:.6f}")
    print(f"  sigma (Vol of Vol):       {heston_lewis.sigma:.6f}")
    print(f"  rho (Correlation):        {heston_lewis.rho:.6f}")
    print(f"  Feller Ratio:             {heston_lewis.feller_ratio():.4f}")
    print(f"  RMSE:                     {result_heston_lewis.rmse:.6f}")

    fig1 = plot_calibration_fit(
        result_heston_lewis.strikes,
        result_heston_lewis.market_prices,
        result_heston_lewis.model_prices,
        result_heston_lewis.option_types,
        "Heston Model - Lewis (2001) Calibration - 15 Days",
    )
    plt.show()

    print("\n" + "-" * 50)
    print("Task 1(b): Carr-Madan (1999) FFT Calibration")
    print("-" * 50)

    result_heston_cm = heston_calibrator.calibrate(
        strikes_15d,
        maturities_15d,
        prices_15d,
        types_15d,
        pricing_method='carr_madan',
        verbose=True,
    )

    heston_cm = result_heston_cm.parameters
    print("\nCalibrated Parameters (Carr-Madan):")
    print(f"  v0:     {heston_cm.v0:.6f}")
    print(f"  kappa:  {heston_cm.kappa:.6f}")
    print(f"  theta:  {heston_cm.theta:.6f}")
    print(f"  sigma:  {heston_cm.sigma:.6f}")
    print(f"  rho:    {heston_cm.rho:.6f}")
    print(f"  RMSE:   {result_heston_cm.rmse:.6f}")

    print("\n" + "=" * 50)
    print("COMPARISON: Lewis vs Carr-Madan")
    print("=" * 50)

    comparison = pd.DataFrame(
        {
            'Parameter': ['v0', 'kappa', 'theta', 'sigma', 'rho', 'RMSE'],
            'Lewis': [
                heston_lewis.v0,
                heston_lewis.kappa,
                heston_lewis.theta,
                heston_lewis.sigma,
                heston_lewis.rho,
                result_heston_lewis.rmse,
            ],
            'Carr-Madan': [
                heston_cm.v0,
                heston_cm.kappa,
                heston_cm.theta,
                heston_cm.sigma,
                heston_cm.rho,
                result_heston_cm.rmse,
            ],
        }
    )
    comparison['Diff'] = np.abs(comparison['Lewis'] - comparison['Carr-Madan'])
    print(comparison.to_string(index=False))

    fig2 = plot_calibration_fit(
        result_heston_cm.strikes,
        result_heston_cm.market_prices,
        result_heston_cm.model_prices,
        result_heston_cm.option_types,
        "Heston Model - Carr-Madan (1999) Calibration - 15 Days",
    )
    plt.show()

    print("\n" + "=" * 70)
    print("STEP 2: BATES MODEL CALIBRATION - 60 DAY MATURITY")
    print("=" * 70)

    data_60d = market_data[market_data['maturity_days'] == 60].copy()

    print("\nData for 60-day maturity:")
    print(f"  Total options: {len(data_60d)}")
    print(f"  Calls: {len(data_60d[data_60d['type'] == 'call'])}")
    print(f"  Puts: {len(data_60d[data_60d['type'] == 'put'])}")

    strikes_60d = data_60d['strike'].values
    maturities_60d = data_60d['maturity_years'].values
    prices_60d = data_60d['price'].values
    types_60d = data_60d['type'].values

    bates_calibrator = BatesCalibrator(config.spot_price, config.risk_free_rate)

    print("\n" + "-" * 50)
    print("Task 2(a): Bates Model - Lewis (2001) Calibration")
    print("-" * 50)

    result_bates_lewis = bates_calibrator.calibrate(
        strikes_60d,
        maturities_60d,
        prices_60d,
        types_60d,
        pricing_method='lewis',
        verbose=True,
    )

    bates_lewis = result_bates_lewis.parameters
    print("\nCalibrated Bates Parameters (Lewis):")
    print("  Stochastic Volatility:")
    print(f"    v0 (Initial Variance):    {bates_lewis.v0:.6f}")
    print(f"    kappa (Mean Reversion):   {bates_lewis.kappa:.6f}")
    print(f"    theta (Long-term Var):    {bates_lewis.theta:.6f}")
    print(f"    sigma (Vol of Vol):       {bates_lewis.sigma:.6f}")
    print(f"    rho (Correlation):        {bates_lewis.rho:.6f}")
    print("  Jump Parameters:")
    print(f"    lambda_j (Jump Intensity): {bates_lewis.lambda_j:.6f}")
    print(f"    mu_j (Mean Jump Size):     {bates_lewis.mu_j:.6f}")
    print(f"    sigma_j (Jump Vol):        {bates_lewis.sigma_j:.6f}")
    print(f"  Feller Ratio:               {bates_lewis.feller_ratio():.4f}")
    print(f"  RMSE:                       {result_bates_lewis.rmse:.6f}")

    fig3 = plot_calibration_fit(
        result_bates_lewis.strikes,
        result_bates_lewis.market_prices,
        result_bates_lewis.model_prices,
        result_bates_lewis.option_types,
        "Bates Model - Lewis (2001) Calibration - 60 Days",
    )
    plt.show()

    print("\n" + "-" * 50)
    print("Task 2(b): Bates Model - Carr-Madan (1999) Calibration")
    print("-" * 50)

    result_bates_cm = bates_calibrator.calibrate(
        strikes_60d,
        maturities_60d,
        prices_60d,
        types_60d,
        pricing_method='carr_madan',
        verbose=True,
    )

    bates_cm = result_bates_cm.parameters
    print("\nCalibrated Bates Parameters (Carr-Madan):")
    print(f"  v0:       {bates_cm.v0:.6f}")
    print(f"  kappa:    {bates_cm.kappa:.6f}")
    print(f"  theta:    {bates_cm.theta:.6f}")
    print(f"  sigma:    {bates_cm.sigma:.6f}")
    print(f"  rho:      {bates_cm.rho:.6f}")
    print(f"  lambda_j: {bates_cm.lambda_j:.6f}")
    print(f"  mu_j:     {bates_cm.mu_j:.6f}")
    print(f"  sigma_j:  {bates_cm.sigma_j:.6f}")
    print(f"  RMSE:     {result_bates_cm.rmse:.6f}")

    print("\n" + "=" * 50)
    print("COMPARISON: Bates Lewis vs Carr-Madan")
    print("=" * 50)

    bates_comparison = pd.DataFrame(
        {
            'Parameter': [
                'v0',
                'kappa',
                'theta',
                'sigma',
                'rho',
                'lambda_j',
                'mu_j',
                'sigma_j',
                'RMSE',
            ],
            'Lewis': [
                bates_lewis.v0,
                bates_lewis.kappa,
                bates_lewis.theta,
                bates_lewis.sigma,
                bates_lewis.rho,
                bates_lewis.lambda_j,
                bates_lewis.mu_j,
                bates_lewis.sigma_j,
                result_bates_lewis.rmse,
            ],
            'Carr-Madan': [
                bates_cm.v0,
                bates_cm.kappa,
                bates_cm.theta,
                bates_cm.sigma,
                bates_cm.rho,
                bates_cm.lambda_j,
                bates_cm.mu_j,
                bates_cm.sigma_j,
                result_bates_cm.rmse,
            ],
        }
    )
    print(bates_comparison.to_string(index=False))

    fig4 = plot_calibration_fit(
        result_bates_cm.strikes,
        result_bates_cm.market_prices,
        result_bates_cm.model_prices,
        result_bates_cm.option_types,
        "Bates Model - Carr-Madan (1999) Calibration - 60 Days",
    )
    plt.show()

    print("\n" + "=" * 70)
    print("STEP 2(c): PRICING PUT OPTION")
    print("=" * 70)

    put_maturity_days = config.PUT_MATURITY_DAYS
    put_maturity_years = put_maturity_days / config.trading_days_per_year
    put_moneyness = config.PUT_MONEYNESS
    put_strike = config.spot_price * put_moneyness

    print("\nPut Option Specifications:")
    print("  Underlying:     SM Energy Company")
    print(f"  Spot Price:     ${config.spot_price:.2f}")
    print(f"  Strike Price:   ${put_strike:.2f}")
    print(f"  Moneyness:      {put_moneyness*100:.0f}%")
    print(
        f"  Maturity:       {put_maturity_days} days ({put_maturity_years:.4f} years)"
    )
    print("  Option Type:    European Put")

    bates_model_pricing = BatesModel(
        result_bates_lewis.parameters, config.risk_free_rate, config.spot_price
    )

    print("\n" + "-" * 50)
    print("Pricing Results")
    print("-" * 50)

    put_price_lewis = bates_model_pricing.price_put_lewis(put_strike, put_maturity_years)
    put_price_cm = bates_model_pricing.price_put_carr_madan(put_strike, put_maturity_years)

    print(f"\n  Lewis (2001) Price:      ${put_price_lewis:.4f}")
    print(f"  Carr-Madan (1999) Price: ${put_price_cm:.4f}")
    print(f"  Difference:              ${abs(put_price_lewis - put_price_cm):.4f}")

    fair_price_put = (put_price_lewis + put_price_cm) / 2

    bank_fee = config.BANK_FEE
    fee_amount = fair_price_put * bank_fee
    client_price_put = fair_price_put + fee_amount

    print("\n" + "-" * 50)
    print("CLIENT PRICING")
    print("-" * 50)
    print(f"  Fair Value:              ${fair_price_put:.4f}")
    print(f"  Bank Fee ({bank_fee*100:.0f}%):          ${fee_amount:.4f}")
    print(f"  Client Price:            ${client_price_put:.4f}")

    print("\n" + "=" * 70)
    print("ASIAN OPTION PRICING - MONTE CARLO")
    print("=" * 70)

    asian_maturity_days = config.ASIAN_MATURITY_DAYS
    asian_maturity_years = asian_maturity_days / config.trading_days_per_year
    asian_strike = config.spot_price

    print("\nAsian Option Specifications:")
    print("  Underlying:     SM Energy Company")
    print(f"  Spot Price:     ${config.spot_price:.2f}")
    print(f"  Strike Price:   ${asian_strike:.2f} (ATM)")
    print(
        f"  Maturity:       {asian_maturity_days} days ({asian_maturity_years:.4f} years)"
    )
    print("  Option Type:    Arithmetic Average Asian Call")
    print("  Averaging:      Daily, including S0")

    heston_model_pricing = HestonModel(
        result_heston_lewis.parameters, config.risk_free_rate, config.spot_price
    )

    asian_pricer = AsianOptionPricer(heston_model_pricing, config.risk_free_rate)

    print("\n" + "-" * 50)
    print("Running Monte Carlo Simulation...")
    print("-" * 50)

    num_sims = 100000
    asian_result = asian_pricer.price_arithmetic_call(
        K=asian_strike,
        T=asian_maturity_years,
        num_simulations=num_sims,
        num_steps=asian_maturity_days,
        include_S0=True,
        seed=42,
    )

    print(f"\nMonte Carlo Results ({num_sims:,} simulations):")
    print(f"  Fair Price:              ${asian_result.price:.4f}")
    print(f"  Standard Error:          ${asian_result.std_error:.4f}")
    print(
        f"  95% Confidence Interval: [${asian_result.ci_lower:.4f}, ${asian_result.ci_upper:.4f}]"
    )
    print(f"  Exercise Probability:    {asian_result.exercise_probability*100:.2f}%")

    fee_amount_asian = asian_result.price * config.BANK_FEE
    client_price_asian = asian_result.price + fee_amount_asian

    print("\n" + "-" * 50)
    print("CLIENT PRICING")
    print("-" * 50)
    print(f"  Fair Value:              ${asian_result.price:.4f}")
    print(f"  Bank Fee ({config.BANK_FEE*100:.0f}%):          ${fee_amount_asian:.4f}")
    print(f"  Client Price:            ${client_price_asian:.4f}")

    print("\n" + "=" * 70)
    print("STEP 3: CIR MODEL CALIBRATION TO EURIBOR TERM STRUCTURE")
    print("=" * 70)

    euribor_data = {
        '1 week': {'maturity': 1 / 52, 'rate': 0.00648},
        '1 month': {'maturity': 1 / 12, 'rate': 0.00679},
        '3 months': {'maturity': 3 / 12, 'rate': 0.01173},
        '6 months': {'maturity': 6 / 12, 'rate': 0.01809},
        '12 months': {'maturity': 12 / 12, 'rate': 0.02556},
    }

    print("\nEuribor Market Rates:")
    print("-" * 40)
    for tenor, data in euribor_data.items():
        print(f"  {tenor:12s}: {data['rate']*100:.3f}%")

    euribor_maturities = np.array([d['maturity'] for d in euribor_data.values()])
    euribor_rates = np.array([d['rate'] for d in euribor_data.values()])

    print("\n" + "-" * 50)
    print("Interpolating Term Structure (Cubic Spline)")
    print("-" * 50)

    cs = CubicSpline(euribor_maturities, euribor_rates, bc_type='natural')
    weekly_maturities = np.array([i / 52 for i in range(1, 53)])
    weekly_rates = cs(weekly_maturities)

    print(f"  Original points: {len(euribor_maturities)}")
    print(f"  Interpolated points: {len(weekly_maturities)}")
    print(
        f"  Maturity range: [{weekly_maturities[0]:.4f}, {weekly_maturities[-1]:.4f}] years"
    )

    print("\n" + "-" * 50)
    print("CIR Model Calibration")
    print("-" * 50)

    cir_calibrator = CIRCalibrator()
    cir_params, cir_rmse, cir_model_rates = cir_calibrator.calibrate(
        weekly_maturities, weekly_rates, verbose=True
    )

    print("\nCalibrated CIR Parameters:")
    print(f"  kappa (Mean Reversion): {cir_params.kappa:.6f}")
    print(f"  theta (Long-term Rate): {cir_params.theta:.6f} ({cir_params.theta*100:.3f}%)")
    print(f"  sigma (Volatility):     {cir_params.sigma:.6f}")
    print(f"  r0 (Initial Rate):      {cir_params.r0:.6f} ({cir_params.r0*100:.3f}%)")
    print(f"  Feller Ratio:           {cir_params.feller_ratio():.4f}")
    print(f"  RMSE:                   {cir_rmse:.8f}")

    fig5 = plot_term_structure(
        weekly_maturities,
        weekly_rates,
        cir_model_rates,
        "CIR Model Calibration to Euribor Term Structure",
    )
    plt.show()

    print("\n" + "=" * 70)
    print("STEP 3(b): MONTE CARLO SIMULATION OF 12-MONTH EURIBOR")
    print("=" * 70)

    cir_model = CIRModel(cir_params)

    sim_horizon = 1.0
    num_sims_cir = 100000
    num_steps_cir = 252

    print("\nSimulation Parameters:")
    print(f"  Number of simulations: {num_sims_cir:,}")
    print(f"  Time horizon:          {sim_horizon} year")
    print(f"  Time steps:            {num_steps_cir} (daily)")

    print("\nRunning Monte Carlo simulation...")
    rate_paths = cir_model.simulate_paths(
        T=sim_horizon, num_steps=num_steps_cir, num_paths=num_sims_cir, seed=42
    )
    print("Simulation completed.")

    terminal_rates = rate_paths[:, -1]

    current_12m = euribor_data['12 months']['rate']

    expected_rate = np.mean(terminal_rates)
    std_rate = np.std(terminal_rates)
    median_rate = np.median(terminal_rates)

    ci_90 = (np.percentile(terminal_rates, 5), np.percentile(terminal_rates, 95))
    ci_95 = (np.percentile(terminal_rates, 2.5), np.percentile(terminal_rates, 97.5))
    ci_99 = (np.percentile(terminal_rates, 0.5), np.percentile(terminal_rates, 99.5))

    print("\n" + "=" * 50)
    print("SIMULATION RESULTS: 12-Month Euribor in 1 Year")
    print("=" * 50)

    print(f"\nCurrent 12-month Euribor:     {current_12m*100:.3f}%")
    print(f"Expected rate in 1 year:      {expected_rate*100:.3f}%")
    print(f"Median rate in 1 year:        {median_rate*100:.3f}%")
    print(f"Standard deviation:           {std_rate*100:.3f}%")

    print("\nConfidence Intervals:")
    print(f"  90% CI: [{ci_90[0]*100:.3f}%, {ci_90[1]*100:.3f}%]")
    print(f"  95% CI: [{ci_95[0]*100:.3f}%, {ci_95[1]*100:.3f}%]")
    print(f"  99% CI: [{ci_99[0]*100:.3f}%, {ci_99[1]*100:.3f}%]")

    print("\nRange Analysis (95% Confidence):")
    print(f"  Minimum expected rate: {ci_95[0]*100:.3f}%")
    print(f"  Maximum expected rate: {ci_95[1]*100:.3f}%")
    print(f"  Range width:           {(ci_95[1] - ci_95[0]) * 100:.3f}%")

    fig6 = plot_rate_distribution(
        terminal_rates,
        current_12m,
        "Distribution of 12-Month Euribor in 1 Year (100,000 Simulations)",
    )
    plt.show()

    time_grid = np.linspace(0, sim_horizon, num_steps_cir + 1)

    fig7 = plot_simulation_paths(
        time_grid,
        rate_paths,
        "Simulated Interest Rate Paths (CIR Model)",
        ylabel="Short Rate",
    )
    plt.show()

    print("\n" + "=" * 70)
    print("IMPACT ANALYSIS: INTEREST RATE CHANGES ON PRICING")
    print("=" * 70)

    print("\nRate Scenarios:")
    print(f"  Current rate:       {config.risk_free_rate*100:.2f}%")
    print(f"  Expected rate (1Y): {expected_rate*100:.3f}%")
    print(f"  Rate change:        {(expected_rate - config.risk_free_rate)*100:+.3f}%")

    heston_model_future = HestonModel(
        result_heston_lewis.parameters,
        expected_rate,
        config.spot_price,
    )

    asian_pricer_future = AsianOptionPricer(heston_model_future, expected_rate)

    asian_result_current = asian_pricer.price_arithmetic_call(
        asian_strike, asian_maturity_years, 50000, seed=123
    )
    asian_result_future = asian_pricer_future.price_arithmetic_call(
        asian_strike, asian_maturity_years, 50000, seed=123
    )

    asian_change = asian_result_future.price - asian_result_current.price
    asian_change_pct = asian_change / asian_result_current.price * 100

    print("\n" + "-" * 50)
    print("Asian Call Option Impact")
    print("-" * 50)
    print(
        f"  Price at current rate ({config.risk_free_rate*100:.2f}%):  ${asian_result_current.price:.4f}"
    )
    print(
        f"  Price at expected rate ({expected_rate*100:.3f}%): ${asian_result_future.price:.4f}"
    )
    print(f"  Price change:                          ${asian_change:.4f} ({asian_change_pct:+.2f}%)")

    bates_model_future = BatesModel(
        result_bates_lewis.parameters,
        expected_rate,
        config.spot_price,
    )

    put_current = bates_model_pricing.price_put_lewis(put_strike, put_maturity_years)
    put_future = bates_model_future.price_put_lewis(put_strike, put_maturity_years)

    put_change = put_future - put_current
    put_change_pct = put_change / put_current * 100

    print("\n" + "-" * 50)
    print("European Put Option Impact")
    print("-" * 50)
    print(
        f"  Price at current rate ({config.risk_free_rate*100:.2f}%):  ${put_current:.4f}"
    )
    print(
        f"  Price at expected rate ({expected_rate*100:.3f}%): ${put_future:.4f}"
    )
    print(f"  Price change:                          ${put_change:.4f} ({put_change_pct:+.2f}%)")

    print("\n" + "=" * 70)
    print("                    FINAL PROJECT SUMMARY")
    print("=" * 70)

    print(
        """
================================================================================
                         OPTIONS PRICING PROJECT
                         EXECUTIVE SUMMARY REPORT
================================================================================
"""
    )

    print("1. HESTON MODEL CALIBRATION (15-Day Maturity)")
    print("-" * 60)
    print("   Pricing Method: Lewis (2001) Closed-Form Solution")
    print("   Calibrated Parameters:")
    print(f"      v0 (Initial Variance):     {heston_lewis.v0:.6f}")
    print(f"      kappa (Mean Reversion):    {heston_lewis.kappa:.6f}")
    print(f"      theta (Long-term Var):     {heston_lewis.theta:.6f}")
    print(f"      sigma (Vol of Vol):        {heston_lewis.sigma:.6f}")
    print(f"      rho (Correlation):         {heston_lewis.rho:.6f}")
    print(f"   Feller Ratio:                 {heston_lewis.feller_ratio():.4f}")
    print(f"   Calibration RMSE:             {result_heston_lewis.rmse:.6f}")

    print("\n2. BATES MODEL CALIBRATION (60-Day Maturity)")
    print("-" * 60)
    print("   Pricing Method: Lewis (2001) with Jump-Diffusion")
    print("   Stochastic Volatility Parameters:")
    print(f"      v0:     {bates_lewis.v0:.6f}")
    print(f"      kappa:  {bates_lewis.kappa:.6f}")
    print(f"      theta:  {bates_lewis.theta:.6f}")
    print(f"      sigma:  {bates_lewis.sigma:.6f}")
    print(f"      rho:    {bates_lewis.rho:.6f}")
    print("   Jump Parameters:")
    print(f"      lambda_j (Intensity):      {bates_lewis.lambda_j:.6f}")
    print(f"      mu_j (Mean Size):          {bates_lewis.mu_j:.6f}")
    print(f"      sigma_j (Size Vol):        {bates_lewis.sigma_j:.6f}")
    print(f"   Calibration RMSE:             {result_bates_lewis.rmse:.6f}")

    print("\n3. CIR MODEL CALIBRATION (Euribor Term Structure)")
    print("-" * 60)
    print("   Calibration Method: Cubic Spline Interpolation + MLE")
    print("   Calibrated Parameters:")
    print(f"      kappa (Mean Reversion):    {cir_params.kappa:.6f}")
    print(f"      theta (Long-term Rate):    {cir_params.theta:.6f} ({cir_params.theta*100:.3f}%)")
    print(f"      sigma (Volatility):        {cir_params.sigma:.6f}")
    print(f"      r0 (Initial Rate):         {cir_params.r0:.6f} ({cir_params.r0*100:.3f}%)")
    print(f"   Feller Ratio:                 {cir_params.feller_ratio():.4f}")
    print(f"   Calibration RMSE:             {cir_rmse:.8f}")

    print("\n4. DERIVATIVE PRICING RESULTS")
    print("-" * 60)
    print(f"   Asian Call Option (ATM, {asian_maturity_days} days):")
    print(f"      Fair Price:                ${asian_result.price:.4f}")
    print(f"      Standard Error:            ${asian_result.std_error:.4f}")
    print(
        f"      95% CI:                    [${asian_result.ci_lower:.4f}, ${asian_result.ci_upper:.4f}]"
    )
    print(f"      Bank Fee (4%):             ${fee_amount_asian:.4f}")
    print(f"      Client Price:              ${client_price_asian:.4f}")

    print(f"\n   European Put Option (95% moneyness, {put_maturity_days} days):")
    print(f"      Fair Price:                ${fair_price_put:.4f}")
    print(f"      Bank Fee (4%):             ${fee_amount:.4f}")
    print(f"      Client Price:              ${client_price_put:.4f}")

    print("\n5. INTEREST RATE FORECAST (12-Month Euribor)")
    print("-" * 60)
    print(f"   Current 12-Month Euribor:     {current_12m*100:.3f}%")
    print(f"   Expected Rate (1 Year):       {expected_rate*100:.3f}%")
    print(f"   Standard Deviation:           {std_rate*100:.3f}%")
    print(f"   95% Confidence Interval:      [{ci_95[0]*100:.3f}%, {ci_95[1]*100:.3f}%]")

    print("\n" + "=" * 70)
    print("                         END OF REPORT")
    print("=" * 70)

    heston_df = pd.DataFrame(
        {
            'Parameter': ['v0', 'kappa', 'theta', 'sigma', 'rho', 'Feller_Ratio', 'RMSE'],
            'Lewis_2001': [
                heston_lewis.v0,
                heston_lewis.kappa,
                heston_lewis.theta,
                heston_lewis.sigma,
                heston_lewis.rho,
                heston_lewis.feller_ratio(),
                result_heston_lewis.rmse,
            ],
            'Carr_Madan_1999': [
                heston_cm.v0,
                heston_cm.kappa,
                heston_cm.theta,
                heston_cm.sigma,
                heston_cm.rho,
                heston_cm.feller_ratio(),
                result_heston_cm.rmse,
            ],
        }
    )

    bates_df = pd.DataFrame(
        {
            'Parameter': [
                'v0',
                'kappa',
                'theta',
                'sigma',
                'rho',
                'lambda_j',
                'mu_j',
                'sigma_j',
                'Feller_Ratio',
                'RMSE',
            ],
            'Lewis_2001': [
                bates_lewis.v0,
                bates_lewis.kappa,
                bates_lewis.theta,
                bates_lewis.sigma,
                bates_lewis.rho,
                bates_lewis.lambda_j,
                bates_lewis.mu_j,
                bates_lewis.sigma_j,
                bates_lewis.feller_ratio(),
                result_bates_lewis.rmse,
            ],
        }
    )

    cir_df = pd.DataFrame(
        {
            'Parameter': ['kappa', 'theta', 'sigma', 'r0', 'Feller_Ratio', 'RMSE'],
            'Value': [
                cir_params.kappa,
                cir_params.theta,
                cir_params.sigma,
                cir_params.r0,
                cir_params.feller_ratio(),
                cir_rmse,
            ],
        }
    )

    pricing_df = pd.DataFrame(
        {
            'Instrument': ['Asian Call (ATM, 20d)', 'European Put (95%, 70d)'],
            'Fair_Price': [asian_result.price, fair_price_put],
            'Bank_Fee_4pct': [fee_amount_asian, fee_amount],
            'Client_Price': [client_price_asian, client_price_put],
        }
    )

    euribor_df = pd.DataFrame(
        {
            'Metric': ['Current_Rate', 'Expected_Rate_1Y', 'Std_Dev', 'CI_95_Lower', 'CI_95_Upper'],
            'Value_Percent': [
                current_12m * 100,
                expected_rate * 100,
                std_rate * 100,
                ci_95[0] * 100,
                ci_95[1] * 100,
            ],
        }
    )

    heston_df.to_csv('heston_calibration.csv', index=False)
    bates_df.to_csv('bates_calibration.csv', index=False)
    cir_df.to_csv('cir_calibration.csv', index=False)
    pricing_df.to_csv('pricing_results.csv', index=False)
    euribor_df.to_csv('euribor_forecast.csv', index=False)

    print("Results exported to CSV files:")
    print("  - heston_calibration.csv")
    print("  - bates_calibration.csv")
    print("  - cir_calibration.csv")
    print("  - pricing_results.csv")
    print("  - euribor_forecast.csv")

    print("\n" + "=" * 50)
    print("HESTON CALIBRATION RESULTS")
    print(heston_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("BATES CALIBRATION RESULTS")
    print(bates_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("CIR CALIBRATION RESULTS")
    print(cir_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("PRICING RESULTS")
    print(pricing_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("EURIBOR FORECAST")
    print(euribor_df.to_string(index=False))


if __name__ == '__main__':
    run_pipeline()
