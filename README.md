# Institutional Options Research Framework

## Abstract
This repository packages an academic, risk-first options research workflow. It combines the original stochastic-volatility pricing notebook with a compact options research layer for Black-Scholes-Merton benchmarks, defined-risk strategy construction, pre-trade risk gates, Monte Carlo scenario analysis, and paper-trade staging.

The project is designed for reproducible research and portfolio demonstration. It does not place live brokerage orders, does not provide investment advice, and deliberately blocks live execution in code.

## Research Principles
- **Reproducibility first**: deterministic sample option chains, fixed random seeds, and runnable scripts.
- **Defined risk first**: undefined short-option exposure is rejected by default.
- **Benchmark before belief**: every strategy is evaluated against transparent pricing and payoff assumptions.
- **Costs and liquidity matter**: bid/ask width, volume, and open interest are part of the risk gate.
- **No hidden execution**: the paper-trading ledger can stage approved paper trades only.

## Methodology Summary
- **Synthetic option market data**: Generated with Black-Scholes prices and a volatility smile across multiple maturities.
- **Heston model calibration**: Lewis (2001) characteristic-function pricing and RMSE-driven calibration to 15-day options.
- **Bates model calibration**: Lewis (2001) jump-diffusion extension calibrated to 60-day options.
- **Asian option pricing**: Monte Carlo simulation under the calibrated Heston model.
- **CIR model calibration**: Cubic-spline interpolation of Euribor rates and CIR parameter calibration to the resulting term structure.
- **Rate impact analysis**: Repricing Asian and European put options at the expected one-year Euribor rate.
- **Options research layer**: Black-Scholes-Merton pricing, Greeks, implied volatility inversion, vertical spreads, iron condors, risk controls, and paper-trade staging.

## Data Sources Used
- **Synthetic option prices** generated within the notebook via a Black-Scholes-based volatility smile.
- **Euribor term structure** rates embedded directly in the notebook (1 week, 1 month, 3 months, 6 months, 12 months).
- **Research option chain** generated deterministically in `src/options_research.py` for repeatable demonstrations.

## How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py
python scripts/run_options_research.py
python tests/test_options_research.py
```

`run_pipeline.py` reproduces the calibration, pricing, and reporting flow from the notebook. `scripts/run_options_research.py` evaluates defined-risk strategies and writes `reports/options_research_report.csv`.

## Repository Structure
```
.
├── .github/
│   └── workflows/
│       └── research-checks.yml
├── data/
├── figures/
├── notebooks/
│   └── Stochastic_Volatility_Option_Pricing_Framework.ipynb
├── reports/
├── scripts/
│   └── run_options_research.py
├── src/
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── calibration.py
│   ├── config.py
│   ├── data_loader.py
│   ├── fixed_calibration.py
│   ├── models.py
│   ├── options_research.py
│   ├── plots.py
│   ├── pricing.py
│   └── utils.py
├── tests/
│   └── test_options_research.py
├── environment.yml
├── requirements.txt
└── run_pipeline.py
```

## Outputs Produced
- `heston_calibration.csv`
- `bates_calibration.csv`
- `cir_calibration.csv`
- `pricing_results.csv`
- `euribor_forecast.csv`
- `reports/options_research_report.csv`

## Limitations
- Option market data are synthetically generated from a volatility smile rather than sourced from real market quotes.
- The workflow is calibrated to specific maturities (15-day Heston, 60-day Bates) and a fixed spot price/risk-free rate from the notebook.
- Monte Carlo results are subject to sampling error; confidence intervals are reported by the notebook logic.
- The options research layer is for education, research, and paper trading. It is not connected to a broker and cannot trade live capital.
- Scenario results are model-dependent and should not be interpreted as profit forecasts.

## References
- Black, F., and Scholes, M. (1973). The pricing of options and corporate liabilities.
- Merton, R. (1973). Theory of rational option pricing.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Bates, D. (1996). Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche mark options.
- Cox, J., Ingersoll, J., & Ross, S. (1985). A theory of the term structure of interest rates.
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.
- Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform.
