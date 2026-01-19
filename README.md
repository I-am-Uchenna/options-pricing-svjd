# Stochastic Volatility Option Pricing Framework

## Abstract
This repository packages the option pricing workflow implemented in the project notebook for SM Energy Company options and Euribor term-structure modeling. The implementation covers Heston (1993), Bates (1996), and CIR (1985) models, along with Monte Carlo pricing for an arithmetic-average Asian option and calibration to synthetic option data plus Euribor rates. The notebook is the source of truth for the modeling choices and outputs.

## Methodology Summary
- **Synthetic option market data**: Generated with Black-Scholes prices and a volatility smile across multiple maturities.
- **Heston model calibration**: Lewis (2001) characteristic-function pricing and RMSE-driven calibration to 15-day options.
- **Bates model calibration**: Lewis (2001) jump-diffusion extension calibrated to 60-day options.
- **Asian option pricing**: Monte Carlo simulation under the calibrated Heston model.
- **CIR model calibration**: Cubic-spline interpolation of Euribor rates and CIR parameter calibration to the resulting term structure.
- **Rate impact analysis**: Repricing Asian and European put options at the expected one-year Euribor rate.

## Data Sources Used
- **Synthetic option prices** generated within the notebook via a Black-Scholes-based volatility smile.
- **Euribor term structure** rates embedded directly in the notebook (1 week, 1 month, 3 months, 6 months, 12 months).

## How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py
```

The pipeline reproduces the calibration, pricing, and reporting flow from the notebook. It writes CSV outputs to the repository root using the same filenames as the notebook.

## Repository Structure
```
.
├── data/
├── figures/
├── notebooks/
│   └── Stochastic_Volatility_Option_Pricing_Framework.ipynb
├── reports/
├── src/
│   ├── __init__.py
│   ├── calibration.py
│   ├── config.py
│   ├── data_loader.py
│   ├── fixed_calibration.py
│   ├── models.py
│   ├── plots.py
│   ├── pricing.py
│   └── utils.py
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

## Limitations
- Option market data are synthetically generated from a volatility smile rather than sourced from real market quotes.
- The workflow is calibrated to specific maturities (15-day Heston, 60-day Bates) and a fixed spot price/risk-free rate from the notebook.
- Monte Carlo results are subject to sampling error; confidence intervals are reported by the notebook logic.

## References
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Bates, D. (1996). Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche mark options.
- Cox, J., Ingersoll, J., & Ross, S. (1985). A theory of the term structure of interest rates.
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes.
- Carr, P., & Madan, D. (1999). Option valuation using the fast Fourier transform.
