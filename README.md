# Stochastic Volatility Option Pricing Framework

A professional implementation of derivative pricing models including Heston (1993), Bates (1996), and Cox-Ingersoll-Ross (1985) for equity options and interest rate modeling.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Mathematical Framework](#mathematical-framework)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Calibration Results](#calibration-results)
7. [Pricing Results](#pricing-results)
8. [Project Structure](#project-structure)
9. [References](#references)
10. [License](#license)

---

## Overview

This project implements a comprehensive option pricing framework for a hypothetical client of an investment bank seeking to price OTC derivatives on SM Energy Company stock. The framework includes:

- **Heston Stochastic Volatility Model** for vanilla option pricing
- **Bates Jump-Diffusion Model** for capturing market discontinuities
- **CIR Interest Rate Model** for term structure modeling
- **Monte Carlo Methods** for path-dependent Asian options
- **FFT-based Pricing** using Carr-Madan (1999) methodology
- **Closed-form Solutions** using Lewis (2001) approach

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Underlying | SM Energy Company |
| Spot Price | $232.90 |
| Risk-free Rate | 1.50% |
| Trading Days/Year | 250 |
| Bank Fee | 4.00% |

---

## Features

### Model Implementations

- **Heston (1993)**: Stochastic volatility with mean-reverting variance process
- **Bates (1996)**: Heston model extended with log-normal jumps
- **CIR (1985)**: Mean-reverting short rate model for interest rates

### Pricing Methods

- Lewis (2001) closed-form characteristic function approach
- Carr-Madan (1999) Fast Fourier Transform method
- Monte Carlo simulation with variance reduction techniques

### Calibration

- Global optimization using differential evolution
- Local optimization using L-BFGS-B
- Feller condition enforcement for variance positivity
- Multi-start optimization for robustness

---

## Mathematical Framework

### Heston Model

The Heston (1993) model specifies the following risk-neutral dynamics:
dS(t) = r * S(t) * dt + sqrt(v(t)) * S(t) * dW_1(t)
dv(t) = kappa * (theta - v(t)) * dt + sigma * sqrt(v(t)) * dW_2(t)


where:
- `S(t)`: Asset price at time t
- `v(t)`: Instantaneous variance at time t
- `r`: Risk-free rate
- `kappa`: Mean reversion speed
- `theta`: Long-term variance
- `sigma`: Volatility of variance
- `rho`: Correlation between Brownian motions (dW_1 * dW_2 = rho * dt)

**Feller Condition**: To ensure variance remains positive: `2 * kappa * theta > sigma^2`

### Bates Model

The Bates (1996) model extends Heston with jump-diffusion:
dS(t)/S(t) = (r - lambda * k) * dt + sqrt(v(t)) * dW_1(t) + (e^J - 1) * dN(t)
dv(t) = kappa * (theta - v(t)) * dt + sigma * sqrt(v(t)) * dW_2(t)


where:
- `N(t)`: Poisson process with intensity `lambda`
- `J ~ N(mu_j, sigma_j^2)`: Log-normal jump size
- `k = E[e^J - 1]`: Jump compensator

### CIR Model

The Cox-Ingersoll-Ross (1985) model for short rates:
dr(t) = kappa * (theta - r(t)) * dt + sigma * sqrt(r(t)) * dW(t)


**Analytical Bond Price**:
P(t,T) = A(t,T) * exp(-B(t,T) * r(t))


### Asian Option Payoff

Arithmetic average Asian call:
Payoff = max(A - K, 0)

where `A = (1/n) * sum(S(t_i))` for i = 0, 1, ..., n

---

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0

### Setup

```bash
# Clone the repository
git clone https://github.com/I-am-Uchenna/options-pricing-svjd.git
cd options-pricing-svjd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
