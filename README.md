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
