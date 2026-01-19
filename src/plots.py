import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_fit(strikes, market_prices, model_prices, option_types, title):
    """Plot calibration fit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"call": "#2E86AB", "put": "#A23B72"}

    ax1 = axes[0]
    for opt_type in ["call", "put"]:
        mask = np.array([t.lower() == opt_type for t in option_types])
        if np.any(mask):
            ax1.scatter(
                strikes[mask],
                market_prices[mask],
                c=colors[opt_type],
                marker="o",
                s=60,
                alpha=0.7,
                label=f"Market {opt_type.title()}s",
                edgecolors="white",
            )
            ax1.scatter(
                strikes[mask],
                model_prices[mask],
                c=colors[opt_type],
                marker="x",
                s=60,
                label=f"Model {opt_type.title()}s",
            )

    ax1.set_xlabel("Strike Price (USD)")
    ax1.set_ylabel("Option Price (USD)")
    ax1.set_title("Market vs Model Prices")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    errors = (model_prices - market_prices) / market_prices * 100

    bar_colors = [
        colors["call"] if t.lower() == "call" else colors["put"] for t in option_types
    ]
    ax2.bar(range(len(errors)), errors, color=bar_colors, alpha=0.7, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("Option Index")
    ax2.set_ylabel("Pricing Error (%)")
    ax2.set_title("Relative Pricing Errors")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_term_structure(maturities, market_rates, model_rates, title):
    """Plot term structure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        maturities * 12,
        market_rates * 100,
        "o-",
        color="#2E86AB",
        markersize=8,
        linewidth=2,
        label="Market Rates",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    ax.plot(
        maturities * 12,
        model_rates * 100,
        "s--",
        color="#A23B72",
        markersize=7,
        linewidth=2,
        label="CIR Model",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    ax.set_xlabel("Maturity (Months)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rate_distribution(rates, current_rate, title):
    """Plot rate distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(rates * 100, bins=50, density=True, alpha=0.7, color="#2E86AB", edgecolor="white")

    mean_rate = np.mean(rates)
    ci_lower = np.percentile(rates, 2.5)
    ci_upper = np.percentile(rates, 97.5)

    ax.axvline(
        current_rate * 100,
        color="#C73E1D",
        linewidth=2,
        linestyle="--",
        label=f"Current: {current_rate*100:.2f}%",
    )
    ax.axvline(
        mean_rate * 100,
        color="#F18F01",
        linewidth=2,
        label=f"Expected: {mean_rate*100:.2f}%",
    )
    ax.axvline(ci_lower * 100, color="#A23B72", linewidth=1.5, linestyle=":")
    ax.axvline(
        ci_upper * 100,
        color="#A23B72",
        linewidth=1.5,
        linestyle=":",
        label=f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
    )

    ax.axvspan(ci_lower * 100, ci_upper * 100, alpha=0.2, color="#A23B72")

    ax.set_xlabel("Rate (%)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_simulation_paths(time_grid, paths, title, ylabel="Value"):
    """Plot simulation paths."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_show = min(200, paths.shape[0])
    indices = np.random.choice(paths.shape[0], n_show, replace=False)

    for idx in indices:
        ax.plot(time_grid, paths[idx, :], alpha=0.1, color="#2E86AB", linewidth=0.5)

    p5 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    ax.plot(time_grid, p50, color="#C73E1D", linewidth=2.5, label="Median")
    ax.plot(
        time_grid,
        p5,
        color="#F18F01",
        linewidth=1.5,
        linestyle="--",
        label="5th percentile",
    )
    ax.plot(
        time_grid,
        p95,
        color="#F18F01",
        linewidth=1.5,
        linestyle="--",
        label="95th percentile",
    )

    ax.set_xlabel("Time (Years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
