"""Stochastic volatility option pricing framework."""

from .config import ProjectConfig, config
from .data_loader import create_market_data
from .calibration import BatesCalibrator, CIRCalibrator, HestonCalibrator
from .models import (
    BatesModel,
    BatesParameters,
    CIRModel,
    CIRParameters,
    HestonModel,
    HestonParameters,
)
from .pricing import AsianOptionPricer, AsianPricingResult
from .black_scholes import (
    Greeks,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    probability_itm,
)
from .options_research import (
    LIVE_TRADING_ENABLED,
    LiveExecutionProhibited,
    OptionContract,
    OptionLeg,
    OptionMarketSnapshot,
    OptionStrategy,
    PaperLedger,
    ResearchAssumptions,
    RiskLimits,
    StrategyEvaluation,
    StrategyRiskReport,
    analyze_strategy_risk,
    build_research_snapshot,
    evaluate_strategy,
    iron_condor,
    place_live_order,
    vertical_spread,
)
from .plots import (
    plot_calibration_fit,
    plot_rate_distribution,
    plot_simulation_paths,
    plot_term_structure,
)
from .utils import configure_environment

__all__ = [
    "ProjectConfig",
    "config",
    "create_market_data",
    "BatesCalibrator",
    "CIRCalibrator",
    "HestonCalibrator",
    "BatesModel",
    "BatesParameters",
    "CIRModel",
    "CIRParameters",
    "HestonModel",
    "HestonParameters",
    "AsianOptionPricer",
    "AsianPricingResult",
    "Greeks",
    "black_scholes_greeks",
    "black_scholes_price",
    "implied_volatility",
    "probability_itm",
    "OptionContract",
    "OptionLeg",
    "OptionStrategy",
    "OptionMarketSnapshot",
    "build_research_snapshot",
    "LIVE_TRADING_ENABLED",
    "LiveExecutionProhibited",
    "PaperLedger",
    "place_live_order",
    "ResearchAssumptions",
    "StrategyEvaluation",
    "evaluate_strategy",
    "vertical_spread",
    "iron_condor",
    "RiskLimits",
    "StrategyRiskReport",
    "analyze_strategy_risk",
    "plot_calibration_fit",
    "plot_rate_distribution",
    "plot_simulation_paths",
    "plot_term_structure",
    "configure_environment",
]
