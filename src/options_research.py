"""Risk-first options research and paper-trading primitives.

This module is deliberately compact: no broker integration, no hidden execution,
and no new dependencies. It supports reproducible research, defined-risk
strategy construction, pre-trade risk gates, and paper-trade staging only.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Literal

import numpy as np

from .black_scholes import black_scholes_price


OptionType = Literal["call", "put"]
Side = Literal["long", "short"]
FillConvention = Literal["mid", "natural"]
CONTRACT_MULTIPLIER = 100
LIVE_TRADING_ENABLED = False


class LiveExecutionProhibited(RuntimeError):
    """Raised when code attempts to place a real-money order."""


def place_live_order(*_args, **_kwargs) -> None:
    """Hard guardrail: this repository never sends live brokerage orders."""

    raise LiveExecutionProhibited(
        "Live execution is prohibited. This project supports research, "
        "order staging, and paper trading only."
    )


@dataclass(frozen=True)
class OptionContract:
    """Listed option quote used by the research engine."""

    symbol: str
    expiry_days: int
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    underlying_price: float
    implied_volatility: float
    open_interest: int
    volume: int

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.expiry_days <= 0 or self.strike <= 0 or self.underlying_price <= 0:
            raise ValueError("expiry, strike, and underlying price must be positive")
        if self.bid < 0 or self.ask < self.bid:
            raise ValueError("option quote must satisfy 0 <= bid <= ask")
        if self.implied_volatility <= 0:
            raise ValueError("implied_volatility must be positive")

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_pct_mid(self) -> float:
        return float("inf") if self.mid == 0 else (self.ask - self.bid) / self.mid

    def intrinsic_value(self, spot_at_expiry: float) -> float:
        if self.option_type == "call":
            return max(spot_at_expiry - self.strike, 0.0)
        return max(self.strike - spot_at_expiry, 0.0)


@dataclass(frozen=True)
class OptionLeg:
    """Signed position in one option contract."""

    contract: OptionContract
    side: Side
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.side not in {"long", "short"}:
            raise ValueError("side must be 'long' or 'short'")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

    @property
    def signed_quantity(self) -> int:
        return self.quantity if self.side == "long" else -self.quantity

    def entry_cash_flow(self, fill: FillConvention = "mid") -> float:
        if fill == "mid":
            price = self.contract.mid
        elif fill == "natural":
            price = self.contract.ask if self.side == "long" else self.contract.bid
        else:
            raise ValueError("fill must be 'mid' or 'natural'")
        return -self.signed_quantity * price * CONTRACT_MULTIPLIER

    def payoff_at_expiry(self, spot_at_expiry: float) -> float:
        return (
            self.signed_quantity
            * self.contract.intrinsic_value(spot_at_expiry)
            * CONTRACT_MULTIPLIER
        )


@dataclass(frozen=True)
class OptionStrategy:
    """Multi-leg options strategy with expiration payoff analytics."""

    name: str
    legs: tuple[OptionLeg, ...]
    thesis: str
    reference: str = ""

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("strategy must contain at least one leg")
        if len({leg.contract.symbol for leg in self.legs}) != 1:
            raise ValueError("all legs must use the same underlying symbol")

    @property
    def symbol(self) -> str:
        return self.legs[0].contract.symbol

    @property
    def underlying_price(self) -> float:
        return self.legs[0].contract.underlying_price

    @property
    def expiry_days(self) -> int:
        return max(leg.contract.expiry_days for leg in self.legs)

    def entry_cash_flow(self, fill: FillConvention = "mid") -> float:
        return sum(leg.entry_cash_flow(fill) for leg in self.legs)

    def profit_at_expiry(
        self, spot_at_expiry: float, fill: FillConvention = "mid"
    ) -> float:
        return self.entry_cash_flow(fill) + sum(
            leg.payoff_at_expiry(spot_at_expiry) for leg in self.legs
        )

    def scenario_grid(self) -> np.ndarray:
        spot = self.underlying_price
        strikes = [leg.contract.strike for leg in self.legs]
        lower = max(0.01, min([spot * 0.25, *strikes]) * 0.8)
        upper = max([spot * 2.0, *strikes]) * 1.2
        return np.linspace(lower, upper, 1200)

    def payoff_curve(self, spot_grid: Iterable[float]) -> np.ndarray:
        return np.array([self.profit_at_expiry(float(spot)) for spot in spot_grid])

    def max_loss(self) -> float:
        return float(-np.min(self.payoff_curve(self.scenario_grid())))

    def max_profit(self) -> float:
        return float(np.max(self.payoff_curve(self.scenario_grid())))

    def breakevens(self) -> tuple[float, ...]:
        spots = self.scenario_grid()
        pnl = self.payoff_curve(spots)
        roots: list[float] = []
        for left, right, pnl_left, pnl_right in zip(
            spots[:-1], spots[1:], pnl[:-1], pnl[1:]
        ):
            if pnl_left == 0:
                roots.append(float(left))
            elif pnl_left * pnl_right < 0:
                weight = abs(pnl_left) / (abs(pnl_left) + abs(pnl_right))
                roots.append(float(left + weight * (right - left)))
        return tuple(roots)

    def is_defined_risk(self) -> bool:
        for option_type in ("call", "put"):
            legs = [leg for leg in self.legs if leg.contract.option_type == option_type]
            for short in [leg for leg in legs if leg.side == "short"]:
                hedges = [
                    leg
                    for leg in legs
                    if leg.side == "long"
                    and leg.contract.expiry_days >= short.contract.expiry_days
                    and leg.quantity >= short.quantity
                ]
                if not hedges:
                    return False
        return True


@dataclass(frozen=True)
class OptionMarketSnapshot:
    """Single-underlying option-chain snapshot."""

    symbol: str
    spot: float
    rate: float
    contracts: tuple[OptionContract, ...]

    def find(
        self, option_type: OptionType, strike: float, expiry_days: int
    ) -> OptionContract:
        for contract in self.contracts:
            if (
                contract.option_type == option_type
                and contract.strike == strike
                and contract.expiry_days == expiry_days
            ):
                return contract
        raise KeyError(
            f"contract not found: {self.symbol} {expiry_days}D {strike} {option_type}"
        )


def volatility_smile(spot: float, strike: float, expiry_days: int) -> float:
    """Transparent stylized smile used for demonstrations and tests."""

    moneyness = strike / spot
    skew = max(1.0 - moneyness, 0.0) * 0.10
    wings = abs(moneyness - 1.0) * 0.18
    term = 0.02 if expiry_days <= 30 else 0.0
    return 0.24 + skew + wings + term


def build_research_snapshot(
    symbol: str = "SME",
    spot: float = 100.0,
    rate: float = 0.04,
    expiries: tuple[int, ...] = (30, 45, 60),
    strikes: tuple[float, ...] = (80, 85, 90, 95, 100, 105, 110, 115, 120),
) -> OptionMarketSnapshot:
    """Build a reproducible option chain with realistic bid/ask frictions."""

    # ponytail: synthetic chain ceiling, replace with OPRA or broker snapshots before market validation.
    contracts: list[OptionContract] = []
    for expiry_days in expiries:
        maturity = expiry_days / 365.0
        for strike in strikes:
            implied_vol = volatility_smile(spot, strike, expiry_days)
            for option_type in ("call", "put"):
                mid = black_scholes_price(
                    option_type, spot, strike, maturity, rate, implied_vol
                )
                spread = max(0.05, mid * 0.045)
                distance = abs(strike / spot - 1.0)
                open_interest = int(max(50, 1200 * (1.0 - min(distance, 0.7))))
                contracts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry_days=expiry_days,
                        strike=float(strike),
                        option_type=option_type,
                        bid=round(max(mid - 0.5 * spread, 0.01), 2),
                        ask=round(mid + 0.5 * spread, 2),
                        underlying_price=spot,
                        implied_volatility=round(implied_vol, 4),
                        open_interest=open_interest,
                        volume=max(10, int(0.18 * open_interest)),
                    )
                )
    return OptionMarketSnapshot(symbol, spot, rate, tuple(contracts))


def vertical_spread(
    market: OptionMarketSnapshot,
    option_type: OptionType,
    expiry_days: int,
    long_strike: float,
    short_strike: float,
    thesis: str,
    quantity: int = 1,
) -> OptionStrategy:
    """Build a defined-risk vertical spread."""

    return OptionStrategy(
        name=f"{market.symbol} {expiry_days}D {option_type} vertical",
        thesis=thesis,
        reference="Black-Scholes benchmark plus expiration payoff analysis.",
        legs=(
            OptionLeg(
                market.find(option_type, long_strike, expiry_days), "long", quantity
            ),
            OptionLeg(
                market.find(option_type, short_strike, expiry_days), "short", quantity
            ),
        ),
    )


def iron_condor(
    market: OptionMarketSnapshot,
    expiry_days: int,
    long_put: float,
    short_put: float,
    short_call: float,
    long_call: float,
    quantity: int = 1,
) -> OptionStrategy:
    """Build a defined-risk range-bound premium strategy."""

    if not long_put < short_put < short_call < long_call:
        raise ValueError("iron condor strikes must satisfy LP < SP < SC < LC")
    return OptionStrategy(
        name=f"{market.symbol} {expiry_days}D iron condor",
        thesis="Range-bound volatility risk premium with explicit tail hedges.",
        reference="Defined-risk short-volatility structure with scenario stress tests.",
        legs=(
            OptionLeg(market.find("put", long_put, expiry_days), "long", quantity),
            OptionLeg(market.find("put", short_put, expiry_days), "short", quantity),
            OptionLeg(market.find("call", short_call, expiry_days), "short", quantity),
            OptionLeg(market.find("call", long_call, expiry_days), "long", quantity),
        ),
    )


@dataclass(frozen=True)
class RiskLimits:
    """Portfolio-level limits used before any order can be staged."""

    account_equity: float = 100_000.0
    max_trade_risk_fraction: float = 0.01
    max_portfolio_risk_fraction: float = 0.06
    min_open_interest: int = 100
    min_volume: int = 10
    max_bid_ask_pct_mid: float = 0.15
    min_days_to_expiry: int = 7
    allow_undefined_risk: bool = False


@dataclass(frozen=True)
class StrategyRiskReport:
    """Risk assessment emitted by the pre-trade gate."""

    strategy_name: str
    approved: bool
    max_loss: float
    max_profit: float
    entry_cash_flow: float
    risk_fraction: float
    breakevens: tuple[float, ...]
    violations: tuple[str, ...]

    @property
    def decision(self) -> str:
        return "APPROVED_FOR_PAPER_TRADING" if self.approved else "REJECTED"


def analyze_strategy_risk(
    strategy: OptionStrategy,
    limits: RiskLimits,
    current_portfolio_risk: float = 0.0,
) -> StrategyRiskReport:
    """Apply deterministic pre-trade controls to an option strategy."""

    violations: list[str] = []
    max_loss = strategy.max_loss()
    max_profit = strategy.max_profit()
    if not strategy.is_defined_risk() and not limits.allow_undefined_risk:
        violations.append("undefined risk structures are prohibited")
    if max_loss > limits.account_equity * limits.max_trade_risk_fraction:
        violations.append("per-trade risk budget would be exceeded")
    if (
        current_portfolio_risk + max_loss
        > limits.account_equity * limits.max_portfolio_risk_fraction
    ):
        violations.append("portfolio risk budget would be exceeded")
    if strategy.expiry_days < limits.min_days_to_expiry:
        violations.append("expiry is too near for this research mandate")

    for leg in strategy.legs:
        contract = leg.contract
        if contract.open_interest < limits.min_open_interest:
            violations.append(
                f"{contract.option_type} {contract.strike:g} open interest is too low"
            )
        if contract.volume < limits.min_volume:
            violations.append(f"{contract.option_type} {contract.strike:g} volume is too low")
        if contract.spread_pct_mid > limits.max_bid_ask_pct_mid:
            violations.append(
                f"{contract.option_type} {contract.strike:g} bid/ask spread is too wide"
            )

    return StrategyRiskReport(
        strategy_name=strategy.name,
        approved=not violations,
        max_loss=max_loss,
        max_profit=max_profit,
        entry_cash_flow=strategy.entry_cash_flow(),
        risk_fraction=max_loss / limits.account_equity,
        breakevens=strategy.breakevens(),
        violations=tuple(dict.fromkeys(violations)),
    )


@dataclass(frozen=True)
class ResearchAssumptions:
    """Transparent assumptions for Monte Carlo terminal-price research."""

    annual_volatility: float = 0.25
    expected_return: float = 0.0
    num_scenarios: int = 20_000
    seed: int = 42


@dataclass(frozen=True)
class StrategyEvaluation:
    """Distributional evaluation of an options strategy."""

    strategy_name: str
    expected_pnl: float
    median_pnl: float
    probability_of_profit: float
    value_at_risk_95: float
    conditional_var_95: float
    best_case: float
    worst_case: float
    terminal_spot_mean: float


def evaluate_strategy(
    strategy: OptionStrategy,
    assumptions: ResearchAssumptions,
) -> StrategyEvaluation:
    """Evaluate terminal PnL under a transparent lognormal benchmark."""

    rng = np.random.default_rng(assumptions.seed)
    maturity = strategy.expiry_days / 365.0
    drift = (
        assumptions.expected_return - 0.5 * assumptions.annual_volatility**2
    ) * maturity
    shock = assumptions.annual_volatility * np.sqrt(maturity) * rng.standard_normal(
        assumptions.num_scenarios
    )
    terminal_spots = strategy.underlying_price * np.exp(drift + shock)
    pnl = np.array([strategy.profit_at_expiry(float(spot)) for spot in terminal_spots])
    losses = -pnl
    var_95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var_95]
    return StrategyEvaluation(
        strategy_name=strategy.name,
        expected_pnl=float(np.mean(pnl)),
        median_pnl=float(np.median(pnl)),
        probability_of_profit=float(np.mean(pnl > 0.0)),
        value_at_risk_95=var_95,
        conditional_var_95=float(np.mean(tail)) if len(tail) else var_95,
        best_case=float(np.max(pnl)),
        worst_case=float(np.min(pnl)),
        terminal_spot_mean=float(np.mean(terminal_spots)),
    )


@dataclass(frozen=True)
class PaperTrade:
    """Accepted paper trade with an immutable risk snapshot."""

    trade_id: str
    strategy_name: str
    opened_at: str
    entry_cash_flow: float
    max_loss: float
    max_profit: float
    risk_decision: str


@dataclass
class PaperLedger:
    """Simple cash ledger for approved paper trades."""

    starting_cash: float = 100_000.0
    cash: float = field(init=False)
    trades: list[PaperTrade] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.starting_cash

    def stage_trade(
        self, strategy: OptionStrategy, risk_report: StrategyRiskReport
    ) -> PaperTrade:
        if not risk_report.approved:
            raise ValueError("risk report rejected the strategy; paper trade blocked")
        trade = PaperTrade(
            trade_id=f"PAPER-{len(self.trades) + 1:04d}",
            strategy_name=strategy.name,
            opened_at=datetime.now(timezone.utc).isoformat(),
            entry_cash_flow=strategy.entry_cash_flow(),
            max_loss=risk_report.max_loss,
            max_profit=risk_report.max_profit,
            risk_decision=risk_report.decision,
        )
        self.cash += trade.entry_cash_flow
        self.trades.append(trade)
        return trade
