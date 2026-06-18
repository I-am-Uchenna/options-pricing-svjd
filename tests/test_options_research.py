"""Small self-checks for the options research layer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.black_scholes import black_scholes_price, implied_volatility
from src.options_research import (
    OptionLeg,
    OptionStrategy,
    PaperLedger,
    RiskLimits,
    LiveExecutionProhibited,
    analyze_strategy_risk,
    build_research_snapshot,
    place_live_order,
    vertical_spread,
)


def test_black_scholes_benchmark() -> None:
    call = black_scholes_price("call", 100, 100, 1.0, 0.05, 0.20)
    put = black_scholes_price("put", 100, 100, 1.0, 0.05, 0.20)
    assert abs(call - 10.4506) < 0.01
    assert abs(put - 5.5735) < 0.01
    assert (
        abs(implied_volatility("call", call, 100, 100, 1.0, 0.05) - 0.20)
        < 0.001
    )


def test_defined_risk_strategy_can_be_paper_staged() -> None:
    market = build_research_snapshot()
    strategy = vertical_spread(
        market,
        "call",
        expiry_days=45,
        long_strike=95,
        short_strike=105,
        thesis="Defined-risk bullish exposure.",
    )
    risk = analyze_strategy_risk(strategy, RiskLimits())
    assert risk.approved
    assert 0 < risk.max_loss < 1000
    assert risk.max_profit > 0
    trade = PaperLedger().stage_trade(strategy, risk)
    assert trade.risk_decision == "APPROVED_FOR_PAPER_TRADING"


def test_undefined_risk_and_live_execution_are_blocked() -> None:
    market = build_research_snapshot()
    naked_short = OptionStrategy(
        name="naked short call",
        thesis="This should be rejected.",
        legs=(OptionLeg(market.find("call", 105, 45), "short"),),
    )
    risk = analyze_strategy_risk(naked_short, RiskLimits())
    assert not risk.approved
    assert "undefined risk structures are prohibited" in risk.violations
    try:
        place_live_order()
    except LiveExecutionProhibited:
        pass
    else:
        raise AssertionError("live execution must remain impossible")


if __name__ == "__main__":
    test_black_scholes_benchmark()
    test_defined_risk_strategy_can_be_paper_staged()
    test_undefined_risk_and_live_execution_are_blocked()
