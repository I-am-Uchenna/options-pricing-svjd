"""Run the reproducible options research demonstration."""

from csv import DictWriter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.options_research import (
    PaperLedger,
    ResearchAssumptions,
    RiskLimits,
    analyze_strategy_risk,
    build_research_snapshot,
    evaluate_strategy,
    iron_condor,
    vertical_spread,
)


def main() -> None:
    market = build_research_snapshot()
    strategies = [
        vertical_spread(
            market,
            "call",
            expiry_days=45,
            long_strike=95,
            short_strike=105,
            thesis="Defined-risk bullish exposure when upside skew is acceptable.",
        ),
        vertical_spread(
            market,
            "put",
            expiry_days=45,
            long_strike=105,
            short_strike=95,
            thesis="Defined-risk bearish exposure with a known debit.",
        ),
        iron_condor(
            market,
            expiry_days=45,
            long_put=90,
            short_put=95,
            short_call=105,
            long_call=110,
        ),
    ]

    limits = RiskLimits()
    assumptions = ResearchAssumptions()
    ledger = PaperLedger(starting_cash=limits.account_equity)
    rows = []

    for strategy in strategies:
        risk = analyze_strategy_risk(strategy, limits)
        evaluation = evaluate_strategy(strategy, assumptions)
        if risk.approved:
            ledger.stage_trade(strategy, risk)
        rows.append(
            {
                "strategy": strategy.name,
                "decision": risk.decision,
                "entry_cash_flow": round(risk.entry_cash_flow, 2),
                "max_loss": round(risk.max_loss, 2),
                "max_profit": round(risk.max_profit, 2),
                "risk_fraction": round(risk.risk_fraction, 4),
                "expected_pnl": round(evaluation.expected_pnl, 2),
                "probability_of_profit": round(evaluation.probability_of_profit, 4),
                "var_95": round(evaluation.value_at_risk_95, 2),
                "cvar_95": round(evaluation.conditional_var_95, 2),
                "violations": "; ".join(risk.violations),
            }
        )

    output = Path("reports/options_research_report.csv")
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    print(f"paper trades staged: {len(ledger.trades)}")


if __name__ == "__main__":
    main()
