"""Explainable red-flag detection for company analysis.

Each detector should return structured :class:`RedFlag` objects instead of
booleans. This preserves the evidence needed by scoring and reporting layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from fintern.analysis.historical import HistoricalAnalysis
from fintern.data.periods import FiscalFrequency
from fintern.metrics._base import (
    FundamentalsInput,
    MetricCandidate,
    MetricScaffoldBase,
)
from fintern.metrics.profitability import Profitability
from fintern.metrics.risk import Risk

FlagSeverity = Literal["low", "medium", "high", "critical"]
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}

_REVENUE_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "Revenues"),
    ("income_statement", "SalesRevenueNet"),
    ("income_statement", "RevenueFromContractWithCustomerExcludingAssessedTax"),
)
_NET_INCOME_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "NetIncomeLoss"),
    ("income_statement", "ProfitLoss"),
)
_EPS_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("income_statement", "EarningsPerShareDiluted"),
    ("income_statement", "EarningsPerShareBasic"),
)
_OPERATING_CASH_FLOW_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "NetCashProvidedByUsedInOperatingActivities"),
)
_CAPEX_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("cash_flow", "PaymentsToAcquirePropertyPlantAndEquipment"),
    ("cash_flow", "PaymentsToAcquireProductiveAssets"),
)
_EQUITY_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "StockholdersEquity"),
    (
        "balance_sheet",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
)
_TOTAL_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligations"),
    ("balance_sheet", "DebtAndFinanceLeaseObligations"),
    ("balance_sheet", "LongTermDebt"),
)
_CURRENT_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    ("balance_sheet", "DebtAndFinanceLeaseObligationsCurrent"),
    ("balance_sheet", "LongTermDebtCurrentMaturities"),
    ("balance_sheet", "ShortTermBorrowings"),
    ("balance_sheet", "CommercialPaper"),
)
_NONCURRENT_DEBT_CANDIDATES: tuple[MetricCandidate, ...] = (
    ("balance_sheet", "LongTermDebtAndCapitalLeaseObligationsNoncurrent"),
    ("balance_sheet", "DebtAndFinanceLeaseObligationsNoncurrent"),
    ("balance_sheet", "LongTermDebtNoncurrent"),
)


@dataclass(frozen=True)
class RedFlagThresholds:
    """Default thresholds used by red-flag detectors.

    Values are deliberately centralized here so applications can configure the
    policy without changing detector code.
    """

    consecutive_declines: int = 2
    margin_decline: float = 0.02
    leverage_increase: float = 0.10
    earnings_cash_flow_gap: float = 0.15
    annualized_volatility: float = 0.40
    maximum_drawdown: float = -0.30
    stale_fundamentals_days: int = 180

    def __post_init__(self) -> None:
        if self.consecutive_declines < 1:
            raise ValueError("consecutive_declines must be strictly positive")

        positive_thresholds = {
            "margin_decline": self.margin_decline,
            "leverage_increase": self.leverage_increase,
            "earnings_cash_flow_gap": self.earnings_cash_flow_gap,
            "annualized_volatility": self.annualized_volatility,
        }

        for name, value in positive_thresholds.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")

        if not -1 <= self.maximum_drawdown <= 0:
            raise ValueError("maximum_drawdown must be between -1 and 0")

        if self.stale_fundamentals_days < 1:
            raise ValueError("stale_fundamentals_days must be strictly positive")


@dataclass(frozen=True)
class RedFlag:
    """One explainable warning and the evidence supporting it."""

    code: str
    severity: FlagSeverity
    message: str
    metric: str | None = None
    value: float | None = None
    reference_value: float | None = None
    threshold: float | None = None
    period_start: pd.Timestamp | None = None
    period_end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        normalized_code = self.code.strip().lower().replace(" ", "_")

        if not normalized_code:
            raise ValueError("code cannot be empty")

        if self.severity not in _VALID_SEVERITIES:
            allowed = ", ".join(sorted(_VALID_SEVERITIES))
            raise ValueError(f"severity must be one of: {allowed}")

        if not self.message.strip():
            raise ValueError("message cannot be empty")

        for name in ("value", "reference_value", "threshold"):
            value = getattr(self, name)

            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

        start = (
            pd.Timestamp(self.period_start) if self.period_start is not None else None
        )
        end = pd.Timestamp(self.period_end) if self.period_end is not None else None

        if start is not None and end is not None and start > end:
            raise ValueError("period_start must be before or equal to period_end")

        object.__setattr__(self, "code", normalized_code)
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "period_start", start)
        object.__setattr__(self, "period_end", end)

    def as_dict(self) -> dict[str, object]:
        """Return a record suitable for DataFrame or report construction."""
        return asdict(self)


@dataclass(frozen=True)
class RedFlagAnalysis(MetricScaffoldBase):
    """Detect explainable company-level warning signals.

    Detectors should return an empty list when enough data exists and no warning
    is present. Missing required data should raise ``ValueError`` so callers can
    distinguish "not detected" from "not measurable".
    """

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None
    as_of: str | pd.Timestamp | None = None
    frequency: FiscalFrequency = "quarterly"
    thresholds: RedFlagThresholds = RedFlagThresholds()

    CHECKS: ClassVar[tuple[str, ...]] = (
        "declining_revenue",
        "declining_earnings",
        "negative_free_cash_flow",
        "margin_deterioration",
        "earnings_cash_flow_divergence",
        "rising_leverage",
        "high_market_risk",
        "stale_fundamentals",
    )

    def _historical(self) -> HistoricalAnalysis:
        """Build a historical-analysis view over the same validated inputs."""
        return HistoricalAnalysis(
            ticker=self.ticker,
            prices=self.prices,
            data=self.data,
            fundamentals=self.fundamentals,
            as_of=self.as_of,
            frequency=self.frequency,
        )

    def _series(self, candidates: tuple[MetricCandidate, ...]) -> pd.Series:
        """Return the first available normalized fundamentals series."""
        return self._fundamental_metric_series_from_candidates(
            candidates,
            date_column="period_end",
            frequency=self.frequency,
        )

    @staticmethod
    def _align(**series: pd.Series) -> pd.DataFrame:
        """Align named financial series on common reporting periods."""
        aligned = pd.concat(series, axis=1, join="inner").dropna()

        if aligned.empty:
            names = ", ".join(series)
            raise ValueError(f"No aligned fundamentals periods found for: {names}")

        return aligned

    def _total_debt(self) -> pd.Series:
        """Return direct total debt or derive it from current components."""
        try:
            return self._series(_TOTAL_DEBT_CANDIDATES)
        except ValueError:
            aligned = self._align(
                current_debt=self._series(_CURRENT_DEBT_CANDIDATES),
                noncurrent_debt=self._series(_NONCURRENT_DEBT_CANDIDATES),
            )
            total_debt = aligned["current_debt"] + aligned["noncurrent_debt"]
            total_debt.name = "total_debt"
            return total_debt

    @staticmethod
    def _result(
        *,
        code: str,
        severity: FlagSeverity,
        message: str,
        metric: str | None = None,
        value: float | None = None,
        reference_value: float | None = None,
        threshold: float | None = None,
        period_start: pd.Timestamp | None = None,
        period_end: pd.Timestamp | None = None,
    ) -> RedFlag:
        """Create one validated red-flag result."""
        return RedFlag(
            code=code,
            severity=severity,
            message=message,
            metric=metric,
            value=value,
            reference_value=reference_value,
            threshold=threshold,
            period_start=period_start,
            period_end=period_end,
        )

    @staticmethod
    def to_frame(flags: list[RedFlag]) -> pd.DataFrame:
        """Convert detector output into a stable analysis table."""
        columns = [
            "code",
            "severity",
            "message",
            "metric",
            "value",
            "reference_value",
            "threshold",
            "period_start",
            "period_end",
        ]

        if not isinstance(flags, list) or any(
            not isinstance(flag, RedFlag) for flag in flags
        ):
            raise TypeError("flags must be a list of RedFlag objects")

        return pd.DataFrame([flag.as_dict() for flag in flags], columns=columns)

    def declining_revenue(self) -> list[RedFlag]:
        """Detect consecutive revenue declines."""
        revenue = self._series(_REVENUE_CANDIDATES)
        required_changes = self.thresholds.consecutive_declines

        if len(revenue) <= required_changes:
            raise ValueError(
                "declining_revenue requires one more revenue observation than "
                "the configured consecutive declines"
            )

        prior = revenue.shift(1)

        if (prior.dropna() == 0).any():
            raise ValueError("revenue growth is undefined when a prior value is zero")

        changes = revenue.pct_change(fill_method=None).dropna()
        recent = changes.iloc[-required_changes:]

        if not (recent < 0).all():
            return []

        start_value = float(revenue.iloc[-required_changes - 1])
        latest_value = float(revenue.iloc[-1])
        cumulative_decline = latest_value / start_value - 1
        severity: FlagSeverity = "high" if cumulative_decline <= -0.20 else "medium"
        return [
            self._result(
                code="declining_revenue",
                severity=severity,
                message=(
                    f"Revenue declined for {required_changes} consecutive periods."
                ),
                metric="revenue_growth",
                value=cumulative_decline,
                reference_value=0.0,
                threshold=0.0,
                period_start=revenue.index[-required_changes - 1],
                period_end=revenue.index[-1],
            )
        ]

    def declining_earnings(self) -> list[RedFlag]:
        """Detect consecutive net-income or EPS deterioration."""
        try:
            earnings = self._series(_NET_INCOME_CANDIDATES)
        except ValueError:
            earnings = self._series(_EPS_CANDIDATES)

        required_changes = self.thresholds.consecutive_declines

        if len(earnings) <= required_changes:
            raise ValueError(
                "declining_earnings requires one more earnings observation than "
                "the configured consecutive declines"
            )

        recent_values = earnings.iloc[-required_changes:]
        period_start = earnings.index[-required_changes - 1]
        period_end = earnings.index[-1]
        latest = float(earnings.iloc[-1])
        previous = float(earnings.iloc[-2])

        if (recent_values < 0).all():
            return [
                self._result(
                    code="recurring_losses",
                    severity="high",
                    message=(
                        f"Earnings were negative for {required_changes} "
                        "consecutive periods."
                    ),
                    metric=earnings.name,
                    value=latest,
                    reference_value=previous,
                    threshold=0.0,
                    period_start=period_start,
                    period_end=period_end,
                )
            ]

        recent_differences = earnings.diff().iloc[-required_changes:]

        if not (recent_differences < 0).all():
            return []

        severity: FlagSeverity = "high" if latest < 0 else "medium"
        return [
            self._result(
                code="declining_earnings",
                severity=severity,
                message=(
                    f"Earnings declined for {required_changes} consecutive periods."
                ),
                metric=earnings.name,
                value=latest,
                reference_value=previous,
                threshold=0.0,
                period_start=period_start,
                period_end=period_end,
            )
        ]

    def negative_free_cash_flow(self) -> list[RedFlag]:
        """Detect current or persistent negative free cash flow."""
        aligned = self._align(
            operating_cash_flow=self._series(_OPERATING_CASH_FLOW_CANDIDATES),
            capex=self._series(_CAPEX_CANDIDATES),
        )
        free_cash_flow = aligned["operating_cash_flow"] - aligned["capex"].abs()

        if float(free_cash_flow.iloc[-1]) >= 0:
            return []

        required_periods = self.thresholds.consecutive_declines
        consecutive_negative = (
            len(free_cash_flow) >= required_periods
            and (free_cash_flow.iloc[-required_periods:] < 0).all()
        )
        severity: FlagSeverity = "high" if consecutive_negative else "medium"
        period_start = (
            free_cash_flow.index[-required_periods]
            if consecutive_negative
            else free_cash_flow.index[-1]
        )
        return [
            self._result(
                code="negative_free_cash_flow",
                severity=severity,
                message=(
                    "Free cash flow is persistently negative."
                    if consecutive_negative
                    else "Free cash flow was negative in the latest period."
                ),
                metric="free_cash_flow",
                value=float(free_cash_flow.iloc[-1]),
                reference_value=(
                    float(free_cash_flow.iloc[-2]) if len(free_cash_flow) >= 2 else None
                ),
                threshold=0.0,
                period_start=period_start,
                period_end=free_cash_flow.index[-1],
            )
        ]

    def margin_deterioration(self) -> list[RedFlag]:
        """Detect material deterioration in operating and net margins."""
        profitability = Profitability(
            ticker=self.ticker,
            prices=self.prices,
            data=self.data,
            fundamentals=self.fundamentals,
            as_of=self.as_of,
            frequency=self.frequency,
        )
        margin_methods = (
            ("operating_margin", profitability.operating_margin),
            ("net_margin", profitability.net_margin),
        )
        flags: list[RedFlag] = []
        errors: list[ValueError] = []
        evaluated = 0

        for metric, method in margin_methods:
            try:
                margins = method()
            except ValueError as exc:
                errors.append(exc)
                continue

            if len(margins) < 2:
                errors.append(ValueError(f"{metric} requires at least two periods"))
                continue

            evaluated += 1
            latest = float(margins.iloc[-1])
            previous = float(margins.iloc[-2])
            decline = latest - previous

            if decline > -self.thresholds.margin_decline:
                continue

            severity: FlagSeverity = (
                "high" if decline <= -2 * self.thresholds.margin_decline else "medium"
            )
            flags.append(
                self._result(
                    code=f"{metric}_deterioration",
                    severity=severity,
                    message=(
                        f"{metric.replace('_', ' ').title()} declined by "
                        f"{abs(decline):.2%}."
                    ),
                    metric=metric,
                    value=latest,
                    reference_value=previous,
                    threshold=self.thresholds.margin_decline,
                    period_start=margins.index[-2],
                    period_end=margins.index[-1],
                )
            )

        if evaluated == 0:
            raise ValueError("No margin series could be evaluated") from (
                errors[-1] if errors else None
            )

        return flags

    def earnings_cash_flow_divergence(self) -> list[RedFlag]:
        """Detect earnings growth unsupported by operating cash flow."""
        aligned = self._align(
            earnings=self._series(_NET_INCOME_CANDIDATES),
            operating_cash_flow=self._series(_OPERATING_CASH_FLOW_CANDIDATES),
        )

        if len(aligned) < 2:
            raise ValueError(
                "earnings_cash_flow_divergence requires at least two periods"
            )

        previous = aligned.iloc[-2]
        latest = aligned.iloc[-1]

        if previous["earnings"] <= 0 or previous["operating_cash_flow"] <= 0:
            raise ValueError(
                "earnings and operating cash flow must have positive prior values"
            )

        earnings_growth = float(latest["earnings"] / previous["earnings"] - 1)
        cash_flow_growth = float(
            latest["operating_cash_flow"] / previous["operating_cash_flow"] - 1
        )
        gap = earnings_growth - cash_flow_growth

        if gap < self.thresholds.earnings_cash_flow_gap:
            return []

        severity: FlagSeverity = "high" if cash_flow_growth < 0 else "medium"
        return [
            self._result(
                code="earnings_cash_flow_divergence",
                severity=severity,
                message=(
                    f"Earnings growth exceeded operating-cash-flow growth by {gap:.2%}."
                ),
                metric="earnings_vs_operating_cash_flow_growth",
                value=earnings_growth,
                reference_value=cash_flow_growth,
                threshold=self.thresholds.earnings_cash_flow_gap,
                period_start=aligned.index[-2],
                period_end=aligned.index[-1],
            )
        ]

    def rising_leverage(self) -> list[RedFlag]:
        """Detect a material increase in debt relative to equity."""
        aligned = self._align(
            debt=self._total_debt(),
            equity=self._series(_EQUITY_CANDIDATES),
        )
        latest = aligned.iloc[-1]
        previous = aligned.iloc[-2] if len(aligned) >= 2 else None

        if latest["equity"] <= 0:
            return [
                self._result(
                    code="non_positive_equity",
                    severity="critical",
                    message="Shareholders' equity is zero or negative.",
                    metric="shareholders_equity",
                    value=float(latest["equity"]),
                    reference_value=(
                        float(previous["equity"]) if previous is not None else None
                    ),
                    threshold=0.0,
                    period_start=(
                        aligned.index[-2] if previous is not None else aligned.index[-1]
                    ),
                    period_end=aligned.index[-1],
                )
            ]

        if previous is None:
            raise ValueError("rising_leverage requires at least two aligned periods")

        if previous["equity"] <= 0:
            return []

        if latest["debt"] < 0 or previous["debt"] < 0:
            raise ValueError("debt values cannot be negative")

        latest_ratio = float(latest["debt"] / latest["equity"])
        previous_ratio = float(previous["debt"] / previous["equity"])

        if previous_ratio == 0:
            if latest_ratio == 0:
                return []
            relative_increase = latest_ratio
        else:
            relative_increase = latest_ratio / previous_ratio - 1

        if relative_increase < self.thresholds.leverage_increase:
            return []

        severity: FlagSeverity = (
            "high"
            if latest_ratio >= 2
            or relative_increase >= 2 * self.thresholds.leverage_increase
            else "medium"
        )
        return [
            self._result(
                code="rising_leverage",
                severity=severity,
                message=(
                    "Debt-to-equity increased from "
                    f"{previous_ratio:.2f} to {latest_ratio:.2f}."
                ),
                metric="debt_to_equity",
                value=latest_ratio,
                reference_value=previous_ratio,
                threshold=self.thresholds.leverage_increase,
                period_start=aligned.index[-2],
                period_end=aligned.index[-1],
            )
        ]

    def high_market_risk(self) -> list[RedFlag]:
        """Detect high annualized volatility or severe drawdown."""
        prices = self._close_prices()
        risk = Risk(
            ticker=self.ticker,
            prices=prices,
            data=self.data if self.data is not None else pd.DataFrame(),
        )
        annualized_volatility = risk.annualized_volatility()
        drawdowns = prices.div(prices.cummax()).sub(1)
        maximum_drawdown = float(drawdowns.min())
        flags: list[RedFlag] = []
        has_datetime_index = isinstance(prices.index, pd.DatetimeIndex)
        period_start = prices.index[0] if has_datetime_index else None
        period_end = prices.index[-1] if has_datetime_index else None

        if annualized_volatility >= self.thresholds.annualized_volatility:
            severity: FlagSeverity = (
                "high"
                if annualized_volatility >= 2 * self.thresholds.annualized_volatility
                else "medium"
            )
            flags.append(
                self._result(
                    code="high_annualized_volatility",
                    severity=severity,
                    message=(
                        f"Annualized volatility reached {annualized_volatility:.2%}."
                    ),
                    metric="annualized_volatility",
                    value=annualized_volatility,
                    threshold=self.thresholds.annualized_volatility,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        if maximum_drawdown <= self.thresholds.maximum_drawdown:
            severity = (
                "high"
                if maximum_drawdown <= 1.5 * self.thresholds.maximum_drawdown
                else "medium"
            )
            flags.append(
                self._result(
                    code="severe_drawdown",
                    severity=severity,
                    message=f"Maximum drawdown reached {maximum_drawdown:.2%}.",
                    metric="maximum_drawdown",
                    value=maximum_drawdown,
                    threshold=self.thresholds.maximum_drawdown,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        return flags

    def stale_fundamentals(
        self,
        as_of: str | pd.Timestamp | None = None,
    ) -> list[RedFlag]:
        """Detect fundamentals older than the configured maximum age."""
        if as_of is None and self.as_of is not None:
            as_of_date = pd.Timestamp(self.as_of)
        elif as_of is None:
            try:
                prices = self._close_prices()
                if not isinstance(prices.index, pd.DatetimeIndex):
                    raise ValueError("prices must use a DatetimeIndex")
                as_of_date = prices.index.max().normalize()
            except ValueError:
                as_of_date = pd.Timestamp.today().normalize()
        else:
            as_of_date = pd.Timestamp(as_of)

        if pd.isna(as_of_date):
            raise ValueError("as_of must be a valid date")

        rows = self._ticker_statements()

        if "filed_date" not in rows.columns:
            raise ValueError("fundamentals statements must contain `filed_date`")

        filed_dates = pd.to_datetime(rows["filed_date"], errors="coerce").dropna()
        filed_dates = filed_dates.loc[filed_dates <= as_of_date]

        if filed_dates.empty:
            raise ValueError(
                f"No fundamentals were available as of {as_of_date.date()}"
            )

        latest_filing = filed_dates.max()
        age_days = int((as_of_date.normalize() - latest_filing.normalize()).days)

        if age_days <= self.thresholds.stale_fundamentals_days:
            return []

        severity: FlagSeverity = (
            "medium"
            if age_days > 2 * self.thresholds.stale_fundamentals_days
            else "low"
        )
        return [
            self._result(
                code="stale_fundamentals",
                severity=severity,
                message=f"Latest fundamentals are {age_days} days old.",
                metric="fundamentals_age_days",
                value=float(age_days),
                threshold=float(self.thresholds.stale_fundamentals_days),
                period_start=latest_filing,
                period_end=as_of_date,
            )
        ]

    def run_all(self, strict: bool = False) -> list[RedFlag]:
        """Run all detectors and sort warnings from critical to low.

        By default, detectors lacking required inputs are skipped. ``strict=True``
        propagates their ``ValueError`` for data-quality diagnostics.
        """
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")

        flags: list[RedFlag] = []

        for check_name in self.CHECKS:
            check = getattr(self, check_name)

            try:
                flags.extend(check())
            except ValueError:
                if strict:
                    raise

        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(flags, key=lambda flag: severity_rank[flag.severity])


__all__ = [
    "FlagSeverity",
    "RedFlag",
    "RedFlagAnalysis",
    "RedFlagThresholds",
]
