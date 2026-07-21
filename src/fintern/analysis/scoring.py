"""Transparent company scoring built from Fintern metrics and red flags.

Scoring must preserve raw values, normalization rules, weights, missing-data
status, and explanations. Final score should never be an unexplained number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fintern.analysis.red_flags import RedFlag, RedFlagAnalysis
from fintern.data.periods import FiscalFrequency
from fintern.metrics._base import (
    FundamentalsInput,
    MetricCandidate,
    MetricScaffoldBase,
)
from fintern.metrics.growth import Growth
from fintern.metrics.profitability import Profitability
from fintern.metrics.risk import Risk
from fintern.metrics.valuation import Valuation

ScoreCategory = Literal[
    "profitability",
    "growth",
    "valuation",
    "risk",
    "momentum",
]
ScoreStatus = Literal["scored", "missing", "invalid"]
ScoreDirection = Literal["higher_is_better", "lower_is_better"]

_VALID_CATEGORIES = {
    "profitability",
    "growth",
    "valuation",
    "risk",
    "momentum",
}
_VALID_STATUSES = {"scored", "missing", "invalid"}
_VALID_GRADES = {"A", "B", "C", "D", "E"}

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


@dataclass(frozen=True)
class ScoringConfig:
    """Weights, coverage requirement, and red-flag penalties."""

    profitability_weight: float = 0.25
    growth_weight: float = 0.20
    valuation_weight: float = 0.20
    risk_weight: float = 0.20
    momentum_weight: float = 0.15
    minimum_coverage: float = 0.60
    low_flag_penalty: float = 1.0
    medium_flag_penalty: float = 3.0
    high_flag_penalty: float = 7.0
    critical_flag_penalty: float = 15.0

    def __post_init__(self) -> None:
        values = self.category_weights()

        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("category weights must be finite and non-negative")

        if sum(values.values()) <= 0:
            raise ValueError("at least one category weight must be positive")

        if not 0 <= self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be between 0 and 1")

        penalties = (
            self.low_flag_penalty,
            self.medium_flag_penalty,
            self.high_flag_penalty,
            self.critical_flag_penalty,
        )

        if any(not np.isfinite(value) or value < 0 for value in penalties):
            raise ValueError("red-flag penalties must be finite and non-negative")

        if penalties != tuple(sorted(penalties)):
            raise ValueError("red-flag penalties must increase with severity")

    def category_weights(self) -> dict[ScoreCategory, float]:
        """Return configured category weights."""
        return {
            "profitability": self.profitability_weight,
            "growth": self.growth_weight,
            "valuation": self.valuation_weight,
            "risk": self.risk_weight,
            "momentum": self.momentum_weight,
        }


@dataclass(frozen=True)
class ScoreComponent:
    """One auditable metric contribution to company score."""

    category: ScoreCategory
    metric: str
    value: float | None
    score: float | None
    weight: float
    explanation: str
    status: ScoreStatus = "scored"

    def __post_init__(self) -> None:
        metric = self.metric.strip()
        explanation = self.explanation.strip()

        if self.category not in _VALID_CATEGORIES:
            allowed = ", ".join(sorted(_VALID_CATEGORIES))
            raise ValueError(f"category must be one of: {allowed}")

        if not metric:
            raise ValueError("metric cannot be empty")

        if self.status not in _VALID_STATUSES:
            allowed = ", ".join(sorted(_VALID_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")

        if not explanation:
            raise ValueError("explanation cannot be empty")

        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and non-negative")

        if self.value is not None and not np.isfinite(self.value):
            raise ValueError("value must be finite when provided")

        if self.status == "scored":
            if self.score is None or not np.isfinite(self.score):
                raise ValueError("scored components require a finite score")

            if not 0 <= self.score <= 100:
                raise ValueError("score must be between 0 and 100")
        elif self.score is not None:
            raise ValueError("missing or invalid components cannot have a score")

        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "explanation", explanation)

    @property
    def contribution(self) -> float | None:
        """Return weighted points, or ``None`` when component is unscored."""
        if self.score is None:
            return None

        return float(self.score * self.weight)

    def as_dict(self) -> dict[str, object]:
        """Return component record including weighted contribution."""
        record = asdict(self)
        record["contribution"] = self.contribution
        return record


@dataclass(frozen=True)
class CompanyScore:
    """Final explainable company score and supporting evidence."""

    ticker: str
    total_score: float | None
    grade: str | None
    coverage: float
    penalty: float
    components: tuple[ScoreComponent, ...]
    red_flags: tuple[RedFlag, ...] = ()

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper()

        if not ticker:
            raise ValueError("ticker cannot be empty")

        if not np.isfinite(self.coverage) or not 0 <= self.coverage <= 1:
            raise ValueError("coverage must be between 0 and 1")

        if not np.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("penalty must be finite and non-negative")

        if any(not isinstance(item, ScoreComponent) for item in self.components):
            raise TypeError("components must contain ScoreComponent objects")

        if any(not isinstance(item, RedFlag) for item in self.red_flags):
            raise TypeError("red_flags must contain RedFlag objects")

        if self.total_score is None:
            if self.grade is not None:
                raise ValueError("grade must be None when total_score is unavailable")
        else:
            if not np.isfinite(self.total_score) or not 0 <= self.total_score <= 100:
                raise ValueError("total_score must be between 0 and 100")

            if self.grade not in _VALID_GRADES:
                allowed = ", ".join(sorted(_VALID_GRADES))
                raise ValueError(f"grade must be one of: {allowed}")

        object.__setattr__(self, "ticker", ticker)

    def component_frame(self) -> pd.DataFrame:
        """Return component-level audit table."""
        columns = [
            "category",
            "metric",
            "value",
            "score",
            "weight",
            "explanation",
            "status",
            "contribution",
        ]
        return pd.DataFrame(
            [component.as_dict() for component in self.components],
            columns=columns,
        )

    def as_dict(self) -> dict[str, object]:
        """Return nested result suitable for reports or JSON conversion."""
        return {
            "ticker": self.ticker,
            "total_score": self.total_score,
            "grade": self.grade,
            "coverage": self.coverage,
            "penalty": self.penalty,
            "components": [item.as_dict() for item in self.components],
            "red_flags": [item.as_dict() for item in self.red_flags],
        }


@dataclass(frozen=True)
class ScoringAnalysis(MetricScaffoldBase):
    """Build transparent score from metrics, coverage, and warning penalties."""

    ticker: str
    prices: pd.Series | None = None
    data: pd.DataFrame | None = None
    fundamentals: FundamentalsInput = None
    as_of: str | pd.Timestamp | None = None
    frequency: FiscalFrequency = "quarterly"
    config: ScoringConfig = ScoringConfig()

    @staticmethod
    def linear_score(
        value: float,
        lower_bound: float,
        upper_bound: float,
        direction: ScoreDirection = "higher_is_better",
    ) -> float:
        """Map value linearly to bounded score between zero and 100."""
        inputs = (value, lower_bound, upper_bound)

        if any(not np.isfinite(item) for item in inputs):
            raise ValueError("value and bounds must be finite")

        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be below upper_bound")

        if direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(
                "direction must be `higher_is_better` or `lower_is_better`"
            )

        normalized = (value - lower_bound) / (upper_bound - lower_bound)
        bounded = float(np.clip(normalized, 0.0, 1.0))

        if direction == "lower_is_better":
            bounded = 1.0 - bounded

        return bounded * 100

    @staticmethod
    def missing_component(
        category: ScoreCategory,
        metric: str,
        weight: float,
        reason: str,
        *,
        invalid: bool = False,
    ) -> ScoreComponent:
        """Create explicit missing/invalid component instead of dropping it."""
        return ScoreComponent(
            category=category,
            metric=metric,
            value=None,
            score=None,
            weight=weight,
            explanation=reason,
            status="invalid" if invalid else "missing",
        )

    def _scoring_prices(self) -> pd.Series:
        """Return sorted close prices restricted to configured as-of date."""
        prices = self._close_prices()

        if not isinstance(prices.index, pd.DatetimeIndex):
            raise ValueError("scoring prices must use a DatetimeIndex")

        prices = prices.sort_index()

        if self.as_of is not None:
            prices = prices.loc[prices.index <= pd.Timestamp(self.as_of)]

        if prices.empty:
            raise ValueError("No prices are available on or before as_of")

        return prices

    def _fundamental_series(
        self,
        candidates: tuple[MetricCandidate, ...],
    ) -> pd.Series:
        """Return configured point-in-time fundamental series."""
        return self._fundamental_metric_series_from_candidates(
            candidates,
            date_column="period_end",
            frequency=self.frequency,
        )

    def _free_cash_flow_series(self) -> pd.Series:
        """Return free cash flow aligned by reporting period."""
        operating_cash_flow = self._fundamental_series(_OPERATING_CASH_FLOW_CANDIDATES)
        capex = self._fundamental_series(_CAPEX_CANDIDATES)
        aligned = pd.concat(
            {
                "operating_cash_flow": operating_cash_flow,
                "capex": capex,
            },
            axis=1,
            join="inner",
        ).dropna()

        if aligned.empty:
            raise ValueError("No aligned operating cash flow and capex periods found")

        free_cash_flow = aligned["operating_cash_flow"] - aligned["capex"].abs()
        free_cash_flow.name = "free_cash_flow"
        return free_cash_flow

    def profitability_components(self) -> list[ScoreComponent]:
        """Build operating margin, net margin, ROE, and ROIC components.

        V1 uses sector-neutral absolute bounds. Values at or below zero score
        zero. Operating margin reaches 100 at 25%, net margin and ROIC at 20%,
        and ROE at 25%. Missing metrics retain their weight for coverage.
        """
        profitability = Profitability(
            ticker=self.ticker,
            prices=self.prices,
            data=self.data,
            fundamentals=self.fundamentals,
            as_of=self.as_of,
            frequency=self.frequency,
        )
        definitions = (
            (
                "operating_margin",
                profitability.operating_margin,
                0.0,
                0.25,
                "Operating margin",
            ),
            (
                "net_margin",
                profitability.net_margin,
                0.0,
                0.20,
                "Net margin",
            ),
            (
                "return_on_equity",
                profitability.return_on_equity,
                0.0,
                0.25,
                "Return on equity",
            ),
            (
                "return_on_invested_capital",
                profitability.return_on_invested_capital,
                0.0,
                0.20,
                "Return on invested capital",
            ),
        )
        component_weight = self.config.profitability_weight / len(definitions)
        components: list[ScoreComponent] = []

        for metric, method, lower_bound, upper_bound, label in definitions:
            try:
                values = method()

                if values.empty:
                    raise ValueError(f"{metric} produced no observations")

                latest_value = float(values.iloc[-1])

                if not np.isfinite(latest_value):
                    components.append(
                        self.missing_component(
                            category="profitability",
                            metric=metric,
                            weight=component_weight,
                            reason=f"{label} latest value is not finite.",
                            invalid=True,
                        )
                    )
                    continue

                score = self.linear_score(
                    value=latest_value,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                components.append(
                    ScoreComponent(
                        category="profitability",
                        metric=metric,
                        value=latest_value,
                        score=score,
                        weight=component_weight,
                        explanation=(
                            f"{label} {latest_value:.2%}; score maps linearly "
                            f"from {lower_bound:.0%} to {upper_bound:.0%}."
                        ),
                    )
                )
            except ValueError as exc:
                message = str(exc)
                invalid = any(
                    marker in message.lower()
                    for marker in ("undefined", "zero", "must be between")
                )
                components.append(
                    self.missing_component(
                        category="profitability",
                        metric=metric,
                        weight=component_weight,
                        reason=f"{label} unavailable: {message}",
                        invalid=invalid,
                    )
                )

        return components

    def growth_components(self) -> list[ScoreComponent]:
        """Build revenue, earnings, and free-cash-flow growth components.

        Latest growth maps from -10% to 25%. With two or more growth
        observations, score is averaged with a stability score based on growth
        dispersion. Non-positive comparison levels are invalid, not growth.
        """
        growth = Growth(
            ticker=self.ticker,
            prices=self.prices,
            data=self.data,
            fundamentals=self.fundamentals,
            as_of=self.as_of,
            frequency=self.frequency,
        )

        def earnings_source() -> pd.Series:
            try:
                return self._fundamental_series(_NET_INCOME_CANDIDATES)
            except ValueError:
                return self._fundamental_series(_EPS_CANDIDATES)

        def earnings_growth() -> pd.Series:
            try:
                return growth.net_income_growth()
            except ValueError:
                return growth.earnings_per_share_growth()

        definitions = (
            (
                "revenue_growth",
                growth.revenue_growth,
                lambda: self._fundamental_series(_REVENUE_CANDIDATES),
                "Revenue growth",
            ),
            (
                "earnings_growth",
                earnings_growth,
                earnings_source,
                "Earnings growth",
            ),
            (
                "free_cash_flow_growth",
                growth.free_cash_flow_growth,
                self._free_cash_flow_series,
                "Free cash flow growth",
            ),
        )
        component_weight = self.config.growth_weight / len(definitions)
        components: list[ScoreComponent] = []

        for metric, method, source_method, label in definitions:
            try:
                source = source_method()

                if len(source) < 2:
                    raise ValueError(f"{metric} requires at least two observations")

                if (source.iloc[-2:] <= 0).any():
                    components.append(
                        self.missing_component(
                            category="growth",
                            metric=metric,
                            weight=component_weight,
                            reason=(
                                f"{label} requires positive current and prior values."
                            ),
                            invalid=True,
                        )
                    )
                    continue

                values = method()
                latest_value = float(values.iloc[-1])
                base_score = self.linear_score(latest_value, -0.10, 0.25)
                score = base_score
                stability_text = "stability unavailable"

                if len(values) >= 2:
                    dispersion = float(values.std(ddof=1))

                    if np.isfinite(dispersion):
                        stability = 1 / (1 + dispersion)
                        score = (base_score + stability * 100) / 2
                        stability_text = f"stability {stability:.2%}"

                components.append(
                    ScoreComponent(
                        category="growth",
                        metric=metric,
                        value=latest_value,
                        score=score,
                        weight=component_weight,
                        explanation=(
                            f"{label} {latest_value:.2%}; latest growth maps from "
                            f"-10% to 25%, blended with {stability_text}."
                        ),
                    )
                )
            except ValueError as exc:
                components.append(
                    self.missing_component(
                        category="growth",
                        metric=metric,
                        weight=component_weight,
                        reason=f"{label} unavailable: {exc}",
                    )
                )

        return components

    def valuation_components(self) -> list[ScoreComponent]:
        """Build absolute valuation components from positive multiples/yield."""
        try:
            prices = self._scoring_prices()
        except ValueError:
            prices = None

        valuation = Valuation(
            ticker=self.ticker,
            prices=prices,
            data=None,
            fundamentals=self.fundamentals,
            as_of=self.as_of,
        )
        definitions = (
            (
                "price_to_earnings",
                valuation.price_to_earnings,
                10.0,
                35.0,
                "lower_is_better",
                "Price-to-earnings",
            ),
            (
                "price_to_sales",
                valuation.price_to_sales,
                1.0,
                8.0,
                "lower_is_better",
                "Price-to-sales",
            ),
            (
                "price_to_book",
                valuation.price_to_book,
                1.0,
                6.0,
                "lower_is_better",
                "Price-to-book",
            ),
            (
                "ev_to_ebitda",
                valuation.ev_to_ebitda,
                6.0,
                20.0,
                "lower_is_better",
                "EV/EBITDA",
            ),
            (
                "free_cash_flow_yield",
                valuation.free_cash_flow_yield,
                0.0,
                0.10,
                "higher_is_better",
                "Free cash flow yield",
            ),
        )
        component_weight = self.config.valuation_weight / len(definitions)
        components: list[ScoreComponent] = []

        for metric, method, lower, upper, direction, label in definitions:
            try:
                value = float(method())

                if not np.isfinite(value):
                    raise ValueError(f"{metric} is not finite")

                if value < 0 or (metric != "free_cash_flow_yield" and value == 0):
                    components.append(
                        self.missing_component(
                            category="valuation",
                            metric=metric,
                            weight=component_weight,
                            reason=f"{label} must be positive for valuation scoring.",
                            invalid=True,
                        )
                    )
                    continue

                score = self.linear_score(value, lower, upper, direction)
                components.append(
                    ScoreComponent(
                        category="valuation",
                        metric=metric,
                        value=value,
                        score=score,
                        weight=component_weight,
                        explanation=(
                            f"{label} {value:.2f}; score maps linearly from "
                            f"{lower:.2f} to {upper:.2f} "
                            f"({direction.replace('_', ' ')})."
                        ),
                    )
                )
            except ValueError as exc:
                message = str(exc)
                invalid = any(
                    marker in message.lower()
                    for marker in ("undefined", "zero", "strictly positive")
                )
                components.append(
                    self.missing_component(
                        category="valuation",
                        metric=metric,
                        weight=component_weight,
                        reason=f"{label} unavailable: {message}",
                        invalid=invalid,
                    )
                )

        return components

    def risk_components(self) -> list[ScoreComponent]:
        """Build annual volatility, annual downside, and drawdown components."""
        definitions = (
            ("annualized_volatility", 0.10, 0.60, "Annualized volatility"),
            ("annualized_downside_deviation", 0.05, 0.35, "Downside deviation"),
            ("maximum_drawdown", 0.0, 0.50, "Maximum drawdown"),
        )
        component_weight = self.config.risk_weight / len(definitions)

        try:
            prices = self._scoring_prices()
            risk = Risk(ticker=self.ticker, prices=prices, data=pd.DataFrame())
            periods_per_year = risk._periods_per_year()
            values = {
                "annualized_volatility": risk.annualized_volatility(periods_per_year),
                "annualized_downside_deviation": (
                    risk.downside_deviation() * np.sqrt(periods_per_year)
                ),
                "maximum_drawdown": float(prices.div(prices.cummax()).sub(1).min()),
            }
        except ValueError as exc:
            return [
                self.missing_component(
                    category="risk",
                    metric=metric,
                    weight=component_weight,
                    reason=f"{label} unavailable: {exc}",
                )
                for metric, _, _, label in definitions
            ]

        components: list[ScoreComponent] = []

        for metric, lower, upper, label in definitions:
            raw_value = float(values[metric])
            scoring_value = (
                abs(raw_value) if metric == "maximum_drawdown" else raw_value
            )

            if not np.isfinite(scoring_value):
                components.append(
                    self.missing_component(
                        category="risk",
                        metric=metric,
                        weight=component_weight,
                        reason=f"{label} is not finite.",
                        invalid=True,
                    )
                )
                continue

            score = self.linear_score(
                scoring_value,
                lower,
                upper,
                direction="lower_is_better",
            )
            components.append(
                ScoreComponent(
                    category="risk",
                    metric=metric,
                    value=raw_value,
                    score=score,
                    weight=component_weight,
                    explanation=(
                        f"{label} {raw_value:.2%}; risk magnitude maps from "
                        f"{lower:.0%} to {upper:.0%}, lower is better."
                    ),
                )
            )

        return components

    def momentum_components(self) -> list[ScoreComponent]:
        """Build calendar-based six- and twelve-month momentum components."""
        definitions = (
            ("price_momentum_6m", 6, -0.30, 0.30, "Six-month momentum"),
            ("price_momentum_12m", 12, -0.40, 0.40, "Twelve-month momentum"),
        )
        component_weight = self.config.momentum_weight / len(definitions)

        try:
            prices = self._scoring_prices()
        except ValueError as exc:
            return [
                self.missing_component(
                    category="momentum",
                    metric=metric,
                    weight=component_weight,
                    reason=f"{label} unavailable: {exc}",
                )
                for metric, _, _, _, label in definitions
            ]

        latest_date = prices.index[-1]
        latest_price = float(prices.iloc[-1])
        components: list[ScoreComponent] = []

        for metric, months, lower, upper, label in definitions:
            target_date = latest_date - pd.DateOffset(months=months)
            history = prices.loc[prices.index <= target_date]

            if history.empty:
                components.append(
                    self.missing_component(
                        category="momentum",
                        metric=metric,
                        weight=component_weight,
                        reason=(
                            f"{label} requires prices on or before "
                            f"{target_date.date()}."
                        ),
                    )
                )
                continue

            base_price = float(history.iloc[-1])
            momentum = latest_price / base_price - 1
            score = self.linear_score(momentum, lower, upper)
            components.append(
                ScoreComponent(
                    category="momentum",
                    metric=metric,
                    value=momentum,
                    score=score,
                    weight=component_weight,
                    explanation=(
                        f"{label} {momentum:.2%}; price behavior maps "
                        f"symmetrically from {lower:.0%} to {upper:.0%}."
                    ),
                )
            )

        return components

    def red_flag_penalty(self, flags: list[RedFlag]) -> float:
        """Return uncapped sum of configured warning penalties."""
        if not isinstance(flags, list) or any(
            not isinstance(flag, RedFlag) for flag in flags
        ):
            raise TypeError("flags must be a list of RedFlag objects")

        penalties = {
            "low": self.config.low_flag_penalty,
            "medium": self.config.medium_flag_penalty,
            "high": self.config.high_flag_penalty,
            "critical": self.config.critical_flag_penalty,
        }
        return float(sum(penalties[flag.severity] for flag in flags))

    def aggregate(
        self,
        components: list[ScoreComponent],
        flags: list[RedFlag],
    ) -> CompanyScore:
        """Aggregate available weights, coverage, and red-flag penalty.

        Available component weights are re-normalized. Score is withheld when
        scored-weight coverage falls below configured minimum.
        """
        if not isinstance(components, list) or any(
            not isinstance(component, ScoreComponent) for component in components
        ):
            raise TypeError("components must be a list of ScoreComponent objects")

        if not components:
            raise ValueError("components cannot be empty")

        if not isinstance(flags, list) or any(
            not isinstance(flag, RedFlag) for flag in flags
        ):
            raise TypeError("flags must be a list of RedFlag objects")

        expected_weight = float(sum(component.weight for component in components))

        if expected_weight <= 0:
            raise ValueError("component weights must have a positive total")

        scored = [component for component in components if component.score is not None]
        scored_weight = float(sum(component.weight for component in scored))
        coverage = scored_weight / expected_weight
        penalty = self.red_flag_penalty(flags)

        if coverage < self.config.minimum_coverage or scored_weight == 0:
            total_score = None
            grade = None
        else:
            weighted_points = sum(
                component.score * component.weight
                for component in scored
                if component.score is not None
            )
            base_score = weighted_points / scored_weight
            total_score = float(np.clip(base_score - penalty, 0.0, 100.0))

            if total_score >= 85:
                grade = "A"
            elif total_score >= 70:
                grade = "B"
            elif total_score >= 55:
                grade = "C"
            elif total_score >= 40:
                grade = "D"
            else:
                grade = "E"

        return CompanyScore(
            ticker=self.ticker,
            total_score=total_score,
            grade=grade,
            coverage=coverage,
            penalty=penalty,
            components=tuple(components),
            red_flags=tuple(flags),
        )

    def score(self) -> CompanyScore:
        """Run categories and deterministic point-in-time red-flag analysis."""
        category_methods = (
            self.profitability_components,
            self.growth_components,
            self.valuation_components,
            self.risk_components,
            self.momentum_components,
        )
        components = [
            component for method in category_methods for component in method()
        ]
        effective_as_of = self.as_of
        scoring_prices: pd.Series | None = None

        try:
            scoring_prices = self._scoring_prices()
        except ValueError:
            pass

        if effective_as_of is None:
            if scoring_prices is not None:
                effective_as_of = scoring_prices.index[-1]
            else:
                if self.fundamentals is not None:
                    statements = self._ticker_statements()

                    if "filed_date" in statements.columns:
                        filed_dates = pd.to_datetime(
                            statements["filed_date"],
                            errors="coerce",
                        ).dropna()

                        if not filed_dates.empty:
                            effective_as_of = filed_dates.max()

        flag_analysis = RedFlagAnalysis(
            ticker=self.ticker,
            prices=scoring_prices,
            data=self.data,
            fundamentals=self.fundamentals,
            as_of=effective_as_of,
            frequency=self.frequency,
        )
        flags = flag_analysis.run_all(strict=False)
        return self.aggregate(components, flags)


__all__ = [
    "CompanyScore",
    "ScoreCategory",
    "ScoreComponent",
    "ScoreDirection",
    "ScoreStatus",
    "ScoringAnalysis",
    "ScoringConfig",
]
