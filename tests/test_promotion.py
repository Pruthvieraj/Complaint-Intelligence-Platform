"""Unit tests for the Phase-3 promotion gate. Pure-function tests only —
no MLflow server needed — so these run fast in CI on every push."""
from src.registry.promote import (
    MINORITY_F1_TOLERANCE,
    PROMOTION_MARGIN,
    decide_promotion,
)


def test_bootstrap_promotion_when_no_production_exists():
    promoted, reason = decide_promotion({"test_macro_f1": 0.50}, None)
    assert promoted is True
    assert "bootstrap" in reason.lower()


def test_promotion_requires_margin_not_just_any_improvement():
    candidate = {"test_macro_f1": 0.70 + (PROMOTION_MARGIN / 2), "test_minority_class_f1": 0.5}
    production = {"test_macro_f1": 0.70, "test_minority_class_f1": 0.5}
    promoted, reason = decide_promotion(candidate, production)
    assert promoted is False
    assert "REJECTED" in reason


def test_promotion_succeeds_when_margin_cleared_and_no_regression():
    candidate = {"test_macro_f1": 0.70 + PROMOTION_MARGIN + 0.001, "test_minority_class_f1": 0.55}
    production = {"test_macro_f1": 0.70, "test_minority_class_f1": 0.50}
    promoted, reason = decide_promotion(candidate, production)
    assert promoted is True
    assert "PROMOTED" in reason


def test_rejected_on_minority_class_regression_even_if_macro_f1_improves():
    candidate = {"test_macro_f1": 0.80, "test_minority_class_f1": 0.30}
    production = {"test_macro_f1": 0.70, "test_minority_class_f1": 0.50 + MINORITY_F1_TOLERANCE}
    promoted, reason = decide_promotion(candidate, production)
    assert promoted is False
    assert "minority" in reason.lower()


def test_missing_metric_is_rejected_not_crashed():
    promoted, reason = decide_promotion({}, {"test_macro_f1": 0.5})
    assert promoted is False
