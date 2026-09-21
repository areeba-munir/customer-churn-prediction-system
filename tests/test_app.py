"""Integration tests for the Streamlit customer workflow."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def widget_by_label(widgets, label):
    """Find a widget by its visible label."""
    return next(widget for widget in widgets if widget.label == label)


def assert_prediction_displayed(app):
    """Check that prediction completed and the alert matches the threshold."""
    assert not app.exception
    assert not app.error

    metrics = {metric.label: metric.value for metric in app.metric}
    probability = float(metrics["Estimated churn probability"].rstrip("%"))
    threshold = float(metrics["Alert threshold"].rstrip("%"))

    assert 0 <= probability <= 100
    assert threshold == 23.0
    assert len(app.warning) + len(app.success) == 1

    # Allow for the probability's display rounding near the threshold.
    if probability > threshold + 0.1:
        assert len(app.warning) == 1
    elif probability < threshold - 0.1:
        assert len(app.success) == 1


def test_default_prediction_and_stale_result_clearing():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert not app.error
    assert app.title[0].value == "Customer Churn Prediction"
    assert len(app.metric) == 0

    widget_by_label(app.button, "Predict churn").click().run()
    assert_prediction_displayed(app)

    widget_by_label(app.number_input, "Tenure (months)").set_value(24).run()

    assert not app.exception
    assert len(app.metric) == 0
    assert len(app.warning) == 0
    assert len(app.success) == 0


def test_customer_without_phone_or_internet():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    widget_by_label(app.selectbox, "Phone service").select("No").run()
    widget_by_label(app.selectbox, "Internet service").select("No").run()

    labels = {widget.label for widget in app.selectbox}
    assert "Multiple phone lines" not in labels
    assert "Online security" not in labels
    assert "Streaming movies" not in labels

    widget_by_label(app.button, "Predict churn").click().run()
    assert_prediction_displayed(app)


def test_unknown_total_charges():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    widget_by_label(app.checkbox, "Total charges are unknown").check().run()

    assert widget_by_label(app.number_input, "Total charges").disabled

    widget_by_label(app.button, "Predict churn").click().run()
    assert_prediction_displayed(app)