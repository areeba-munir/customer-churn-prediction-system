"""Streamlit interface for the saved customer churn model."""

import logging
import sys
from pathlib import Path

import streamlit as st

# Allow Streamlit to import project modules when launched from this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    CATEGORY_OPTIONS,
    INTERNET_ADDONS,
    load_artifact,
    predict_customer,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Load the saved pipeline once without retraining."""
    return load_artifact()


def main():
    """Collect customer details and display a validated prediction."""
    st.title("Customer Churn Prediction")
    st.write(
        "Estimate whether a customer may leave using their subscription, "
        "services, and billing information."
    )

    try:
        artifact = get_model()
    except Exception:
        logger.exception("Unable to load the saved churn model.")
        st.error(
            "The prediction model could not be loaded. "
            "Check that the saved model and project dependencies are available."
        )
        st.stop()

    with st.sidebar:
        st.header("About this model")
        st.write(f"**Model:** {artifact['metadata']['model']}")
        st.write(f"**Churn alert threshold:** {artifact['threshold']:.0%}")
        st.caption(
            "The threshold was selected using training-data validation "
            "to prioritize recall while maintaining at least 50% precision."
        )
        st.markdown("**Final holdout results**")
        st.write("Recall: 82.35%")
        st.write("Precision: 50.41%")
        st.write("ROC-AUC: 0.8434")
        st.caption(
            "These are results on 1,409 held-out customers, "
            "not guarantees for future customers."
        )

    st.info(
        "The fields contain an illustrative starting profile. "
        "Replace the values with the customer details you want to assess."
    )

    customer = {}

    st.subheader("1. Customer and subscription")
    left, middle, right = st.columns(3)

    with left:
        customer["gender"] = st.selectbox(
            "Gender",
            CATEGORY_OPTIONS["gender"],
        )
        customer["SeniorCitizen"] = int(
            st.selectbox("Senior citizen", ["No", "Yes"]) == "Yes"
        )

    with middle:
        customer["Partner"] = st.selectbox(
            "Has a partner",
            CATEGORY_OPTIONS["Partner"],
        )
        customer["Dependents"] = st.selectbox(
            "Has dependents",
            CATEGORY_OPTIONS["Dependents"],
        )

    with right:
        customer["tenure"] = st.number_input(
            "Tenure (months)",
            min_value=0,
            value=12,
            step=1,
            help="Number of months the customer has been with the company.",
        )
        customer["Contract"] = st.selectbox(
            "Contract",
            CATEGORY_OPTIONS["Contract"],
        )

    st.subheader("2. Services")
    left, right = st.columns(2)

    with left:
        customer["PhoneService"] = st.selectbox(
            "Phone service",
            ["Yes", "No"],
        )

        if customer["PhoneService"] == "Yes":
            customer["MultipleLines"] = st.selectbox(
                "Multiple phone lines",
                ["No", "Yes"],
            )
        else:
            customer["MultipleLines"] = "No phone service"
            st.caption("Multiple lines: not applicable without phone service.")

    with right:
        customer["InternetService"] = st.selectbox(
            "Internet service",
            CATEGORY_OPTIONS["InternetService"],
        )

    addon_labels = {
        "OnlineSecurity": "Online security",
        "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection",
        "TechSupport": "Technical support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies",
    }

    if customer["InternetService"] == "No":
        for field in INTERNET_ADDONS:
            customer[field] = "No internet service"
        st.caption("Internet add-ons are not applicable without internet service.")
    else:
        columns = st.columns(3)
        for index, field in enumerate(INTERNET_ADDONS):
            with columns[index % 3]:
                customer[field] = st.selectbox(
                    addon_labels[field],
                    ["No", "Yes"],
                    key=field,
                )
    st.subheader("3. Billing")

    total_unknown = st.checkbox("Total charges are unknown")

    left, middle, right = st.columns(3)

    with left:
        customer["MonthlyCharges"] = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=55.0,
            step=1.0,
            format="%.2f",
        )

    with middle:
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=660.0,
            step=10.0,
            format="%.2f",
            disabled=total_unknown,
            help="Cumulative charges, using the same currency units as monthly charges.",
        )
        customer["TotalCharges"] = None if total_unknown else total_charges

    with right:
        customer["PaymentMethod"] = st.selectbox(
            "Payment method",
            CATEGORY_OPTIONS["PaymentMethod"],
        )

    if total_unknown:
        st.caption("The saved training-data median will fill this missing value.")

    billing_options, _ = st.columns([1, 2])

    with billing_options:
        customer["PaperlessBilling"] = st.selectbox(
            "Paperless billing",
            CATEGORY_OPTIONS["PaperlessBilling"],
        )

    if st.button("Predict churn", type="primary"):
        try:
            with st.spinner("Calculating prediction..."):
                result = predict_customer(customer, artifact)
        except ValueError as exc:
            st.error(f"Please check the customer details: {exc}")
        except Exception:
            logger.exception("Customer prediction failed.")
            st.error("Prediction could not be completed. Please try again.")
        else:
            st.divider()
            st.subheader("Prediction result")

            probability = result["churn_probability"]
            above_threshold = result["prediction"] == "Churn"

            if above_threshold:
                st.warning(
                    "Churn alert: this customer is above the selected risk threshold. "
                    "Consider reviewing their account for retention outreach."
                )
            else:
                st.success(
                    "No churn alert: this customer is below the selected risk threshold. "
                    "This does not guarantee that they will stay."
                )

            first, second = st.columns(2)
            first.metric("Estimated churn probability", f"{probability:.1%}")
            second.metric("Alert threshold", f"{result['threshold']:.0%}")
            st.progress(probability)
            st.caption(
                "The bar shows the model's estimated probability, "
                "not a guaranteed outcome or a calibrated confidence score."
            )

    with st.expander("How to interpret the prediction"):
        st.write(
            "The model learns patterns from the IBM Telco sample dataset. "
            "Contract type and tenure had the largest native feature importance. "
            "These are overall model findings, not explanations for this individual customer."
        )
        st.write(
            "At the selected threshold, the holdout evaluation caught 308 churners, "
            "missed 66, and incorrectly flagged 303 customers who stayed."
        )
        st.write(
            "Performance can differ for other companies, time periods, or customer groups. "
            "Unusual inputs may be outside the model's training experience."
        )

    st.caption(
        "For analytical and educational use. Predictions support human review "
        "and do not establish why a customer will leave."
    )


if __name__ == "__main__":
    main()