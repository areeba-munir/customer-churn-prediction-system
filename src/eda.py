"""Explore churn patterns using training data only."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def create_group_summary(
    df: pd.DataFrame,
    field: str,
    output_name: str,
) -> pd.DataFrame:
    """Calculate and save churn statistics for a categorical field."""
    summary = (
        df.groupby(field, observed=True)["ChurnFlag"]
        .agg(customers="size", churned="sum", churn_rate="mean")
        .reset_index()
    )
    summary["churn_rate_percent"] = summary["churn_rate"].mul(100)
    summary.to_csv(
        METRICS_DIR / f"eda_{output_name}.csv",
        index=False,
    )
    return summary


def plot_group_summary(
    ax: plt.Axes,
    summary: pd.DataFrame,
    field: str,
    title: str,
) -> None:
    """Plot churn rates for one categorical field."""
    labels = summary[field].astype(str).tolist()
    rates = summary["churn_rate_percent"].tolist()
    bars = ax.barh(labels, rates, color="#2878A5")

    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Customers who churned (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.invert_yaxis()


def format_group_report(
    title: str,
    summary: pd.DataFrame,
    field: str,
) -> str:
    """Format one group summary for the text report."""
    columns = [
        field,
        "customers",
        "churned",
        "churn_rate_percent",
    ]
    return (
        f"\nChurn by {title}:\n"
        + summary[columns].round(2).to_string(index=False)
    )


def main() -> None:
    """Save training-data charts and group summaries."""
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    df = pd.read_csv(train_path)

    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})
    if df["ChurnFlag"].isna().any():
        raise ValueError(
            "Training data contains missing or unknown churn labels."
        )

    total_text = df["TotalCharges"].astype("string").str.strip()
    df["TotalChargesNumeric"] = pd.to_numeric(
        total_text,
        errors="coerce",
    )

    invalid_total = (
        df["TotalChargesNumeric"].isna()
        & total_text.notna()
        & total_text.ne("")
    )
    if invalid_total.any():
        raise ValueError(
            "Training data contains invalid nonblank TotalCharges values."
        )

    senior_labels = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    if senior_labels.isna().any():
        raise ValueError("Training data contains unknown SeniorCitizen values.")
    df["SeniorCitizen"] = senior_labels

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49+ months",
        ],
    )

    primary_fields = [
        ("Contract", "Contract type", "Contract"),
        ("TenureGroup", "Tenure group", "TenureGroup"),
        (
            "InternetService",
            "Internet service",
            "InternetService",
        ),
        ("PaymentMethod", "Payment method", "PaymentMethod"),
    ]

    secondary_fields = [
        (
            "OnlineSecurity",
            "Online security service",
            "OnlineSecurity",
        ),
        ("TechSupport", "Technical support service", "TechSupport"),
        ("SeniorCitizen", "Senior-citizen status", "SeniorCitizen"),
    ]

    report_sections: list[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for (
        field,
        title,
        output_name,
    ), ax in zip(primary_fields, axes.flat, strict=True):
        summary = create_group_summary(df, field, output_name)
        plot_group_summary(ax, summary, field, title)
        report_sections.append(
            format_group_report(title, summary, field)
        )

    fig.suptitle(
        "Customer churn patterns - training data only",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        FIGURES_DIR / "training_churn_patterns.png",
        dpi=160,
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for (
        field,
        title,
        output_name,
    ), ax in zip(secondary_fields, axes.flat, strict=True):
        summary = create_group_summary(df, field, output_name)
        plot_group_summary(ax, summary, field, title)
        report_sections.append(
            format_group_report(title, summary, field)
        )

    fig.suptitle(
        "Services and demographics - training data only",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        FIGURES_DIR / "training_service_demographics.png",
        dpi=160,
    )
    plt.close(fig)

    charge_fields = [
        (
            "MonthlyCharges",
            "Monthly charges",
            "monthly_charges",
        ),
        (
            "TotalChargesNumeric",
            "Total charges",
            "total_charges",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    charge_sections: list[str] = []

    for (
        field,
        title,
        output_name,
    ), ax in zip(charge_fields, axes.flat, strict=True):
        sns.boxplot(
            data=df,
            x="Churn",
            y=field,
            order=["No", "Yes"],
            color="#8EBBD2",
            ax=ax,
        )
        ax.set_title(f"{title} by churn status")
        ax.set_xlabel("Customer churned")
        ax.set_ylabel(title)

        charge_summary = (
            df.groupby("Churn")[field]
            .agg(
                customers="size",
                missing=lambda values: values.isna().sum(),
                median="median",
                mean="mean",
            )
        )
        charge_summary.to_csv(
            METRICS_DIR / f"eda_{output_name}.csv"
        )
        charge_sections.append(
            f"\n{title} by churn status:\n"
            + charge_summary.round(2).to_string()
        )

    fig.suptitle(
        "Charges by churn status - training data only",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        FIGURES_DIR / "training_charges.png",
        dpi=160,
    )
    plt.close(fig)

    report = (
        f"Training customers analyzed: {len(df)}\n"
        f"Training churn rate: {df['ChurnFlag'].mean():.2%}\n"
        + "\n".join(report_sections)
        + "\n"
        + "\n".join(charge_sections)
        + "\n\nThese summaries show associations, not causes.\n"
    )
    report = (
        "\n".join(line.rstrip() for line in report.splitlines())
        + "\n"
    )

    (METRICS_DIR / "eda_summary.txt").write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print("Saved three EDA charts in reports/figures/.")


if __name__ == "__main__":
    main()