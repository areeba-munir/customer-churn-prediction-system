"""Explore churn patterns using training data only."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def main() -> None:
    """Save training-data charts and group summaries."""
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    df = pd.read_csv(train_path)

    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})
    if df["ChurnFlag"].isna().any():
        raise ValueError("Training data contains missing or unknown churn labels.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Include zero-month customers in the first group.
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    )

    group_fields = [
        "Contract",
        "TenureGroup",
        "InternetService",
        "PaymentMethod",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    summaries = []

    for field, ax in zip(group_fields, axes.flat, strict=True):
        summary = (
            df.groupby(field, observed=True)["ChurnFlag"]
            .agg(customers="size", churned="sum", churn_rate="mean")
            .reset_index()
        )
        summary["churn_rate_percent"] = summary["churn_rate"].mul(100)

        summary.to_csv(METRICS_DIR / f"eda_{field}.csv", index=False)

        labels = summary[field].astype(str).tolist()
        rates = summary["churn_rate_percent"].tolist()
        bars = ax.barh(labels, rates, color="#2878A5")

        ax.bar_label(bars, fmt="%.1f%%", padding=4)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Customers who churned (%)")
        ax.set_ylabel("")
        ax.set_title(f"Churn by {field}")
        ax.invert_yaxis()

        summaries.append(
            f"\nChurn by {field}:\n"
            + summary[[field, "customers", "churned", "churn_rate_percent"]]
            .round(2)
            .to_string(index=False)
        )

    fig.suptitle("Customer churn patterns — training data only", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "training_churn_patterns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        order=["No", "Yes"],
        color="#8EBBD2",
        ax=ax,
    )
    ax.set_title("Monthly charges by churn status — training data only")
    ax.set_xlabel("Customer churned")
    ax.set_ylabel("Monthly charges")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_monthly_charges.png", dpi=160)
    plt.close(fig)

    charge_summary = df.groupby("Churn")["MonthlyCharges"].agg(
        customers="size",
        median="median",
        mean="mean",
    )
    charge_summary.to_csv(METRICS_DIR / "eda_monthly_charges.csv")

    report = (
        f"Training customers analyzed: {len(df)}\n"
        f"Training churn rate: {df['ChurnFlag'].mean():.2%}\n"
        + "\n".join(summaries)
        + "\n\nMonthly charges by churn status:\n"
        + charge_summary.round(2).to_string()
        + "\n\nThese summaries show associations, not causes.\n"
    )
    report = "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
    (METRICS_DIR / "eda_summary.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Saved two charts in reports/figures/.")


if __name__ == "__main__":
    main()