"""Create simple EDA tables and professional plots."""

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR
from src.data import load_dataset


def create_eda() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = No Disease, 1 = Disease)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close()

    # Histograms
    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df[numeric_columns].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribution of Numeric Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "numeric_histograms.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Age versus maximum heart rate by target
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y="thalach", hue="target")
    plt.title("Age vs Maximum Heart Rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_vs_thalach.png", dpi=150)
    plt.close()

    print(f"EDA plots saved in {FIGURES_DIR}")


if __name__ == "__main__":
    create_eda()
