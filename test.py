import numpy as np
import pandas as pd
from typing import Sequence, Mapping, Optional
import matplotlib.pyplot as plt


def generate_sales_data(
    start_date: str,
    end_date: str,
    regions: Sequence[str],
    *,
    seed: Optional[int] = 42,
    slope: float = 0.5,
    base_level: float = 50.0,
    weekly_amp: float = 15.0,
    yearly_amp: float = 30.0,
    noise_std: float = 10.0,
    region_multipliers: Mapping[str, float] = None,
) -> pd.DataFrame:
    """
    Generate synthetic daily vehicle sales for given regions.

    Returns a DataFrame with columns: Date (datetime64[ns]), Region (str), Sales (int).
    Reproducible when seed is provided.
    """
    if region_multipliers is None:
        region_multipliers = {"North": 1.5, "Central": 1.0, "South": 0.8}

    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    t = np.arange(len(dates), dtype=float)

    base_trend = base_level + slope * t
    weekly = np.sin(2 * np.pi * t / 7) * weekly_amp
    yearly = np.sin(2 * np.pi * t / 365.25) * yearly_amp

    rows = []
    for region in regions:
        scale = float(region_multipliers.get(region, 1.0))
        noise = rng.normal(0, noise_std, size=len(t))
        sales = base_trend + weekly + yearly + noise
        sales = np.round(np.clip(sales * scale, 0, None)).astype(int)

        df_region = pd.DataFrame({"Date": dates, "Region": region, "Sales": sales})
        rows.append(df_region)

    result = pd.concat(rows, ignore_index=True)
    # keep deterministic ordering
    result = result.sort_values(["Region", "Date"]).reset_index(drop=True)
    return result


def plot_region_sales(df: pd.DataFrame, region: str, window: int = 30):
    """Quick plot of recent sales for a given region."""
    sub = df.loc[df["Region"] == region].set_index("Date")["Sales"]
    if sub.empty:
        raise ValueError(f"No data for region {region!r}")
    ax = sub.rolling(window).mean().plot(title=f"{region} - Rolling mean ({window} days)")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # example usage
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    regions = ["North", "Central", "South"]

    df = generate_sales_data(start_date, end_date, regions, seed=123)
    print(df.head())
    print(df.info())

    # quick plot for one region
    try:
        plot_region_sales(df, "North", window=14)
    except Exception as e:
        print("Plot error:", e)