"""Build reproducible daily and weekly HAR-RV inputs from IQQH closes.

The resulting realized variance is a weekly proxy formed by summing squared
daily percentage log returns from Thursday through Wednesday.  It is not an
intraday realized variance.  The existing weekly benchmark split files remain
unchanged and are used only to supply common split labels and covariates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRICE_FILE = Path("clean_energy_etf_IQQH.csv")
DEFAULT_DAILY_OUTPUT = Path("har_rv_daily_dataset.csv")
DEFAULT_WEEKLY_OUTPUT = Path("har_rv_weekly_dataset.csv")
SPLIT_FILES = {
    "train": Path("train_dataset.csv"),
    "valid": Path("valid_dataset.csv"),
    "test": Path("test_dataset.csv"),
}


def make_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Return sorted daily IQQH closes with percentage log-return quantities."""
    required = {"Date", "Close"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily price data are missing columns: {sorted(missing)}")

    output = daily.loc[:, ["Date", "Close"]].copy()
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce")
    output["Close"] = pd.to_numeric(output["Close"], errors="coerce")
    output = output.dropna(subset=["Date", "Close"])
    output = output.loc[output["Close"] > 0].sort_values("Date")
    output = output.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)

    output["daily_log_return"] = 100 * np.log(output["Close"]).diff()
    output["daily_squared_return"] = output["daily_log_return"] ** 2
    output["week_end"] = (
        output["Date"].dt.to_period("W-WED").dt.end_time.dt.normalize()
    )
    return output


def aggregate_weekly_rv(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate finite daily returns into Thursday-to-Wednesday RV proxies."""
    required = {"week_end", "daily_log_return", "daily_squared_return"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily return data are missing columns: {sorted(missing)}")

    retained = daily.dropna(subset=["daily_log_return", "daily_squared_return"])
    weekly = (
        retained.groupby("week_end", as_index=False)
        .agg(
            realized_variance_daily_proxy=("daily_squared_return", "sum"),
            daily_log_return_sum=("daily_log_return", "sum"),
            trading_days=("daily_log_return", "count"),
        )
        .rename(columns={"week_end": "Date"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return weekly


def read_iqqh_closes(price_file: Path) -> pd.DataFrame:
    """Read the yfinance-style IQQH file using processing.ipynb's header rule."""
    raw = pd.read_csv(price_file, skiprows=[1, 2])
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    if "Close" not in raw.columns:
        raise ValueError(f"{price_file} has no Close column after parsing")
    return raw.loc[:, ["Date", "Close"]]


def load_weekly_splits(split_files: dict[str, Path] = SPLIT_FILES) -> pd.DataFrame:
    """Load the existing weekly rows solely for split labels and covariates."""
    pieces = []
    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Weekly split file not found: {path}")
        piece = pd.read_csv(path)
        if "Date" not in piece.columns:
            raise ValueError(f"Weekly split file has no Date column: {path}")
        piece["Date"] = pd.to_datetime(piece["Date"], errors="coerce")
        piece["split"] = split
        pieces.append(piece)

    weekly = pd.concat(pieces, ignore_index=True)
    weekly = weekly.dropna(subset=["Date"]).sort_values("Date")
    if weekly["Date"].duplicated().any():
        raise ValueError("Weekly split dates must be unique before joining HAR-RV data")
    return weekly.reset_index(drop=True)


def validate_outputs(daily: pd.DataFrame, weekly: pd.DataFrame) -> None:
    """Raise clear errors for output conditions that would invalidate HAR input."""
    if daily.empty or weekly.empty:
        raise ValueError("HAR-RV daily and weekly outputs must both be non-empty")
    if daily["Date"].duplicated().any() or weekly["Date"].duplicated().any():
        raise ValueError("HAR-RV output dates must be unique")

    for column in ("daily_log_return", "daily_squared_return"):
        if not np.isfinite(daily[column]).all():
            raise ValueError(f"Daily HAR-RV output contains non-finite {column}")
    if not np.isfinite(weekly["realized_variance_daily_proxy"]).all():
        raise ValueError("Weekly HAR-RV output contains non-finite realized variance")
    if not (weekly["realized_variance_daily_proxy"] > 0).all():
        raise ValueError("Weekly HAR-RV realized-variance proxies must be positive")
    if not (weekly["trading_days"] > 0).all():
        raise ValueError("Each weekly HAR-RV row must contain at least one trading day")


def build_har_rv_datasets(
    price_file: Path = DEFAULT_PRICE_FILE,
    daily_output: Path = DEFAULT_DAILY_OUTPUT,
    weekly_output: Path = DEFAULT_WEEKLY_OUTPUT,
    split_files: dict[str, Path] = SPLIT_FILES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save separate daily and weekly HAR-RV input data sets."""
    weekly_base = load_weekly_splits(split_files)
    daily = make_daily_returns(read_iqqh_closes(price_file))

    weekly_dates = weekly_base.loc[:, ["Date", "split"]].rename(
        columns={"Date": "week_end"}
    )
    daily = daily.merge(
        weekly_dates,
        on="week_end",
        how="inner",
        validate="many_to_one",
    )
    daily = daily.loc[
        :, ["Date", "week_end", "split", "Close", "daily_log_return", "daily_squared_return"]
    ]
    daily = daily.dropna(subset=["daily_log_return", "daily_squared_return"])

    weekly_rv = aggregate_weekly_rv(daily.rename(columns={"week_end": "week_end"}))
    weekly = weekly_base.merge(weekly_rv, on="Date", how="inner", validate="one_to_one")
    weekly = weekly.loc[
        :, [
            "Date",
            "split",
            "realized_variance_daily_proxy",
            "daily_log_return_sum",
            "trading_days",
            *[column for column in weekly_base.columns if column not in {"Date", "split"}],
        ]
    ]

    validate_outputs(daily, weekly)
    daily.to_csv(daily_output, index=False)
    weekly.to_csv(weekly_output, index=False)
    return daily, weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-file", type=Path, default=DEFAULT_PRICE_FILE)
    parser.add_argument("--daily-output", type=Path, default=DEFAULT_DAILY_OUTPUT)
    parser.add_argument("--weekly-output", type=Path, default=DEFAULT_WEEKLY_OUTPUT)
    args = parser.parse_args()

    daily, weekly = build_har_rv_datasets(
        price_file=args.price_file,
        daily_output=args.daily_output,
        weekly_output=args.weekly_output,
    )
    print(
        f"Wrote {len(daily)} daily rows to {args.daily_output} and "
        f"{len(weekly)} weekly rows to {args.weekly_output}."
    )


if __name__ == "__main__":
    main()
