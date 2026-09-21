"""Build the train/validation/test datasets from the raw source files.
Usage
-----
    python 01_build_datasets.py                     # raw data/ -> project root
    python 01_build_datasets.py --raw-dir "raw data" --out-dir .
    python 01_build_datasets.py --summary           # also write the data-availability table

Outputs
-------
    train_dataset.csv   2016-01-20 .. 2021-12-29
    valid_dataset.csv   2022-01-05 .. 2022-12-28
    test_dataset.csv    2023-01-04 .. 2025-12-31
    data_availability.csv  (with --summary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

# --- Series that are lagged one week so that x_t is F_{t-1}-measurable. The
#     two event dummies are deterministic functions of the calendar and are
#     deliberately NOT lagged.
COVARIATE_COLS = [
    "GCPU_baseline", "l_t", "c_t", "itraxx",
    "CPU_EU_step", "CPU_EU_spline", "GEPU_current", "GEPU_ppp",
    "rate_10y", "Term_Spread", "VSTOXX", "TTF_return", "Brent_return",
]

CPU_COUNTRIES = ["CPU_DEU", "CPU_FRA", "CPU_ITA", "CPU_ESP", "CPU_IRL"]

REQUIRED_FILES = [
    "GCPU.xlsx",
    "clean_energy_etf_IQQH.csv",
    "clean_energy_etf_ICLN.csv",
    "clean_energy_etf_TAN.csv",
    "Libro1.xlsx",
    "iTraxx.xlsx",
    "Global Clean Energy Transition Index.xls",
    "cpu_all_countries_monthly.csv",
    "Global_Economic_Policy_Uncertainty_EPU.xlsx",
    "rate_3month.csv",
    "rate_10yield.csv",
    "v2tx.txt",
    "ICE Dutch TTF Natural Gas Futures Historical Data.csv",
    "DCOILBRENTEU.csv",
]

SPLITS = {
    "train": ("2016-01-01", "2022-01-01"),
    "valid": ("2022-01-01", "2023-01-01"),
    "test": ("2023-01-01", None),
}


def check_inputs(raw: Path) -> None:
    """Fail early and by name rather than deep inside a merge."""
    missing = [f for f in REQUIRED_FILES if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw input file(s) in {}:\n  ".format(raw)
            + "\n  ".join(missing)
        )


def weekly_return(path: Path, col: str, name: str, skiprows=0) -> pd.DataFrame:
    """Wednesday-to-Wednesday percentage log return of a price column.

    The series is resampled to the W-WED grid taking the LAST observation in
    each week before differencing, so a market holiday on a Wednesday carries
    the most recent prior price rather than dropping the week.
    """
    data = pd.read_csv(path, skiprows=skiprows)
    data.rename(columns={data.columns[0]: "Date"}, inplace=True)
    data["Date"] = pd.to_datetime(data["Date"])
    data[col] = pd.to_numeric(data[col], errors="coerce")
    weekly = (
        data.set_index("Date")[[col]]
        .resample("W-WED")
        .last()
        .dropna(subset=[col])
        .reset_index()
    )
    weekly[name] = 100 * np.log(weekly[col]).diff()
    return weekly[["Date", name]]


def wednesday_slice(frame: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    return frame[frame[date_col].dt.weekday == 2].copy()


def build(raw: Path) -> pd.DataFrame:
    check_inputs(raw)

    # --- weekly Wednesday spine ------------------------------------------------
    df = pd.DataFrame(
        {"Date": pd.date_range(start="2016-01-01", end="2025-12-31", freq="W-WED")}
    )

    # --- global climate policy uncertainty (daily index, PPP-GDP weighted) -----
    gcpu = pd.read_excel(raw / "GCPU.xlsx", sheet_name="GCPU_daily")
    gcpu["GCPU_baseline"] = pd.to_numeric(gcpu["GCPU(PPP-Adjusted GDP)"], errors="coerce")
    gcpu_wed = gcpu[gcpu["date"].dt.weekday == 2][["date", "GCPU_baseline"]]
    df = pd.merge(df, gcpu_wed, left_on="Date", right_on="date", how="left").drop(columns=["date"])

    # --- equity returns --------------------------------------------------------
    df = pd.merge(df, weekly_return(raw / "clean_energy_etf_IQQH.csv", "Close",
                                    "y_IQQH_EUR", skiprows=[1, 2]), on="Date", how="left")
    df = pd.merge(df, weekly_return(raw / "clean_energy_etf_ICLN.csv", "Close",
                                    "y_ICLN", skiprows=[1, 2]), on="Date", how="left")
    df = pd.merge(df, weekly_return(raw / "clean_energy_etf_TAN.csv", "Close",
                                    "y_TAN", skiprows=[1, 2]), on="Date", how="left")

    stoxx = pd.read_excel(raw / "Libro1.xlsx", sheet_name="stoxx600 industrial")
    stoxx["y_stoxx"] = pd.to_numeric(stoxx["TRDPRC_1"], errors="coerce").ffill()
    stoxx.rename(columns={"Timestamp": "Date"}, inplace=True)
    stoxx_wed = wednesday_slice(stoxx)[["Date", "y_stoxx"]]
    stoxx_wed["y_stoxx"] = 100 * np.log(stoxx_wed["y_stoxx"]).diff()
    df = pd.merge(df, stoxx_wed, on="Date", how="left")

    idx = pd.read_excel(raw / "Global Clean Energy Transition Index.xls", skiprows=6)
    idx.columns = idx.columns.str.strip()
    idx["Date"] = pd.to_datetime(idx["Effective date"], errors="coerce")
    price_col = "S&P Global Clean Energy Transition Index (USD)"
    idx[price_col] = idx[price_col].ffill()
    idx_wed = wednesday_slice(idx).sort_values("Date").dropna(subset=[price_col])
    idx_wed["y_Global_Clean_Index"] = 100 * np.log(idx_wed[price_col]).diff()
    df = pd.merge(df, idx_wed[["Date", "y_Global_Clean_Index"]], on="Date", how="left")

    # --- European climate policy uncertainty, monthly --------------------------
    cpu = pd.read_csv(raw / "cpu_all_countries_monthly.csv")
    cpu.drop(columns="cit", inplace=True)
    cpu["ym"] = pd.to_datetime(
        cpu["year"].astype(int).astype(str) + "-" + cpu["month"].astype(int).astype(str) + "-01"
    ).dt.to_period("M")
    for c in CPU_COUNTRIES:
        cpu[c] = zscore(cpu[c], nan_policy="omit")
    cpu["CPU_EU"] = cpu[CPU_COUNTRIES].mean(axis=1)

    df["ym"] = df["Date"].dt.to_period("M")
    df = pd.merge(df, cpu[["ym", "CPU_EU"]], on="ym", how="left").rename(
        columns={"CPU_EU": "CPU_EU_step"}
    )

    # cubic-spline interpolation of the same monthly series onto the daily grid
    cpu["Interpolation_Date"] = cpu["ym"].dt.to_timestamp()
    daily = pd.DataFrame(
        {"Date": pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")}
    )
    daily = pd.merge(daily, cpu[["Interpolation_Date", "CPU_EU"]],
                     left_on="Date", right_on="Interpolation_Date",
                     how="left").drop(columns=["Interpolation_Date"])
    daily = daily.set_index("Date")
    daily["CPU_EU"] = daily["CPU_EU"].interpolate(method="cubicspline", limit_area="inside")
    daily = daily.reset_index()
    df = pd.merge(df, daily[["Date", "CPU_EU"]].rename(columns={"CPU_EU": "CPU_EU_spline"}),
                  on="Date", how="left")

    # --- global economic policy uncertainty ------------------------------------
    gepu = pd.read_excel(raw / "Global_Economic_Policy_Uncertainty_EPU.xlsx")
    gepu["ym"] = pd.to_datetime(
        gepu["Year"].astype(str) + "-" + gepu["Month"].astype(str) + "-01"
    ).dt.to_period("M")
    for c in ("GEPU_current", "GEPU_ppp"):
        gepu[c] = pd.to_numeric(gepu[c], errors="coerce")
    df = pd.merge(df, gepu[["ym", "GEPU_current", "GEPU_ppp"]], on="ym", how="left")

    # --- EUA carbon price: log level and weekly log return ---------------------
    carbon = pd.read_excel(raw / "Libro1.xlsx", sheet_name="FEUAc1")
    carbon["EUA_Carbon"] = pd.to_numeric(carbon["SETTLE"], errors="coerce")
    carbon.rename(columns={"Timestamp": "Date"}, inplace=True)
    carbon["l_t"] = 100 * np.log(carbon["EUA_Carbon"])
    carbon_wed = wednesday_slice(carbon)
    carbon_wed["c_t"] = carbon_wed["l_t"].diff()
    df = pd.merge(df, carbon_wed[["Date", "l_t", "c_t"]], on="Date", how="left")

    # --- iTraxx Crossover: weekly FIRST DIFFERENCE of the index ---------------
    itraxx = pd.read_excel(raw / "iTraxx.xlsx")
    itraxx["itraxx"] = pd.to_numeric(itraxx["TRDPRC_1"], errors="coerce")
    itraxx.rename(columns={"Timestamp": "Date"}, inplace=True)
    itraxx_wed = wednesday_slice(itraxx)
    itraxx_wed["itraxx"] = itraxx_wed["itraxx"].diff()
    df = pd.merge(df, itraxx_wed[["Date", "itraxx"]], on="Date", how="left")

    # --- euro-area rates and term spread (monthly, held constant within month) -
    # NOTE: rate_2yield.csv holds FRED series IR3TIB01EZM156N, which is the
    # 3-month interbank rate, not a 2-year yield, despite the file name. The
    # term spread built here is therefore the 10-year government bond yield
    # minus the 3-month interbank rate. Documented rather than silently renamed
    # so the thesis text and the data agree.
    r_short = pd.read_csv(raw / "rate_3month.csv", parse_dates=["observation_date"])
    r_long = pd.read_csv(raw / "rate_10yield.csv", parse_dates=["observation_date"])
    rates = pd.merge(r_short, r_long, on="observation_date")
    rates["rate_short"] = pd.to_numeric(rates["IR3TIB01EZM156N"], errors="coerce")
    rates["rate_10y"] = pd.to_numeric(rates["IRLTLT01EZM156N"], errors="coerce")
    rates["Term_Spread"] = rates["rate_10y"] - rates["rate_short"]
    rates["ym"] = rates["observation_date"].dt.to_period("M")
    df = pd.merge(df, rates[["ym", "rate_10y", "Term_Spread"]], on="ym", how="left")

    # --- VSTOXX in logs --------------------------------------------------------
    v2tx = pd.read_csv(raw / "v2tx.txt", sep=";", parse_dates=["Date"], dayfirst=True)
    v2tx["VSTOXX"] = 100 * np.log(pd.to_numeric(v2tx["Indexvalue"], errors="coerce"))
    df = pd.merge(df, wednesday_slice(v2tx)[["Date", "VSTOXX"]], on="Date", how="left")

    # --- energy returns --------------------------------------------------------
    df = pd.merge(df, weekly_return(raw / "ICE Dutch TTF Natural Gas Futures Historical Data.csv",
                                    "Price", "TTF_return"), on="Date", how="left")
    df = pd.merge(df, weekly_return(raw / "DCOILBRENTEU.csv", "DCOILBRENTEU",
                                    "Brent_return"), on="Date", how="left")
    df.drop(columns=["ym"], inplace=True)

    # --- event dummies ---------------------------------------------------------
    df["COVID_dummy"] = ((df["Date"] >= "2020-02-19") & (df["Date"] <= "2021-04-30")).astype(int)
    df["Energy_crisis_dummy"] = ((df["Date"] >= "2021-01-01") & (df["Date"] <= "2022-12-31")).astype(int)

    # --- lag every continuous covariate by one week ----------------------------
    for col in COVARIATE_COLS:
        df[col] = df[col].shift(1)

    # the first row has no lagged covariates by construction
    df = df.drop(df.index[0])
    return df


def split_and_write(df: pd.DataFrame, out: Path) -> dict:
    written = {}
    for name, (lo, hi) in SPLITS.items():
        mask = df["Date"] >= lo
        if hi is not None:
            mask &= df["Date"] < hi
        part = df[mask].copy()
        path = out / f"{name}_dataset.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        written[name] = (path, part)
    return written


def availability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "Date":
            continue
        present = df[["Date", col]].dropna(subset=[col])
        rows.append({
            "Variable": col,
            "Start date": present["Date"].min().date() if len(present) else None,
            "End date": present["Date"].max().date() if len(present) else None,
            "Missing %": round(df[col].isna().mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="raw data", type=Path)
    parser.add_argument("--out-dir", default=".", type=Path)
    parser.add_argument("--summary", action="store_true",
                        help="also write data_availability.csv")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = build(args.raw_dir)
    written = split_and_write(df, args.out_dir)

    print(f"built {len(df)} weekly rows, {df['Date'].min().date()} .. {df['Date'].max().date()}")
    for name, (path, part) in written.items():
        print(f"  {name:5s} {len(part):4d} rows  "
              f"{part['Date'].min().date()} .. {part['Date'].max().date()}  -> {path}")

    if args.summary:
        table = availability_table(df)
        path = args.out_dir / "data_availability.csv"
        table.to_csv(path, index=False)
        print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
