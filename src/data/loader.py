"""
Load and do initial cleaning of the UCI Individual Household Electric
Power Consumption dataset.

Expected raw file: data/raw/household_power_consumption.txt
Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The raw file is ';'-separated, uses '?' for missing values, and has
Date/Time as separate columns.
"""

from pathlib import Path

import pandas as pd

# Anchor to the project root (two levels up from src/data/loader.py) rather
# than the current working directory. This matters because Jupyter's cwd is
# the notebook's own folder (notebooks/), not the repo root — a relative
# path like "data/processed/..." would silently resolve to
# "notebooks/data/processed/..." instead of the real data/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data/raw/household_power_consumption.txt"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/household_power_consumption.parquet"

COLUMN_DTYPES = {
    "Global_active_power": "float64",
    "Global_reactive_power": "float64",
    "Voltage": "float64",
    "Global_intensity": "float64",
    "Sub_metering_1": "float64",
    "Sub_metering_2": "float64",
    "Sub_metering_3": "float64",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw semicolon-separated UCI file into a DataFrame with a
    proper datetime index. '?' is treated as missing (NaN)."""
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()

    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Save to parquet. Tries pyarrow first; falls back to fastparquet if
    pyarrow's extension-type registration is in a bad state (a known
    pandas/pyarrow issue that shows up as `ArrowKeyError: No type extension
    with name arrow.py_extension_type found` after re-running import cells
    or with %autoreload in Jupyter — restarting the kernel usually fixes it,
    but this fallback keeps the pipeline working either way)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, engine="pyarrow")
    except Exception:
        df.to_parquet(path, engine="fastparquet")


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows, {df.index.min()} to {df.index.max()}")
    print(f"Missing values per column:\n{df.isna().sum()}")
    save_processed(df)
    print(f"Saved cleaned parquet to {PROCESSED_PATH}")
