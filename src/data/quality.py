"""Data quality diagnostics: missing values, time gaps, impossible values."""

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column."""
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(3)
    return pd.DataFrame({"missing_count": counts, "missing_pct": pct}).sort_values(
        "missing_count", ascending=False
    )


def time_gap_report(df: pd.DataFrame, expected_freq: str = "1min") -> pd.DataFrame:
    """Find gaps in the datetime index larger than the expected frequency."""
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    missing_timestamps = full_range.difference(df.index)

    gaps = []
    if len(missing_timestamps) > 0:
        diffs = missing_timestamps.to_series().diff()
        # group consecutive missing timestamps into gap blocks
        breaks = diffs != pd.Timedelta(expected_freq)
        group_id = breaks.cumsum()
        for _, block in missing_timestamps.to_series().groupby(group_id):
            gaps.append(
                {"gap_start": block.index[0], "gap_end": block.index[-1], "n_missing": len(block)}
            )
    return pd.DataFrame(gaps, columns=["gap_start", "gap_end", "n_missing"])


def impossible_value_report(df: pd.DataFrame) -> dict:
    """Flag physically implausible readings."""
    checks = {}
    if "Global_active_power" in df.columns:
        checks["negative_active_power"] = int((df["Global_active_power"] < 0).sum())
    if "Voltage" in df.columns:
        checks["voltage_out_of_range"] = int(
            ((df["Voltage"] < 200) | (df["Voltage"] > 260)).sum()
        )
    if "Global_intensity" in df.columns:
        checks["negative_intensity"] = int((df["Global_intensity"] < 0).sum())
    return checks


def duplicate_timestamp_report(df: pd.DataFrame) -> int:
    return int(df.index.duplicated().sum())
