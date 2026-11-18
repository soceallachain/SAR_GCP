#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:04:41 2026

@author: joe
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone


def get_ndbc_weather(station_id: str, target_utc: str) -> pd.Series:
    """
    Fetches and parses historical NOAA NDBC standard meteorological data
    for a given station and finds the observation closest to target_utc.

    Parameters
    ----------
    station_id : str
        NDBC station ID (e.g., '42035' for Galveston, TX).
    target_utc : str
        Target timestamp in UTC (e.g., '2023-08-15 12:15:00').

    Returns
    -------
    pd.Series
        Closest meteorological record with time offset and cleaned physical units.
    """
    dt = pd.to_datetime(target_utc, utc=True)
    year = dt.year

    # NDBC historical annual standard meteorological file URL
    url = (
        f"https://www.ndbc.noaa.gov/view_text_file.php?"
        f"filename={station_id.lower()}h{year}.txt.gz&dir=data/historical/stdmet/"
    )

    print(f"Downloading NDBC {station_id} data for {year}...")

    # Read space-delimited file; skip row 1 which contains unit strings (#yr mo dy...)
    df = pd.read_csv(
        url,
        sep=r"\s+",
        skiprows=[1],
        na_values=["MM", "99.0", "99.00", "999", "999.0", "9999.0"],
    )

    # Clean leading '#' on the '#YY' column header
    df.columns = [col.lstrip("#") for col in df.columns]

    # Reconstruct UTC DatetimeIndex (handles both 'YY' and 'YYYY' headers)
    year_col = "YY" if "YY" in df.columns else "YYYY"
    df["timestamp_utc"] = pd.to_datetime(
        {
            "year": df[year_col],
            "month": df["MM"],
            "day": df["DD"],
            "hour": df["hh"],
            "minute": df["mm"],
        },
        utc=True,
    )
    df = df.set_index("timestamp_utc").sort_index()

    # Locate the observation nearest to the SAR acquisition timestamp
    idx_pos = df.index.get_indexer([dt], method="nearest")[0]
    closest_row = df.iloc[idx_pos].copy()
    matched_time = df.index[idx_pos]

    closest_row["matched_utc"] = matched_time
    closest_row["time_diff_minutes"] = abs((matched_time - dt).total_seconds()) / 60.0

    return closest_row


# ==========================================
# Example Usage: Match a SAR Acquisition Time
# ==========================================
if __name__ == "__main__":
    station = "42035"  # Galveston, TX (22 NM East of Galveston)
    sar_acquisition_utc = "2023-07-14 00:26:30"

    record = get_ndbc_weather(station, sar_acquisition_utc)

    print("\n--- Closest Buoy Observation ---")
    print(f"Target UTC:       {sar_acquisition_utc}")
    print(f"Buoy Timestamp:   {record['matched_utc']} (Δt = {record['time_diff_minutes']:.1f} min)")
    print(f"Wind Speed:       {record['WSPD']} m/s")
    print(f"Wind Gust:        {record['GST']} m/s")
    print(f"Wind Direction:   {record['WDIR']}°")
    print(f"Sig. Wave Height: {record['WVHT']} m")
    print(f"Dominant Period:  {record['DPD']} s")
    print(f"Sea Surface Temp: {record['WTMP']} °C")