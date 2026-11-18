#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:47:03 2026

@author: joe
"""

import io
import zipfile
from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Polygon


def parse_custom_time(time_str: str) -> datetime:
    """
    Parses timestamps formatted as 'ddmmyyyy_T_hhmmss' (or 'ddmmyyy_T_hhmmss').
    Example: '15012024_T_143000' -> Jan 15, 2024 at 14:30:00
    """
    date_part, time_part = time_str.split("_T_")

    # Pad leading zero if day is a single digit (7-char date string)
    if len(date_part) == 7:
        date_part = "0" + date_part

    return datetime.strptime(f"{date_part}_T_{time_part}", "%d%m%Y_T_%H%M%S")


def query_marine_cadastre_ais(
    vertices: list[tuple[float, float]], start_str: str, end_str: str
) -> gpd.GeoDataFrame:
    """
    Queries MarineCadastre.gov AIS datasets for ship positions within a 4-vertex polygon
    and time window.

    Parameters:
    -----------
    vertices : list of tuple(float, float)
        List of 4 (longitude, latitude) tuples defining the polygon boundary.
    start_str : str
        Start timestamp in format 'ddmmyyyy_T_hhmmss'.
    end_str : str
        End timestamp in format 'ddmmyyyy_T_hhmmss'.

    Returns:
    --------
    gpd.GeoDataFrame
        Spatial GeoDataFrame containing filtered AIS vessel positions.
    """
    # 1. Parse Start & End Timestamps
    start_dt = parse_custom_time(start_str)
    end_dt = parse_custom_time(end_str)

    # 2. Build Spatial Polygon & Bounding Box for Pre-Filtering
    polygon = Polygon(vertices)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    # 3. Iterate Through Each Day in the Date Range
    current_date = start_dt.date()
    end_date = end_dt.date()

    matching_gdfs = []

    while current_date <= end_date:
        year = current_date.year
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"

        # Official MarineCadastre / NOAA AIS HTTP Download Endpoint
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month}_{day}.zip"
        print(f"Fetching data from: {url}")

        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        # Load daily CSV
                        df = pd.read_csv(f)

                        # Convert timestamp column
                        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

                        # --- Filter 1: Temporal Filter ---
                        mask_time = (df["BaseDateTime"] >= start_dt) & (
                            df["BaseDateTime"] <= end_dt
                        )
                        df_filtered = df[mask_time]

                        if not df_filtered.empty:
                            # --- Filter 2: Bounding Box Pre-Filter (Fast Vectorized Operations) ---
                            mask_bbox = (
                                (df_filtered["LON"] >= min_lon)
                                & (df_filtered["LON"] <= max_lon)
                                & (df_filtered["LAT"] >= min_lat)
                                & (df_filtered["LAT"] <= max_lat)
                            )
                            df_bbox = df_filtered[mask_bbox]

                            if not df_bbox.empty:
                                # Convert to GeoDataFrame
                                gdf = gpd.GeoDataFrame(
                                    df_bbox,
                                    geometry=gpd.points_from_xy(
                                        df_bbox["LON"], df_bbox["LAT"]
                                    ),
                                    crs="EPSG:4326",
                                )

                                # --- Filter 3: Exact Polygon Intersection ---
                                gdf_poly = gdf[gdf.geometry.within(polygon)]

                                if not gdf_poly.empty:
                                    matching_gdfs.append(gdf_poly)

            else:
                print(
                    f"Notice: Data file not found for date {current_date} (HTTP {res.status_code})"
                )

        except Exception as e:
            print(f"Error processing date {current_date}: {e}")

        current_date += timedelta(days=1)

    # 4. Concatenate and Return Final Results
    if matching_gdfs:
        final_gdf = gpd.GeoDataFrame(
            pd.concat(matching_gdfs, ignore_index=True), crs="EPSG:4326"
        )
        return final_gdf
    else:
        print("No AIS signals found matching the spatial-temporal criteria.")
        return gpd.GeoDataFrame()


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # Define 4 GPS vertices (Longitude, Latitude) near the Port of Los Angeles / Long Beach
    polygon_vertices = [
        (-118.2800, 33.7200),  # Top-Left
        (-118.2000, 33.7200),  # Top-Right
        (-118.2000, 33.6800),  # Bottom-Right
        (-118.2800, 33.6800),  # Bottom-Left
    ]

    # Time frame in format ddmmyyyy_T_hhmmss
    start_time = "01012019_T_100000"  # 1st Jan 2024 at 10:00:00
    end_time =   "01012019_T_105000"  # 1st Jan 2024 at 14:00:00

    # Execute query
    ais_data = query_marine_cadastre_ais(
        polygon_vertices, start_time, end_time
    )

    if not ais_data.empty:
        print(
            f"\nRetrieved {len(ais_data)} AIS position records within polygon."
        )
        print(
            ais_data[
                ["BaseDateTime", "MMSI", "VesselName", "LAT", "LON", "SOG"]
            ].head()
        )