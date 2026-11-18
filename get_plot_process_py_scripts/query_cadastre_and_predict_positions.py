#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:40:27 2026

@author: joe
"""

import io
import zipfile
from datetime import datetime, timedelta
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import interp1d
from shapely.geometry import Point, Polygon



import math
import os
import platform
import subprocess
import pandas as pd


def generate_circle_kml_coords(
    lat: float, lon: float, radius_m: float = 100.0, num_points: int = 36
) -> str:
    """Generates KML coordinate string for a circular polygon around a LAT/LON point.

    Parameters:
    -----------
    lat, lon : float
        Center coordinates in decimal degrees.
    radius_m : float
        Radius of the circle in meters.
    num_points : int
        Number of vertices to approximate the circle.
    """
    coords = []
    lat_rad = math.radians(lat)

    for i in range(num_points + 1):
        angle = math.radians(i * (360.0 / num_points))
        # Latitude shift (meters to degrees)
        d_lat = (radius_m * math.cos(angle)) / 111320.0
        # Longitude shift adjusted for latitude convergence
        d_lon = (radius_m * math.sin(angle)) / (111320.0 * math.cos(lat_rad))

        coords.append(f"{lon + d_lon:.7f},{lat + d_lat:.7f},0")

    return " ".join(coords)


def export_to_kml_and_open(
    predictions: pd.DataFrame,
    polygon_vertices: list[tuple[float, float]],
    output_filename: str = "ship_predictions.kml",
    circle_radius_m: float = 150.0,
    auto_open: bool = True,
) -> str:
    """Generates a Google Earth compatible KML overlay with the search area box

    and circles at predicted ship positions, then opens Google Earth.

    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame output from predict_ship_positions().
    polygon_vertices : list of tuple(float, float)
        List of (lon, lat) coordinates defining the bounding polygon.
    output_filename : str
        Target file path for the .kml output.
    circle_radius_m : float
        Radius in meters for the circle overlay at each vessel position.
    auto_open : bool
        If True, executes system command to open Google Earth with the generated file.

    Returns:
    --------
    str
        Absolute file path to generated KML file.
    """
    # 1. Format Bounding Polygon Coordinates (ensure ring closure)
    poly_coords_list = [f"{v[0]},{v[1]},0" for v in polygon_vertices]
    if polygon_vertices[0] != polygon_vertices[-1]:
        poly_coords_list.append(
            f"{polygon_vertices[0][0]},{polygon_vertices[0][1]},0"
        )
    poly_coords_str = " ".join(poly_coords_list)

    # 2. Build XML / KML Header and Styles
    kml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        "    <name>AIS Ship Position Predictions</name>",
        "    <!-- Style for Area Polygon Box -->",
        '    <Style id="boundingPolyStyle">',
        "      <LineStyle><color>ff00ffff</color><width>3</width></LineStyle>",  # Yellow outline
        "      <PolyStyle><color>3300ffff</color></PolyStyle>",  # Semi-transparent yellow fill
        "    </Style>",
        "    <!-- Style for Ship Circles -->",
        '    <Style id="shipCircleStyle">',
        "      <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>",  # Red outline
        "      <PolyStyle><color>660000ff</color></PolyStyle>",  # Semi-transparent red fill
        "    </Style>",
        "    <!-- Bounding Polygon Overlay -->",
        "    <Placemark>",
        "      <name>Search Area Polygon</name>",
        "      <styleUrl>#boundingPolyStyle</styleUrl>",
        "      <Polygon>",
        "        <outerBoundaryIs>",
        "          <LinearRing>",
        f"            <coordinates>{poly_coords_str}</coordinates>",
        "          </LinearRing>",
        "        </outerBoundaryIs>",
        "      </Polygon>",
        "    </Placemark>",
    ]

    # 3. Build Ship Prediction Overlays (Point Labels + Circular Polygons)
    if not predictions.empty:
        for idx, row in predictions.iterrows():
            mmsi = row.get("MMSI", "Unknown")
            name = row.get("VesselName", f"MMSI {mmsi}")
            lat = row.get("LAT")
            lon = row.get("LON")
            sog = row.get("SOG", 0.0)
            cog = row.get("COG", 0.0)
            dt_str = str(row.get("BaseDateTime"))

            if pd.isna(lat) or pd.isna(lon):
                continue

            # Generate Circle Polygon Coordinates around vessel location
            circle_coords_str = generate_circle_kml_coords(
                lat, lon, radius_m=circle_radius_m
            )

            kml_parts.extend(
                [
                    "    <Folder>",
                    f"      <name>Vessel: {name} ({mmsi})</name>",
                    "      <!-- Point Placemark with Label -->",
                    "      <Placemark>",
                    f"        <name>{name}</name>",
                    "        <description><![CDATA[",
                    f"          <b>MMSI:</b> {mmsi}<br/>",
                    f"          <b>Timestamp:</b> {dt_str}<br/>",
                    f"          <b>LAT:</b> {lat:.5f}<br/>",
                    f"          <b>LON:</b> {lon:.5f}<br/>",
                    f"          <b>SOG:</b> {sog} knots<br/>",
                    f"          <b>COG:</b> {cog}°",
                    "        ]]></description>",
                    "        <Point>",
                    f"          <coordinates>{lon},{lat},0</coordinates>",
                    "        </Point>",
                    "      </Placemark>",
                    "      <!-- Physical Circle Overlay -->",
                    "      <Placemark>",
                    f"        <name>{name} Position Circle</name>",
                    "        <styleUrl>#shipCircleStyle</styleUrl>",
                    "        <Polygon>",
                    "          <outerBoundaryIs>",
                    "            <LinearRing>",
                    f"              <coordinates>{circle_coords_str}</coordinates>",
                    "            </LinearRing>",
                    "          </outerBoundaryIs>",
                    "        </Polygon>",
                    "      </Placemark>",
                    "    </Folder>",
                ]
            )

    kml_parts.extend(["  </Document>", "</kml>"])

    # 4. Write KML File to Disk
    abs_filepath = os.path.abspath(output_filename)
    with open(abs_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_parts))

    print(f"\n[KML Export] File successfully written to: {abs_filepath}")

    # 5. Execute Command to Launch Google Earth
    if auto_open:
        print(f"[KML Export] Executing command to launch Google Earth...")
        system_os = platform.system()
        try:
            if system_os == "Linux":
                subprocess.run(["xdg-open", abs_filepath], check=False)
            elif system_os == "Darwin":  # macOS
                subprocess.run(["open", abs_filepath], check=False)
            elif system_os == "Windows":
                os.startfile(abs_filepath)
            else:
                print(
                    f"Unsupported OS '{system_os}' for automatic file launching."
                )
        except Exception as e:
            print(f"Failed to auto-launch Google Earth: {e}")

    return abs_filepath


def parse_custom_time_old(time_str: str) -> datetime:
    """Parses timestamps formatted as 'ddmmyyyy_T_hhmmss' or 'ddmmyyy_T_hhmmss'.

    Example: '15012024_T_143000' -> Jan 15, 2024 at 14:30:00
    """
    date_part, time_part = time_str.split("_T_")
    if len(date_part) == 7:
        date_part = "0" + date_part
    return datetime.strptime(f"{date_part}_T_{time_part}", "%d%m%Y_T_%H%M%S")



def parse_custom_time(time_str: str) -> datetime:
    """Parses timestamps formatted as 'yyyymmyyThhmmss'.

    Example: '15012024_T_143000' -> Jan 15, 2024 at 14:30:00
    """
    date_part, time_part = time_str.split("T")
    if len(date_part) == 7:
        date_part = "0" + date_part

    return datetime.strptime(f"{date_part}T{time_part}", "%Y%m%dT%H%M%S")


def query_marine_cadastre_ais(
    vertices: list[tuple[float, float]], start_str: str, end_str: str
) -> gpd.GeoDataFrame:
    """Queries MarineCadastre.gov AIS datasets for ship positions within a

    4-vertex polygon and time window.

    Parameters:
    -----------
    vertices : list of tuple(float, float)
        List of (longitude, latitude) tuples defining the bounding polygon.
    start_str : str
        Start timestamp in format 'ddmmyyyy_T_hhmmss'.
    end_str : str
        End timestamp in format 'ddmmyyyy_T_hhmmss'.

    Returns:
    --------
    gpd.GeoDataFrame
        Spatial GeoDataFrame containing filtered AIS vessel positions.
    """
    start_dt = parse_custom_time(start_str)
    end_dt = parse_custom_time(end_str)

    polygon = Polygon(vertices)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    current_date = start_dt.date()
    end_date = end_dt.date()
    matching_gdfs = []

    while current_date <= end_date:
        year = current_date.year
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"

        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month}_{day}.zip"
        print(f"Fetching data: {url}")

        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                    csv_name = z.namelist()[0]
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

                        # Temporal filter
                        mask_time = (df["BaseDateTime"] >= start_dt) & (
                            df["BaseDateTime"] <= end_dt
                        )
                        df_filtered = df[mask_time]

                        if not df_filtered.empty:
                            # Spatial bounding box filter
                            mask_bbox = (
                                (df_filtered["LON"] >= min_lon)
                                & (df_filtered["LON"] <= max_lon)
                                & (df_filtered["LAT"] >= min_lat)
                                & (df_filtered["LAT"] <= max_lat)
                            )
                            df_bbox = df_filtered[mask_bbox]

                            if not df_bbox.empty:
                                gdf = gpd.GeoDataFrame(
                                    df_bbox,
                                    geometry=gpd.points_from_xy(
                                        df_bbox["LON"], df_bbox["LAT"]
                                    ),
                                    crs="EPSG:4326",
                                )
                                # Exact polygon intersection
                                gdf_poly = gdf[gdf.geometry.within(polygon)]
                                if not gdf_poly.empty:
                                    matching_gdfs.append(gdf_poly)
            else:
                print(
                    f"Notice: Data file not available for date {current_date} (HTTP {res.status_code})"
                )

        except Exception as e:
            print(f"Error processing date {current_date}: {e}")

        current_date += timedelta(days=1)

    if matching_gdfs:
        return gpd.GeoDataFrame(
            pd.concat(matching_gdfs, ignore_index=True), crs="EPSG:4326"
        )
    else:
        print("No AIS signals found matching the spatial-temporal criteria.")
        return gpd.GeoDataFrame()


def predict_ship_positions(
    ais_data: pd.DataFrame,
    query_times: str | list[str] | datetime | list[datetime],
    min_sog_threshold: float = 0.3,
    max_extrapolation_hours: float = 12.0,
) -> pd.DataFrame:
    """Predicts vessel position, SOG, and COG at target query time(s).

    Includes safeguards against GPS speed jitter and runaway extrapolation:
      - Uses map(lambda x: x.timestamp()) to prevent epoch unit scaling bugs.
      - Treats SOG < min_sog_threshold as stationary (0.0 knots).
      - Applies Linear Dead Reckoning outside the observation window up to max_extrapolation_hours.
      - Prints a warning and holds last position if max_extrapolation_hours is exceeded.

    Parameters:
    -----------
    ais_data : pd.DataFrame or gpd.GeoDataFrame
        Input AIS records returned by query_marine_cadastre_ais.
    query_times : str, datetime, or list
        Target evaluation timestamp(s).
    min_sog_threshold : float
        Minimum SOG in knots to consider a vessel moving (default: 0.3 knots).
    max_extrapolation_hours : float
        Maximum allowed hours for dead-reckoning extrapolation before capping (default: 12.0 hours).

    Returns:
    --------
    pd.DataFrame or gpd.GeoDataFrame
        DataFrame with calculated vessel coordinates and movement vectors.
    """
    if ais_data.empty:
        return ais_data.copy()

    # Normalize query_times parameter
    if not isinstance(query_times, list):
        query_times = [query_times]

    target_dt_list = []
    for q_time in query_times:
        if isinstance(q_time, str):
            target_dt_list.append(pd.to_datetime(parse_custom_time(q_time)))
        else:
            target_dt_list.append(pd.to_datetime(q_time))

    target_secs = np.array([ts.timestamp() for ts in target_dt_list])
    interpolated_records = []
    max_detected_extrap_hours = 0.0

    # Process each unique vessel individually
    for mmsi, group in ais_data.groupby("MMSI"):
        vessel_track = group.sort_values("BaseDateTime").copy()

        # Safe Unix timestamp conversion in seconds
        vessel_track["time_sec"] = vessel_track["BaseDateTime"].map(
            lambda x: x.timestamp()
        )

        # Deduplicate identical timestamp records to satisfy Scipy requirements
        vessel_track = vessel_track.drop_duplicates(subset=["time_sec"])

        n_obs = len(vessel_track)
        if n_obs == 0:
            continue

        time_secs = vessel_track["time_sec"].values
        lats = vessel_track["LAT"].values
        lons = vessel_track["LON"].values
        sogs = (
            vessel_track["SOG"].values if "SOG" in vessel_track.columns else None
        )
        cogs = (
            vessel_track["COG"].values if "COG" in vessel_track.columns else None
        )

        base_template = vessel_track.iloc[-1].to_dict()

        # Case A: Exactly 1 observation available
        if n_obs == 1:
            lat0, lon0 = lats[0], lons[0]
            t0_sec = time_secs[0]
            raw_sog0 = (
                sogs[0] if (sogs is not None and not np.isnan(sogs[0])) else 0.0
            )
            cog0 = (
                cogs[0] if (cogs is not None and not np.isnan(cogs[0])) else None
            )

            # Apply SOG threshold filter
            sog0 = 0.0 if raw_sog0 < min_sog_threshold else raw_sog0

            for target_dt, t_sec in zip(target_dt_list, target_secs):
                record = base_template.copy()
                record["BaseDateTime"] = target_dt
                dt_hours = (t_sec - t0_sec) / 3600.0
                abs_dt_hours = abs(dt_hours)

                if abs_dt_hours > max_detected_extrap_hours:
                    max_detected_extrap_hours = abs_dt_hours

                if (
                    abs_dt_hours <= max_extrapolation_hours
                    and sog0 > 0.0
                    and cog0 is not None
                ):
                    # Valid dead reckoning
                    dist_nm = sog0 * dt_hours
                    cog_rad = np.radians(cog0)
                    record["LAT"] = lat0 + (dist_nm * np.cos(cog_rad) / 60.0)
                    avg_lat_rad = np.radians(lat0)
                    record["LON"] = lon0 + (
                        dist_nm
                        * np.sin(cog_rad)
                        / (60.0 * np.cos(avg_lat_rad))
                    )
                    record["SOG"] = float(sog0)
                else:
                    # Stationary hold or exceeded threshold
                    record["LAT"] = lat0
                    record["LON"] = lon0
                    record["SOG"] = 0.0 if abs_dt_hours > max_extrapolation_hours else float(sog0)

                if cogs is not None:
                    record["COG"] = float(cog0) if cog0 is not None else np.nan

                if "geometry" in record and record["geometry"] is not None:
                    record["geometry"] = Point(record["LON"], record["LAT"])

                interpolated_records.append(record)
            continue

        # Case B: 2 or more observations available
        kind = (
            "linear"
            if n_obs == 2
            else ("quadratic" if n_obs == 3 else "cubic")
        )

        lat_interp = interp1d(time_secs, lats, kind=kind)
        lon_interp = interp1d(time_secs, lons, kind=kind)
        sog_interp = (
            interp1d(time_secs, sogs, kind=kind) if sogs is not None else None
        )

        if cogs is not None:
            cog_rad = np.radians(cogs)
            cog_x_interp = interp1d(time_secs, np.sin(cog_rad), kind=kind)
            cog_y_interp = interp1d(time_secs, np.cos(cog_rad), kind=kind)
        else:
            cog_x_interp, cog_y_interp = None, None

        min_t, max_t = time_secs[0], time_secs[-1]

        for target_dt, t_sec in zip(target_dt_list, target_secs):
            record = base_template.copy()
            record["BaseDateTime"] = target_dt

            # Sub-case B1: Extrapolating into Future
            if t_sec > max_t:
                dt_hours = (t_sec - max_t) / 3600.0
                if dt_hours > max_detected_extrap_hours:
                    max_detected_extrap_hours = dt_hours

                last_lat, last_lon = lats[-1], lons[-1]
                raw_last_sog = (
                    sogs[-1]
                    if (sogs is not None and not np.isnan(sogs[-1]))
                    else 0.0
                )
                last_cog = (
                    cogs[-1]
                    if (cogs is not None and not np.isnan(cogs[-1]))
                    else 0.0
                )

                effective_sog = (
                    0.0 if raw_last_sog < min_sog_threshold else raw_last_sog
                )

                if dt_hours > max_extrapolation_hours:
                    # Threshold exceeded: cap extrapolation to stationary position
                    pred_lat, pred_lon = last_lat, last_lon
                    pred_sog = 0.0
                else:
                    dist_nm = effective_sog * dt_hours
                    cog_rad = np.radians(last_cog)
                    pred_lat = last_lat + (dist_nm * np.cos(cog_rad) / 60.0)
                    pred_lon = last_lon + (
                        dist_nm
                        * np.sin(cog_rad)
                        / (60.0 * np.cos(np.radians(last_lat)))
                    )
                    pred_sog = effective_sog

                pred_cog = last_cog

            # Sub-case B2: Extrapolating into Past
            elif t_sec < min_t:
                dt_hours = (t_sec - min_t) / 3600.0  # Negative hours
                abs_dt_hours = abs(dt_hours)
                if abs_dt_hours > max_detected_extrap_hours:
                    max_detected_extrap_hours = abs_dt_hours

                first_lat, first_lon = lats[0], lons[0]
                raw_first_sog = (
                    sogs[0]
                    if (sogs is not None and not np.isnan(sogs[0]))
                    else 0.0
                )
                first_cog = (
                    cogs[0]
                    if (cogs is not None and not np.isnan(cogs[0]))
                    else 0.0
                )

                effective_sog = (
                    0.0 if raw_first_sog < min_sog_threshold else raw_first_sog
                )

                if abs_dt_hours > max_extrapolation_hours:
                    # Threshold exceeded: cap extrapolation
                    pred_lat, pred_lon = first_lat, first_lon
                    pred_sog = 0.0
                else:
                    dist_nm = effective_sog * dt_hours
                    cog_rad = np.radians(first_cog)
                    pred_lat = first_lat + (dist_nm * np.cos(cog_rad) / 60.0)
                    pred_lon = first_lon + (
                        dist_nm
                        * np.sin(cog_rad)
                        / (60.0 * np.cos(np.radians(first_lat)))
                    )
                    pred_sog = effective_sog

                pred_cog = first_cog

            # Sub-case B3: Spline Interpolation within Observation Window
            else:
                pred_lat = float(lat_interp(t_sec))
                pred_lon = float(lon_interp(t_sec))

                raw_sog = (
                    max(0.0, float(sog_interp(t_sec)))
                    if sog_interp
                    else base_template.get("SOG", 0.0)
                )
                pred_sog = 0.0 if raw_sog < min_sog_threshold else raw_sog

                if cog_x_interp and cog_y_interp:
                    x_val = float(cog_x_interp(t_sec))
                    y_val = float(cog_y_interp(t_sec))
                    pred_cog = float(np.degrees(np.arctan2(x_val, y_val)) % 360.0)
                else:
                    pred_cog = base_template.get("COG")

            record["LAT"] = pred_lat
            record["LON"] = pred_lon
            if sogs is not None:
                record["SOG"] = pred_sog
            if cogs is not None:
                record["COG"] = pred_cog

            if "geometry" in record and record["geometry"] is not None:
                record["geometry"] = Point(record["LON"], record["LAT"])

            interpolated_records.append(record)

    # Output threshold warning if extrapolation limit was breached
    if max_detected_extrap_hours > max_extrapolation_hours:
        exceeded_by = max_detected_extrap_hours - max_extrapolation_hours
        print("\n" + "=" * 80)
        print("[WARNING] EXTRAPOLATION THRESHOLD EXCEEDED")
        print("=" * 80)
        print(
            f"  • Maximum allowed extrapolation threshold : {max_extrapolation_hours:.2f} hours"
        )
        print(
            f"  • Max requested extrapolation gap        : {max_detected_extrap_hours:.2f} hours ({max_detected_extrap_hours / 24.0:.2f} days)"
        )
        print(
            f"  • Exceeded threshold by                   : {exceeded_by:.2f} hours ({exceeded_by / 24.0:.2f} days)"
        )
        print(
            "  • Action taken                            : Long-range predictions were capped at"
        )
        print(
            "                                              the boundary position to prevent unrealistic drift."
        )
        print("=" * 80 + "\n")

    if not interpolated_records:
        return pd.DataFrame(columns=ais_data.columns)

    if isinstance(ais_data, gpd.GeoDataFrame):
        predicted_df = gpd.GeoDataFrame(
            interpolated_records, crs=ais_data.crs
        )
    else:
        predicted_df = pd.DataFrame(interpolated_records)

    return predicted_df[ais_data.columns]


# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    polygon_vertices = [
        (-118.2800, 33.7200),
        (-118.2000, 33.7200),
        (-118.2000, 33.6800),
        (-118.2800, 33.6800),
    ]

    # Query AIS data
    ais_data = query_marine_cadastre_ais(
        vertices=polygon_vertices,
        start_str="20190101T110000",
        end_str="20190101T120000",
    )

    if not ais_data.empty:
        eval_times = ["20190101T113000"]

        # Generate predictions
        predictions = predict_ship_positions(
            ais_data=ais_data,
            query_times=eval_times,
            min_sog_threshold=0.3,
            max_extrapolation_hours=12.0,
        )

        # Generate KML overlay and open Google Earth automatically
        kml_path = export_to_kml_and_open(
            predictions=predictions,
            polygon_vertices=polygon_vertices,
            output_filename="ship_predictions.kml",
            circle_radius_m=100.0,  # 200m circle around ships
            auto_open=True,
        )


'''
if __name__ == "__main__":
    polygon_vertices = [
        (-118.2800, 33.7200),
        (-118.2000, 33.7200),
        (-118.2000, 33.6800),
        (-118.2800, 33.6800),
    ]

    # Query AIS data for Jan 1, 2019
    ais_data = query_marine_cadastre_ais(
        vertices=polygon_vertices,
        start_str="01012019_T_110000",
        end_str="01012019_T_120000",
    )

    if not ais_data.empty:
        # Evaluate target query timestamps:
        # 1. In-window time: '01012019_T_113000'
        # 2. Out-of-window time (17 hours later): '02012019_T_050000' -> Triggers warning (> 12h threshold)!
        eval_times = ["01012019_T_113000", "01012019_T_140000"]

        predictions = predict_ship_positions(
            ais_data=ais_data,
            query_times=eval_times,
            min_sog_threshold=0.3,
            max_extrapolation_hours=12.0,
        )

        print("Calculated Vessel Positions:")
        print(
            predictions[
                ["MMSI", "VesselName", "BaseDateTime", "LAT", "LON", "SOG", "COG"]
            ].head(10)
        )
        
'''