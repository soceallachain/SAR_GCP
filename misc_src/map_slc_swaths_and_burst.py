#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:27:43 2025
Script to determine spatial resolution in a SLC file from annotated data
@author: joe
"""
from  SAR_code_module import *

import shapely.geometry as sg
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import reverse_geocoder as rg
import pyproj
import simplekml



#load annotation file
slc_annotation_dir  ='/home/joe/Work/Projects/RSAR/Datasets/xView3/Data/Misc/Tests/TestData/S1A_IW_SLC_1SDV_20200328T051410_20200328T051438_031871_03ADC3_1ABD/SLC/S1A_IW_SLC__1SDV_20200328T051410_20200328T051438_031871_03ADC3_1ABD.SAFE/annotation/'
slc_annotation_files =['s1a-iw1-slc-vh-20200328t051410-20200328t051438-031871-03adc3-001.xml',
                      's1a-iw2-slc-vh-20200328t051411-20200328t051436-031871-03adc3-002.xml',
                      's1a-iw3-slc-vh-20200328t051412-20200328t051437-031871-03adc3-003.xml']



#get swath information
IW_mode = 2
annotation_file = f'{slc_annotation_dir}{slc_annotation_files[IW_mode -1]}'
burst_information = getSwathBurstGridGPSBoundaries(annotation_file)
burst_number_to_plot =1
burst_index=burst_number_to_plot -1
gps_coords_of_swath = burst_information[2][burst_index]

# Convert coordinates to floats
coords = [(float(x), float(y)) for x, y in gps_coords_of_swath]

# Determine the UTM zone
# Assuming the ROI is in the northern hemisphere
zone = int(np.floor((np.mean([x for x, y in coords]) + 180) / 6) + 1)
utm_crs = pyproj.CRS(f"EPSG:326{zone}")


# Separate longitudes and latitudes
lons = [x for x, y in coords]
lats = [y for x, y in coords]

# Project the coordinates to UTM
projector = pyproj.Transformer.from_crs(pyproj.CRS("EPSG:4326"), utm_crs, always_xy=True).transform
utm_x, utm_y = projector(lons, lats)

# Combine into tuples
utm_coords = list(zip(utm_x, utm_y))
# Create a Polygon object in UTM
utm_polygon = Polygon(utm_coords)

# Calculate the area of the polygon in square km
area = utm_polygon.area
area = area/1e6

# Calculate the lengths of the sides in meters
side_lengths = []
for i in range(len(utm_coords)):
    p1 = utm_coords[i]
    p2 = utm_coords[(i+1) % len(utm_coords)]  # Wrap around for the last point
    side_length = LineString([p1, p2]).length
    side_lengths.append(side_length)

# Identify the country using reverse geocoding
center = np.mean([x for x, y in coords]), np.mean([y for x, y in coords])
country = rg.search((center[1], center[0]))[0]['cc']

# Plot the polygon on a map
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle='--')
ax.add_geometries([Polygon(coords)], ccrs.PlateCarree(), facecolor='none', edgecolor='red')
ax.set_title(f"ROI in/nearest {country}")
plt.show()

# Print results
print(f"Area of ROI: {area:.1f} km^2")
side_lengths_arr = np.array(side_lengths)
for elem in side_lengths_arr:
    print(f"Side lengths are : {elem/1000:.1f} km")



# Convert coordinates to floats
#coords = [(float(x), float(y)) for x, y in gps_coords]

# Create a KML object
kml = simplekml.Kml()

# Add a polygon to the KML object
pol = kml.newpolygon(name='ROI')
closed_coords = coords + [coords[0]]
pol.outerboundaryis.coords = [(x, y) for x, y in closed_coords]  # Note: KML uses (lon, lat) order
#pol.style.polystyle.color = simplekml.Color.aqua  # Default color
#pol.style.polystyle.color.a = 64  # Set transparency to about 25% (64 out of 255)

# Alternatively, set the color directly with transparency
# pol.style.polystyle.color = simplekml.Color.hexa('40ffffff')  # White with 25% transparency
pol.style.polystyle.color = simplekml.Color.hexa('60ff0000')  # Red with 25% transparency
pol.style.linestyle.color = simplekml.Color.red  # Red outline
pol.style.linestyle.width = 2  # Thickness in pixels
#pol.style.linestyle.color = 'ff0000'  # 
#kml.newpolygon(name='ROI', coords=[(x, y) for x, y in coords])

# Save the KML file
kml.save('roi.kml')



#get gps coordinates of longest and shortest side

#get length of longest and shorted side in pixels

#get length of longest and shortest side in m