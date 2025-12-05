import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_sea_status(gps_coords, plot_map):
    """
    Function to return whether ROI specified by gps_coords is wholly sea
    based or whether there is land within the boundary.
    User can determine if a map of the location of the ROI is plotted
    Returns the status of the ROI: True for land, False if sea
    
    Parameters
    ----------
    gps_coords : Array of 4 gps locations defining the vertices of a polygon
                 Each gps location is in [latitude, longitude] format
                 The vertices are ordered as [bottom-left, top-left, top-right
                 and bottom right]
    plot_map : Boolean - True => plot map; False => Do not plot map

    Returns
    -------
    is_land : Boolean - True if ROI contains land; False if wholly sea based
    country : String - country closest to or encompassing ROI
    min_dist : Real - distance from centre of ROI to land
        
    """
    # Reformat GPS coordinates for Polygon ([lon, lat] instead of [lat, lon])
    #gps_coords_lon_lat = [(lon, lat) for lat, lon in gps_coords]
    #print(f'Re-formated gps_coords :{gps_coords_lon_lat}')
    print(f'gps_coords :{gps_coords}')
    gps_coords_lon_lat = gps_coords
    # Convert GPS coordinates to a polygon
    roi_polygon = Polygon(gps_coords_lon_lat)
    print(f'ROI_POLYGON : {roi_polygon}')
    
    # Calculate the center of the ROI
    centroid = roi_polygon.centroid
    center_lat, center_lon = centroid.y, centroid.x
    
    # Calculate the bounding box of the ROI
    min_lon, min_lat, max_lon, max_lat = roi_polygon.bounds
    
    # Expand the bounding box to 50 times the area
    lon_margin = (max_lon - min_lon) * 3.5
    lat_margin = (max_lat - min_lat) * 3.5
    expanded_bbox = [min_lon - lon_margin, max_lon + lon_margin, min_lat - lat_margin, max_lat + lat_margin]
    
    # Set up the map
    if plot_map:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, edgecolor='black')
    
        # Plot the ROI
        roi_x, roi_y = roi_polygon.exterior.xy
        ax.plot(roi_x, roi_y, color='red', linewidth=2, label='ROI')
        ax.scatter(center_lon, center_lat, color='blue', label='ROI Center')
    
    # Load global land geometries
    land_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    land_geometry = land_gdf.geometry.unary_union  # Combine all land polygons
    
    # Check if the ROI is on land or sea
    geolocator = Nominatim(user_agent="geoapi")
    is_land = land_geometry.intersects(roi_polygon)
    
    if is_land:
        # Identify the country
        location = geolocator.reverse((center_lat, center_lon), language='en')
        country = location.raw.get('address', {}).get('country', 'Unknown')
        min_dist = 0
        print(f"The ROI is land-based and is located in {country}.")
    else:
        # If sea-based, find the nearest land-based location
        print("The ROI is sea-based.")
        min_dist = float('inf')
        nearest_land = None
        
        # Iterate through all exterior points of the polygons in the land geometry
        for polygon in land_geometry.geoms:  # Iterate through each polygon in the MultiPolygon
            for point in polygon.exterior.coords:  # Iterate through the exterior points of the polygon
                land_point = Point(point)
                dist = geodesic((center_lat, center_lon), (land_point.y, land_point.x)).kilometers
                if dist < min_dist:
                    min_dist = dist
                    nearest_land = land_point
        
        # Identify the nearest land-based location and its country
        nearest_lat, nearest_lon = nearest_land.y, nearest_land.x
        location = geolocator.reverse((nearest_lat, nearest_lon), language='en')
        country = location.raw.get('address', {}).get('country', 'Unknown')
        
        if plot_map:
            # Expand the bounding box further to show the nearest land
            expanded_bbox = [
                min(expanded_bbox[0], nearest_land.x - lon_margin),
                max(expanded_bbox[1], nearest_land.x + lon_margin),
                min(expanded_bbox[2], nearest_land.y - lat_margin),
                max(expanded_bbox[3], nearest_land.y + lat_margin),
                ]
            ax.set_extent([expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]])
            ax.scatter(nearest_land.x, nearest_land.y, color='green', label='Nearest Land')
            print(f"The nearest land is approximately {min_dist:.1f} km away at ({nearest_lat:.2f}, {nearest_lon:.2f}) in {country}.")
    
    if plot_map:
        # Add a legend and display the map
        ax.legend()
        plt.title(f'Region of Interest and Expanded Area: ')
        plt.show()
        
    return is_land, country, min_dist

# Example usage
gps_coords = [
    [50.0, -7.0],  # Bottom-left
    [51.0, -7.0],  # Top-left
    [51.0, -6.0],   # Top-right
    [50.0, -6.0]    # Bottom-right
]

gen_map = 1
land_status, nation, distance = get_sea_status(gps_coords, gen_map)
sea_status = ~land_status
if sea_status:
    print(f'ROI {gps_coords} is sea based and {distance:.1f}km off shore of {nation}')
else:
    print(f'ROI {gps_coords} includes land from {nation}')

