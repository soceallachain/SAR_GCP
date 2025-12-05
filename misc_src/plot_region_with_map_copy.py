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

def plot_region_and_identify_location(gps_coords):
    # Convert GPS coordinates to a polygon
    roi_polygon = Polygon(gps_coords)
    
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
        
        # Expand the bounding box further to show the nearest land
        expanded_bbox = [
            min(expanded_bbox[0], nearest_land.x - lon_margin),
            max(expanded_bbox[1], nearest_land.x + lon_margin),
            min(expanded_bbox[2], nearest_land.y - lat_margin),
            max(expanded_bbox[3], nearest_land.y + lat_margin),
        ]
        ax.set_extent([expanded_bbox[0], expanded_bbox[1], expanded_bbox[2], expanded_bbox[3]])
        ax.scatter(nearest_land.x, nearest_land.y, color='green', label='Nearest Land')
        print(f"The nearest land is approximately {min_dist:.2f} km away at {nearest_land.y:.2f}, {nearest_land.x:.2f}.")
    
    # Add a legend and display the map
    ax.legend()
    plt.title('Region of Interest and Expanded Area')
    plt.show()

# Example usage
gps_coords = [
    (-10.0, 50.0),  # Bottom-left
    (-9.0, 50.0),   # Bottom-right
    (-9.0, 51.0),   # Top-right
    (-10.0, 51.0)   # Top-left
]
plot_region_and_identify_location(gps_coords)

