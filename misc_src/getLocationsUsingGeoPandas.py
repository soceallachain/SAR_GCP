import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

def get_nearest_city(point, cities_gdf):
    nearest_city = None
    min_dist = float('inf')

    for idx, city in cities_gdf.iterrows():
        dist = point.distance(city.geometry)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city

    return nearest_city['name'] if nearest_city is not None else "Unknown"

def get_location_info(latitude, longitude, countries_gdf, cities_gdf):
    point = Point(longitude, latitude)
    nearest_country = countries_gdf[countries_gdf.contains(point)]['NAME'].values[0]

    if nearest_country:
        nearest_city = get_nearest_city(point, cities_gdf)
        return nearest_country, nearest_city
    else:
        return None, None

def get_distance_to_shore(latitude, longitude, countries_gdf):
    point = Point(longitude, latitude)
    nearest_country = countries_gdf.loc[countries_gdf.distance(point).idxmin()]['NAME']
    distance_to_shore = point.distance(nearest_country.boundary)

    return nearest_country, distance_to_shore

def main():
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))

    country, nearest_city = get_location_info(latitude, longitude, world, cities)

    if country:
        print(f"The coordinates {latitude}, {longitude} are in {country}.")
        if nearest_city != "Unknown":
            print(f"The nearest urban center is {nearest_city}.")
        else:
            print("No urban center found nearby.")
    else:
        nearest_country, distance_to_shore = get_distance_to_shore(latitude, longitude, world)
        print(f"The coordinates {latitude}, {longitude} are in the ocean.")
        print(f"The nearest country is {nearest_country}, approximately {distance_to_shore:.2f} degrees from shore.")

if __name__ == "__main__":
    main()
