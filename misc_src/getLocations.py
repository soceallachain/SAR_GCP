from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut


def get_location_info(latitude, longitude):
    geolocator = Nominatim(user_agent="geo_locator")

    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        if location is None:
            return None, None

        country = location.raw['address']['country']
        if 'town' in location.raw['address']:
            nearest_city = location.raw['address']['town']
        elif 'city' in location.raw['address']:
            nearest_city = location.raw['address']['city']
        else:
            nearest_city = "Unknown"

        return country, nearest_city

    except (GeocoderTimedOut, KeyError):
        return None, None


def get_distance_to_shore(latitude, longitude):
    geolocator = Nominatim(user_agent="geo_locator")
    
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        if location is None:
            return None, None

        country = location.raw['address']['country']

        # Get the coordinates of the nearest point on the coastline
        shore_location = geolocator.geocode(country, exactly_one=False)
        if shore_location is None:
            return None, None

        shore_latitude, shore_longitude = shore_location.latitude, shore_location.longitude

        # Calculate the distance from the input location to the nearest point on the coastline
        distance_to_shore = geodesic((latitude, longitude), (shore_latitude, shore_longitude)).kilometers

        return country, distance_to_shore

    except (GeocoderTimedOut, KeyError):
        return None, None
def main():
    latitude = float(input("Enter latitude: "))
    longitude = float(input("Enter longitude: "))

    country, nearest_city = get_location_info(latitude, longitude)

    if country:
        print(f"The coordinates {latitude}, {longitude} are in {country}.")
        if nearest_city != "Unknown":
            print(f"The nearest urban center is {nearest_city}.")
        else:
            print("No urban center found nearby.")
    else:
        nearest_country, distance_to_shore = get_distance_to_shore(latitude, longitude)
        if nearest_country and distance_to_shore:
            print(f"The coordinates {latitude}, {longitude} are in the ocean.")
            print(f"The nearest country is {nearest_country}, approximately {distance_to_shore:.2f} kilometers from shore.")
        else:
            print("Failed to determine location.")


if __name__ == "__main__":
    main()
