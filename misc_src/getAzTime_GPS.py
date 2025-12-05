from lxml import etree

def parse_annotation_file(annotation_file):
    """Parse the annotation XML file to extract burst information."""
    tree = etree.parse(annotation_file)
    root = tree.getroot()

    # Namespaces required to access the elements properly
    namespaces = {
        's1': 'http://www.esa.int/safe/sentinel-1.0'
    }

    bursts_info = []

    for burst in root.findall('.//s1:burst', namespaces):
        azimuth_start_time = burst.find('.//s1:azimuthTime', namespaces).text
        
        # Extract corner coordinates (assume 4 corners per burst)
        corners = burst.findall('.//s1:geolocationGridPoint', namespaces)
        corner_coordinates = []
        for corner in corners:
            latitude = float(corner.find('s1:latitude', namespaces).text)
            longitude = float(corner.find('s1:longitude', namespaces).text)
            corner_coordinates.append((latitude, longitude))
        
        bursts_info.append({
            'azimuth_start_time': azimuth_start_time,
            'corner_coordinates': corner_coordinates
        })

    return bursts_info

def main(annotation_file):
    """Main function to process the annotation file and print burst info."""
    bursts_info = parse_annotation_file(annotation_file)

    for i, burst_info in enumerate(bursts_info):
        print(f"Burst {i + 1}:")
        print(f"  Azimuth Start Time: {burst_info['azimuth_start_time']}")
        print(f"  Corner Coordinates:")
        for j, (lat, lon) in enumerate(burst_info['corner_coordinates']):
            print(f"    Corner {j + 1}: Latitude = {lat}, Longitude = {lon}")
        print()

if __name__ == "__main__":
    # Example usage:
    annotation_file = 'path_to_your_annotation_file.xml'
    main(annotation_file)
