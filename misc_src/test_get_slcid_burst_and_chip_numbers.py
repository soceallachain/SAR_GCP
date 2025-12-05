import re
import pandas as pd

# Given filename
filename = "S1A_IW_SLC__1SDV_20200221T051410_20200221T051437_031346_039B7B_EB4B_RAW_IW2_imag_burst#3_chip#3_#Ships5_SeaStatusTrue_H5M0L0.npy"
csv_filename = '/home/joe/Work/Projects/RSAR/Datasets/xView3/Info/SARFishLabels/SARFish_RAW_Public_BurstChipInfoShipsPerChipWithShipLocations.csv'




def get_ship_locations(chip_information, csv_file):
    df           = pd.read_csv(csv_file)    
    slc_id       = chip_information[0]
    burst_number = chip_information[1]
    chip_number  = chip_information[2]
    # Filter rows based on extracted parameters
    filtered_df  = df[(df['Product_ID'] == slc_id) &
                  (df['Burst_Number'] == burst_number) &
                  (df['Chip_Number'] == chip_number)]
    # Extract relevant columns
    relevant_columns = ['gps_lon', 'gps_lat', 'ship_burst_row', 'ship_burst_col', 'ship_chip_row', 'ship_chip_col', 'ship_length']
    extracted_data = filtered_df[relevant_columns]
    return extracted_data



def get_slc_id_burst_and_chip_numbers(filename):

    # Extract SLC_ID, burst_number, chip_number, and number_of_ships
    slc_id_match = re.match(r'(S1A_IW_SLC__1SDV_\d{8}T\d{6}_\d{8}T\d{6}_\d{6}_\w{6}_\w{4})', filename)
    slc_id = slc_id_match.group(1) if slc_id_match else None

    burst_number_match = re.search(r'burst#(\d+)', filename)
    burst_number = int(burst_number_match.group(1)) if burst_number_match else None

    chip_number_match = re.search(r'chip#(\d+)', filename)
    chip_number = int(chip_number_match.group(1)) if chip_number_match else None
    return slc_id, burst_number, chip_number



chip_info=get_slc_id_burst_and_chip_numbers(file)
print(f'SLC_ID = {chip_info[0]}')
print(f'burst# = {chip_info[1]}')
print(f'chip# = {chip_info[2]}')


ship_locations = get_ship_locations(chip_info, csv_filename)
print(ship_locations)



