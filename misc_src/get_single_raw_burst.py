
from SAR_code_module import *


def get_single_raw_burst_from_files(raw_file, slc_annotation_file, burst_to_get, swath_of_interest=2, polarization='vh'):

##
# Function to return a raw burst based on the SLC burst number
# Requires access to raw data file and SLC annotation file
# Assumes these are in standard RAW and SLC directories
#
##    

IW_mode         = swath_of_interest - 1
raw_filename    = raw_file
annotation_file = slc_annotation_file
if not (os.path.exists(raw_filename)):
    print(f'Error: Filename {raw_filename} not found.Exiting!')
    sys.exit(0) 
else:
    raw_iw_data       = get_raw_burst_info(raw_filename)
    print(f' File {raw_filename} found. ')

if (os.path.exists(annotation_file): 
    print(f'{annotation_file} found ')
else:
    print(f'ERROR: {annotation_file}  Not found. Exiting')
    sys.exit(0)
#Check RAW and SLC annotation 
if not (polarization in annotation_file and polarization in raw_filename and 
        (('iw1' in annotation_file and IW_mode==0) or
         ('iw2' in annotation_file and IW_mode==1) or
         ('iw3' in annotation_file and IW_mode==2))):
    print('ERROR: IW mode of SLC annotation file does not match expected swath ')
    print('Exiting ....')
    sys.exit(0)

# Have SLC annotation file and raw filename
slc_iw_burst_data = []
slc_iw_data       = []
#Get timing positions of slc bursts to enable matching with raw times
pre_post_burst_information = get_pre_post_mid_times(annotation_file)
slc_iw_data.append(pre_post_burst_information)
#Get boundaries for each swath, burst &  grid in [row,col] & GPS coordinates 
burst_data        = getSwathBurstGridGPSBoundaries(annotation_file)
iw_burst_data.append(burst_data)

iw_mode = IW_mode
tree = etree.parse(annotation_file)
root = tree.getroot()
slc_range_sampling_freq = float(root.find('.//rangeSamplingRate').text)

bursts_not_to_save = []
for slc_burst in [burst_to_get]:
    burst_of_interest = str(slc_burst)
    matching_burst    = slc_iw_data[0][slc_iw_data[0]['burst_number']==burst_of_interest]
    if not matching_burst.empty:
        index_of_burst  = find_raw_start_burst(raw_iw_data[iw_mode], matching_burst)
        selected_burst  = index_of_burst
        l0file          = sentinel1decoder.Level0File(raw_filename)
        raw_burst_info  = l0file.get_burst_metadata(selected_burst)
        radar_data      = l0file.get_burst_data(index_of_burst)
    else:
        print(f'WARNING: No matching bursts for burst #{burst_of_interest} in {slc_src_file}')

return radar_data


# Test function
raw_file_name = '/home/joe/Misc/tempTest/Data/RAW_Data/4bf6c82ef5130dabt/RAW/S1A_IW_RAW__0SDV_20200116T051408_20200116T051441_030821_038933_4F07.SAFE/s1a-iw-raw-s-vh-20200116t051408-20200116t051441-030821-038933.dat'
slc_annotation_file_name ='/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200116T051410_20200116T051438_030821_038933_CD1B/SLC/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B.SAFE/annotation/s1a-iw3-slc-vh-20200116t051412-20200116t051438-030821-038933-003.xml'
swath = 2
burst = 3

raw_data = get_single_raw_birst_from_files(raw_file_name, slc_annotation_file_name, burst, swath)
plot_2d_array(np.real(raw_data), raw_file_name, 'linear')
