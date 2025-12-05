
"""
Spyder Editor

This is a temporary script file.
"""


# Function to return non-debursted burst boundaries from annotation file
# 
from SAR_code_module import *
"""
def getBurstBoundaries(annotation_file):
    
    tree = etree.parse(annotation_file)
    root = tree.getroot()
    bursts = []
    for burst in root.findall('.//burst'):
        azimuth_time = burst.find('.//azimuthTime').text
        bursts.append({
            'azimuth_time': azimuth_time,
        })
    nLines         = int(root.find('.//numberOfLines').text)
    nSamples       = int(root.find('.//numberOfSamples').text)
    nBursts        = len(bursts)
    nBurstLines    = int(root.find('.//linesPerBurst').text)
    nRangeSamples  = int(root.find('.//samplesPerBurst').text)
    azTimeInterval = float(root.find('.//azimuthTimeInterval').text)
    
    burst_number    = 0
    burstGridPoints = []
    for burstGridPoint in root.findall('.//geolocationGridPoint'):
        azimuth_time = burstGridPoint.find('.//azimuthTime').text
        longitude    = burstGridPoint.find('.//longitude').text
        latitude     = burstGridPoint.find('.//latitude').text
        line         = burstGridPoint.find('.//line').text
        pixel        = burstGridPoint.find('.//pixel').text

        if pixel =='0':
            burst_number +=1

        azimuth_burst_end_time = datetime.strptime(azimuth_time, 
                                                   '%Y-%m-%dT%H:%M:%S.%f') \
                                  + timedelta(seconds 
                                  = (nBurstLines * azTimeInterval))

        burstGridPoints.append({
            'azimuth_time': azimuth_time,
             'longitude': longitude,
             'latitude': latitude,
             'line': line,
             'pixel': pixel,
             'burst_number' : str(burst_number),
             'azimuth_burst_end_time': azimuth_burst_end_time
        })
    #Get Ground Control Points from GeoLocationGridPoints
    df                = pd.DataFrame(burstGridPoints)
    gridPoints        = df[['burst_number','line','pixel',
                            'longitude','latitude']]
    gridControlPoints = gridPoints.to_numpy().astype(float)
    
    pixel_value = '0'
    burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']
    burst_end_time = []
   
    merged_burst_start_line = []
    merged_burst_end_line   = []
    burst_corners           = []
    burst_gps_corners       = []
    gridPoints['merged_burst_start_line']= 0
    gridPoints['merged_burst_end_line']  = 0
    
    burst_ends           = gridPoints[(gridPoints['pixel'] 
                                 == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude',
                                       'latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude',
                                         'latitude']].to_numpy().astype(float)
   
    for i in range(0,nBursts):
        burst_start_line  = gridPoints[gridPoints['burst_number'] 
                                 == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number'] 
                                 == str(i+2)]['line'].to_numpy().astype(int)[0]
        merged_start_line = burst_start_line
        merged_end_line   = burst_end_line -1

        burst_corners.append([(merged_start_line,0),
                              (merged_end_line,0),
                              (merged_end_line,nRangeSamples-1), 
                              (merged_start_line,nRangeSamples-1)]
                              )

        burst_gps_corners.append([burst_start_long_lat[i,:], 
                                  burst_start_long_lat[i+1,:], 
                                  burst_end_long_lat[i+1,:], 
                                  burst_end_long_lat[i,:]]
                                  )

    burst_info=[]
    burst_info.append(burst_corners)
    burst_info.append(burst_gps_corners)
    
    return burst_info

def get_single_non_debursted_slc_burst(slc_file, slc_annotation_file, burst):
    all_burst_data = getBurstBoundaries(slc_annotation_file)
    all_burst_corners = all_burst_data[0]
    burst_boundary = all_burst_corners[burst -1]
    src            = rasterio.open(slc_src_file)
    img            = src.read(1)
    burst_img = create_roi_from_polygon(img, burst_boundary)
    return burst_img

"""
# Test function
raw_file            = '/home/joe/Misc/tempTest/Data/RAW_Data/4bf6c82ef5130dabt/RAW/S1A_IW_RAW__0SDV_20200116T051408_20200116T051441_030821_038933_4F07.SAFE/s1a-iw-raw-s-vh-20200116t051408-20200116t051441-030821-038933.dat'
slc_annotation_file = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200116T051410_20200116T051438_030821_038933_CD1B/SLC/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B.SAFE/annotation/s1a-iw2-slc-vh-20200116t051411-20200116t051437-030821-038933-002.xml'
slc_src_file        = '/home/joe/Misc/tempTest/Data/SLC_Data/S1A_IW_SLC_1SDV_20200116T051410_20200116T051438_030821_038933_CD1B/SLC/S1A_IW_SLC__1SDV_20200116T051410_20200116T051438_030821_038933_CD1B.SAFE/measurement/s1a-iw2-slc-vh-20200116t051411-20200116t051437-030821-038933-002.tiff'   

slc_id= slc_annotation_file.split('/SLC/')[1].split('.SAFE')[0]
burst = 2
swath = 2

burst_image = get_single_non_debursted_slc_burst(slc_src_file, slc_annotation_file, burst)

plot_2d_array(np.abs(burst_image)**2,f'{slc_id} IW{swath} Non De-Bursted Burst#{burst}')

    
