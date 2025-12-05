def getSwathBurstGridGPSBoundaries(annotation_file):

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
    print(f'Number of bursts in {annotation_file} is {nBursts}')
    print(f'Number of azimuth lines in each burst is {nLines}')
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

        azimuth_burst_end_time = datetime.strptime(azimuth_time,'%Y-%m-%dT%H:%M:%S.%f') + timedelta(seconds = (nBurstLines * azTimeInterval))
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
    gridPoints        = df[['burst_number','line','pixel','longitude','latitude']]
    gridControlPoints = gridPoints.to_numpy().astype(float)
    
    pixel_value = '0'
    burst_start_times = df[df['pixel'] == pixel_value]['azimuth_time']
    burst_end_time = []
    number_overlap_lines = []
    for i in range(0,len(burst_start_times)-1):
        # Convert string to datetime object
        t = datetime.strptime(burst_start_times.iloc[i], '%Y-%m-%dT%H:%M:%S.%f')
        # Add burst interval time to start time
        t = t + timedelta(seconds=(nBurstLines*azTimeInterval))
        # Subtract end time of burst i from start of next burst i+1
        overlap_time = t- datetime.strptime(burst_start_times.iloc[i+1], '%Y-%m-%dT%H:%M:%S.%f')
        overlap_lines = round(overlap_time.total_seconds() / azTimeInterval)
        # Convert back to string if necessary (optional)
        burst_end_time.append(t)
        number_overlap_lines.append(overlap_lines)

    overlap_offset_lines = [x//2 for x in number_overlap_lines]
    merged_burst_start_line = []
    merged_burst_end_line   = []
    burst_corners           = []
    burst_gps_corners       = []
    gridPoints['merged_burst_start_line']= 0
    gridPoints['merged_burst_end_line']  = 0
    
    burst_ends           = gridPoints[(gridPoints['pixel'] == str(nRangeSamples-1))]
    burst_starts         = gridPoints[(gridPoints['pixel'] == str(0))]
    burst_end_long_lat   = burst_ends[['longitude','latitude']].to_numpy().astype(float)
    burst_start_long_lat = burst_starts[['longitude','latitude']].to_numpy().astype(float)
    
    burst_grid_corners =[]
    burst_grid_gps     =[]

    for i in range(0,nBursts):
        burst_start_line  = gridPoints[gridPoints['burst_number'] == str(i+1)]['line'].to_numpy().astype(int)[0]
        burst_end_line    = gridPoints[gridPoints['burst_number'] == str(i+2)]['line'].to_numpy().astype(int)[0]
        if i==0:
            merged_start_line = burst_start_line
            merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
        else:
            merged_start_line = burst_start_line + round(overlap_offset_lines[i-1])
            merged_end_line   = burst_end_line   - round(overlap_offset_lines[i])
            
        gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_start_line'] = merged_start_line
        gridPoints.loc[gridPoints['burst_number'] == str(i+1), 'merged_burst_end_line']   = merged_end_line
        burst_corners.append([(merged_start_line,0),(merged_end_line,0),(merged_end_line,nRangeSamples-1), (merged_start_line,nRangeSamples-1)])
        burst_gps_corners.append([burst_start_long_lat[i,:], burst_start_long_lat[i+1,:], burst_end_long_lat[i+1,:], burst_end_long_lat[i,:]])

        ###### Code for sub-grids

        boi1           = gridPoints[(gridPoints['burst_number'] == str(i+1))]
        boi2           = gridPoints[(gridPoints['burst_number'] == str(i+2))]
        long_lat_start = np.array(boi1[['longitude','latitude']].to_numpy().astype(float))
        long_lat_end   = np.array(boi2[['longitude','latitude']].to_numpy().astype(float))
        range_col      = bboi1[['pixel']].to_numpy().astype(int)
        grid_corners   = []
        grid_gps       = []
        for j in range(0,len(boi1)-1):
            grid_corners.append([(merged_start_line,range_col[j]), (merged_end_line, range_col[j]), (merged_end_line, range_col[j+1]), (merged_start_line, range_col[j+1])])
            grid_gps.append([long_lat_start[j,:], long_lat_end[j,:], long_lat_end[j+1,:], long_lat_start[j+1,:]])

        burst_grid_corners.append(grid_corners)
        burst_grid_gps.append(grid_gps)


    #Define boundaries of SLC file 
    file_corners = [(0,0),(nBurstLines*nBursts-1,0),(nBurstLines*nBursts-1,nRangeSamples-1),(0,nRangeSamples-1)]
    #gps corner locations for ROI of IW swath - Note values are strings
    swath_gps_corners = find_coordinates_array(df, file_corners)
    print(f'GPS corners of SLC file are {gps_corners}')
    return swath_gps_corners, burst_corners, burst_gps_corners, burst_grid_corners, burst_grid_gps

