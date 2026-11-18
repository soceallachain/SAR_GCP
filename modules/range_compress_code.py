
import logging
import math
import cmath
import struct
import sys
import numpy as np





# Focus a burst or ROI with Range Compression Only
# Assign Sentinel 1 chirp parameters: Unique for each IW swath
# Can be obtained from decoded RAW data OR from SLC annotation file
# samplingFrequencyAfterDecimation, txPulseLength, txPulseStartFrequency, txPulseRampRate
S1IW = np.zeros((3, 4)) 
S1IW[0,:]=[6.434523812571429e+07, 5.240481033595628e-05, -2.825153419637256e+07, 1.078230321255894e+12 ]
S1IW[1,:]=[5.459595962181818e+07, 6.199592966536363e-05, -2.415503791125222e+07, 7.792817275120481e+11]
S1IW[2,:]=[4.691840280000000e+07, 5.339056426703426e-05, -2.139431040091632e+07, 8.014509490705804e+11  ]


## 
# Amended 15th October 2026 to enable range compression of Strip Map data
# Only for swaths 1, 3 & 6 for now
#
S1SM = np.zeros((7,4))
S1SM[0,:]=[100092592.64, 4.5451248822135946e-05, -4.3801344997e+07, 1.927378686406996e+12] #S1
S1SM[1,:]=[0, 0, 0, 0]                                                                  #S2
S1SM[2,:]=[6.67283950933e+07, 4.41724e-05,	-2.97045032241e+07, 1.344932774550995e+12]     #S3
S1SM[3,:]=[0, 0, 0, 0]                                                                  #S4
S1SM[4,:]=[0, 0, 0, 0]                                                                  #S5N
S1SM[5,:]=[0, 0, 0, 0]                                                                  #S5S
S1SM[6,:]=[4.69184028e+07, 5.10993524272e-05, -2.109403648895e+07, 8.2563555441e+11]    #S6


range_sampling_freq_index = 0
pulse_length_index        = 1
pulse_start_freq_index    = 2
pulse_ramp_rate_index     = 3



def range_compress_burst(rd:np.array, satellite='S1IW', swath=2)->np.array: 
     
    #radar_data     = rd 
    range_padded_size    = 4096
    azimuth_padded_size  = 1296
    range_compressed_data = range_compress_burst_with_zero_padding(rd,
                                                  range_padded_size,
                                                  azimuth_padded_size,
                                                  satellite,
                                                  swath
                                                  )
    return range_compressed_data



# Function to combine padding and range compression
# Returns range compressed data with padding removed
def range_compress_padded_raw_data(raw_data,satellite='S1IW', swath=2,
                                   range_padding=1000,
                                   az_padding=200
                                   ):
    
    padded_raw_burst = np.pad(raw_data, ((az_padding,az_padding),
                               (range_padding,range_padding)),
                               mode='constant', constant_values=0
                               )
    rc_padded_burst = range_compress_burst(padded_raw_burst, satellite, 
                                           swath
                                           )
    resized_rc_padded_burst=rc_padded_burst[
                    az_padding:az_padding+raw_data.shape[0],
                    range_padding:range_padding+raw_data.shape[1]
                    ]
    del rc_padded_burst
    rc_padded_burst = resized_rc_padded_burst
    return rc_padded_burst

def range_compress_burst_with_zero_padding(rd:np.array, rzp, azp, 
                                           satellite='S1IW', swath=2
                                           )->np.array: 
     
    #radar_data     = rd 
    range_padded_size    = rzp
    azimuth_padded_size  = azp
    
    
    if satellite =='S1IW':
        range_sample_freq = S1IW[swath-1, range_sampling_freq_index]
        TXPL              = S1IW[swath-1, pulse_length_index]
        TXPSF             = S1IW[swath-1, pulse_start_freq_index]
        TXPRR             = S1IW[swath-1, pulse_ramp_rate_index]
    elif satellite =='S1SM':
        # SM swaths are 1->6 but there are 2 S5 swaths: N & S
        # For S6 swath index is 6 (ignore S5 N and S for now)
        # 
        if swath == 6:
            swath_index = swath
        elif swath ==2 or swath ==4 or swath ==5:
            print(f'Warning: Parameters for SM Swaths 2,4 and 5 unavailable')
            print(f'Exiting function: range_compress_burst_with_zero_padding()')
            return -1

        else:
            swath_index = swath-1
        range_sample_freq = S1SM[swath_index, range_sampling_freq_index]
        TXPL              = S1SM[swath_index, pulse_length_index]
        TXPSF             = S1SM[swath_index, pulse_start_freq_index]
        TXPRR             = S1SM[swath_index, pulse_ramp_rate_index]
        
    elif satellite =='ICEYE':
        range_sample_freq = 0
        TXPL              = 0
        TXPSF             = 0
        TXPRR             = 0

    else:
        print('Error: Satellite not known. Exiting ....')
        sys.exit()
   
    # Test code : remove 
    print(f'#################################################################')
    print(f'TXPL = {TXPL}') 
    print(f'TXPSF = {TXPSF}') 
    print(f'TXPRR = {TXPRR}') 
    print(f'Range Sampling Freq = {range_sample_freq}')
    print(f'Satellite = {satellite}')
    print(f'Swath = {swath}')
    print(f'Swath Index = {swath_index}')
    print(f'#################################################################')

    #Generate chirp replica pulse for convolution with matched filter
    num_tx_vals           = int(TXPL*range_sample_freq) 
    tx_replica_time_vals  = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals) 
    phi1                  = TXPSF + TXPRR*TXPL/2 
    phi2                  = TXPRR/2 
    tx_replica            = np.exp(2j * np.pi 
                                      * (phi1*tx_replica_time_vals 
                                         + phi2*tx_replica_time_vals**2
                                         )
                                   )     
 
    # Check if radar data needs padding to reduce risk of folding
    # NOTE: Check if azimuth padding causes timing issues!!!!!!
    # Ensure padded size is smaller than complete burst size which is:
    # total_burst_size = len(selection)
    az_num_zeroes   = 0
    az_num_zeroes_t = 0
    num_zeroes      = 0
    num_zeroes_r    = 0
    if rd.shape[1] < num_tx_vals or rd.shape[0] < azimuth_padded_size:
        if rd.shape[1] < num_tx_vals:
            num_zeroes   = int((range_padded_size - rd.shape[1])/2)
            num_zeroes_r = num_zeroes + (rd.shape[1] % 2)

        if rd.shape[0] < azimuth_padded_size:
            az_num_zeroes = int((azimuth_padded_size - rd.shape[0])/2)
            az_num_zeroes_t = az_num_zeroes + (rd.shape[0] % 2)


        radar_data = np.pad(rd, pad_width=((az_num_zeroes, az_num_zeroes_t),
                                          (num_zeroes,num_zeroes_r)
                                          ),
                           mode='constant', constant_values=0
                           )

    else:
        radar_data = rd
    
    # Image sizes
    len_range_line = radar_data.shape[1]
    len_az_line    = radar_data.shape[0]
    
    
    # Sample rates
    range_sample_period  = 1/range_sample_freq
    
    # Fast time vector - defines the time axis along the fast time direction
    sample_num_along_range_line = np.arange(0, len_range_line, 1)
    
    # Axes - defines the frequency axes in each direction after FFT
    SWL                  = len_range_line/range_sample_freq
    range_freq_vals      = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)

     
    #print(range_sample_freq) 
     
    #Get posiiton and velocity vectors 
     
    
    #FFT in range direction 
    radar_data = np.fft.fft(radar_data, axis=1) 
     
    # FFT each azimuth line - is this needed for range compression?
    #radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0) 

    # Create range filter from replica pulse 
    range_filter         = np.zeros(len_range_line, dtype=complex) 
    index_start          = np.ceil((len_range_line-num_tx_vals)/2)-1 
    index_end            = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2 
    range_filter[int(index_start):int(index_end+1)] = tx_replica 
    range_filter         = np.conjugate(np.fft.fft(range_filter)) 
     
    # Apply filter 
    radar_data           = np.multiply(radar_data, range_filter) 
     
    #print(f'TXPL, (Pulse Length) = {TXPL} ')
    #print(f' Number of  Pulse samples   = {num_tx_vals}')
    del range_filter 
    del tx_replica 

    radar_data = np.fft.ifftshift(np.fft.ifft(radar_data, axis=1), axes=1)
    #radar_data = np.fft.ifft(radar_data, axis=0) 

    return radar_data[az_num_zeroes:az_num_zeroes + rd.shape[0], 
                      num_zeroes:num_zeroes + rd.shape[1]
                      ] 

